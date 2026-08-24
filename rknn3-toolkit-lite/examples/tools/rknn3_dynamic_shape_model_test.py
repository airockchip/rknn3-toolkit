#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RKNN3Lite dynamic-shape model test.

One invocation runs only one dynamic-shape group. The selected shape group is
inferred automatically from the shapes of ``--input_npy_paths``; no shape_id is
passed to ``inference``.

Examples:
    # Single-input model
    python rknn3_dynamic_model_test.py \
        --rknn_path model.rknn \
        --input_npy_paths input.npy

    # Multi-input model: all npy files belong to one dynamic-shape group
    python rknn3_dynamic_model_test.py \
        --rknn_path model.rknn \
        --input_npy_paths 'input0.npy#input1.npy#input2.npy' \
        --golden_npy_paths 'golden0.npy#golden1.npy' \
        --loop_count 10
"""

import argparse
import os
import sys
import time
from pprint import pformat
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rknn3lite.api import RKNN3Lite
from rknn3lite.api.rknn3_types import (
    RKNN3QueryCmd,
    RKNN3TensorQntType,
    rknn3_get_layout_string,
    rknn3_get_type_string,
)


DEFAULT_LOOP_COUNT = 1


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a64 = np.asarray(a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    dot = np.dot(a64, b64)
    norm_a = np.linalg.norm(a64)
    norm_b = np.linalg.norm(b64)
    if norm_a < 1e-30 or norm_b < 1e-30:
        return 0.0
    return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))


def print_data_values(data: np.ndarray, count: int = 10) -> None:
    """Print the first values of a NumPy array."""
    flat = np.asarray(data).reshape(-1)
    for index in range(min(count, flat.size)):
        print(f"  [{index}] {flat[index]}")


def split_paths(value: Optional[str]) -> List[str]:
    """Split command-line paths separated by '#'."""
    if not value:
        return []
    return [item.strip() for item in value.split("#") if item.strip()]


def load_npy_files(
    paths: Sequence[str], kind: str, verbose: bool = False
) -> List[np.ndarray]:
    """Load npy files and ensure the returned arrays are contiguous."""
    arrays: List[np.ndarray] = []
    for index, path in enumerate(paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{kind}[{index}] file not found: {path}")
        array = np.load(path, allow_pickle=False)
        array = np.ascontiguousarray(array)
        arrays.append(array)
        if verbose:
            print(
                f"Loaded {kind}[{index}]: {path}, "
                f"shape={list(array.shape)}, dtype={array.dtype}, "
                f"contiguous={array.flags.c_contiguous}"
            )
    return arrays


def normalize_shape(shape: Sequence[Any]) -> Tuple[int, ...]:
    return tuple(int(value) for value in shape)


def input_shape_match_mode(
    npy_shape: Sequence[int], model_shape: Sequence[int]
) -> Optional[str]:
    """Return how a user npy shape matches a dynamic model input shape.

    RKNN3Lite dynamic metadata may expose a 4-D model input as NCHW while the
    normal user input accepted by inference is NHWC. Therefore both exact shape
    matching and the common NCHW<->NHWC representation are accepted.
    """
    user = normalize_shape(npy_shape)
    model = normalize_shape(model_shape)

    if user == model:
        return "exact"

    if len(user) == 4 and len(model) == 4:
        model_as_nhwc = (model[0], model[2], model[3], model[1])
        if user == model_as_nhwc:
            return "user-NHWC/model-NCHW"

        model_as_nchw = (model[0], model[3], model[1], model[2])
        if user == model_as_nchw:
            return "user-NCHW/model-NHWC"

    return None


def get_entry_input_shapes(entry: Dict[str, Any]) -> List[Tuple[int, ...]]:
    inputs = entry.get("inputs", [])
    result: List[Tuple[int, ...]] = []
    for index, input_info in enumerate(inputs):
        if "shape" not in input_info:
            raise KeyError(
                f"Dynamic shape entry {entry.get('shape_id', '<unknown>')} "
                f"input[{index}] has no 'shape': {input_info}"
            )
        result.append(normalize_shape(input_info["shape"]))
    return result


def match_dynamic_shape_entry(
    shape_infos: Sequence[Dict[str, Any]], input_arrays: Sequence[np.ndarray]
) -> Tuple[Dict[str, Any], List[str]]:
    """Find the unique dynamic shape entry matching this one group of inputs."""
    matches: List[Tuple[Dict[str, Any], List[str]]] = []

    for entry in shape_infos:
        model_shapes = get_entry_input_shapes(entry)
        if len(model_shapes) != len(input_arrays):
            continue

        modes: List[str] = []
        matched = True
        for array, model_shape in zip(input_arrays, model_shapes):
            mode = input_shape_match_mode(array.shape, model_shape)
            if mode is None:
                matched = False
                break
            modes.append(mode)

        if matched:
            matches.append((entry, modes))

    if len(matches) == 1:
        return matches[0]

    input_shapes = [list(array.shape) for array in input_arrays]
    available = [
        {
            "shape_id": entry.get("shape_id", "<unknown>"),
            "input_shapes": [list(shape) for shape in get_entry_input_shapes(entry)],
        }
        for entry in shape_infos
    ]

    if not matches:
        raise RuntimeError(
            "No dynamic shape group matches the input npy shapes.\n"
            f"Input npy shapes: {input_shapes}\n"
            f"Available dynamic shapes:\n{pformat(available, sort_dicts=False)}"
        )

    ambiguous = [
        {
            "shape_id": entry.get("shape_id", "<unknown>"),
            "match_modes": modes,
            "input_shapes": [list(shape) for shape in get_entry_input_shapes(entry)],
        }
        for entry, modes in matches
    ]
    raise RuntimeError(
        "The input npy shapes match more than one dynamic shape group.\n"
        f"Input npy shapes: {input_shapes}\n"
        f"Matched entries:\n{pformat(ambiguous, sort_dicts=False)}"
    )


def first_device_id(rknn_lite: RKNN3Lite) -> Optional[str]:
    try:
        device_ids = rknn_lite.get_devices_id()
    except Exception as exc:  # Device enumeration is optional.
        print(f"Warning: get_devices_id failed: {exc}")
        return None

    if isinstance(device_ids, (list, tuple)) and device_ids:
        return device_ids[0]
    return None


def _decode_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _enum_short_name(value: Any) -> str:
    """Convert RKNN enum values to compact readable names."""
    name = getattr(value, "name", None)
    text = str(name if name is not None else value)
    for prefix in (
        "RKNN3_TENSOR_QNT_",
        "RKNN3_TENSOR_",
        "RKNN3_LAYOUT_",
        "RKNN3_",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    aliases = {
        "FLOAT16": "FP16",
        "FLOAT32": "FP32",
        "BFLOAT16": "BF16",
        "QNT_NONE": "NONE",
        "TENSOR_QNT_NONE": "NONE",
    }
    return aliases.get(text, text)


def _layout_short_name(value: Any) -> str:
    try:
        return _enum_short_name(rknn3_get_layout_string(value))
    except Exception:
        return _enum_short_name(value)


def _dtype_short_name(value: Any) -> str:
    try:
        return _enum_short_name(rknn3_get_type_string(value))
    except Exception:
        return _enum_short_name(value)


def _qnt_short_name(value: Any) -> str:
    for name in dir(RKNN3TensorQntType):
        if not name.startswith("RKNN3_TENSOR_QNT_"):
            continue
        try:
            if value == getattr(RKNN3TensorQntType, name):
                return name.replace("RKNN3_TENSOR_QNT_", "")
        except Exception:
            continue
    return _enum_short_name(value)


def _to_int_list(value: Any, count: Optional[int] = None) -> List[int]:
    if value is None:
        return []
    try:
        values = list(value)
    except TypeError:
        return []
    if count is not None:
        values = values[:count]
    return [int(item) for item in values]


def _entry_is_default(entry: Dict[str, Any], index: int) -> bool:
    for key in ("is_default", "default", "isDefault"):
        if key in entry:
            return bool(entry[key])
    return index == 0


def _entry_input_name(input_info: Dict[str, Any], index: int) -> str:
    return _decode_name(input_info.get("name", f"input_{index}"))


def _entry_aligned_size(input_info: Dict[str, Any]) -> Optional[int]:
    for key in ("aligned_size", "size_with_stride", "alignedSize", "size"):
        value = input_info.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def get_default_shape_id(shape_infos: Sequence[Dict[str, Any]]) -> int:
    for index, entry in enumerate(shape_infos):
        if _entry_is_default(entry, index):
            return int(entry.get("shape_id", index))
    return int(shape_infos[0].get("shape_id", 0))


def get_current_shape_id(
    rknn_lite: RKNN3Lite, shape_infos: Sequence[Dict[str, Any]]
) -> int:
    """Best-effort query of the runtime shape ID, with metadata fallback."""
    for method_name in ("get_current_shape_id", "get_dynamic_shape_id"):
        method = getattr(rknn_lite, method_name, None)
        if callable(method):
            try:
                value = method()
                if value is not None:
                    return int(value)
            except Exception:
                pass

    for attr_name in ("current_shape_id", "active_shape_id"):
        value = getattr(rknn_lite, attr_name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    for index, entry in enumerate(shape_infos):
        if bool(entry.get("is_current", False)) or bool(entry.get("active", False)):
            return int(entry.get("shape_id", index))
    return get_default_shape_id(shape_infos)


def print_dynamic_shape_summary(
    shape_infos: Sequence[Dict[str, Any]], current_shape_id: int
) -> None:
    print(f"Current shape ID: {current_shape_id}")
    for index, entry in enumerate(shape_infos):
        shape_id = int(entry.get("shape_id", index))
        default_suffix = " [Default]" if _entry_is_default(entry, index) else ""
        print(f"Shape {index} (ID: {shape_id}){default_suffix}:")
        for input_index, input_info in enumerate(entry.get("inputs", [])):
            name = _entry_input_name(input_info, input_index)
            shape = _to_int_list(input_info.get("shape"))
            aligned_size = _entry_aligned_size(input_info)
            size_text = (
                f"{aligned_size} bytes" if aligned_size is not None else "unknown"
            )
            print(
                f"  Input {input_index} ({name}): {shape} "
                f"Aligned size: {size_text}"
            )


def print_tensor_attr(attr: Any, index: int, kind: str) -> None:
    n_dims = int(getattr(attr, "n_dims", 0))
    shape = _to_int_list(getattr(attr, "shape", None), n_dims)
    stride = _to_int_list(getattr(attr, "stride", None), n_dims)
    name = _decode_name(getattr(attr, "name", f"{kind}_{index}"))
    core_id = int(getattr(attr, "core_id", 0))
    aligned_size = getattr(attr, "aligned_size", None)
    if aligned_size is None:
        aligned_size = getattr(attr, "size_with_stride", None)
    aligned_text = str(int(aligned_size)) if aligned_size is not None else "unknown"
    layout = _layout_short_name(getattr(attr, "layout", "UNKNOWN"))
    dtype = _dtype_short_name(getattr(attr, "dtype", "UNKNOWN"))
    qnt_type = _qnt_short_name(getattr(attr, "qnt_type", "NONE"))
    scale = float(getattr(attr, "scale", 1.0))
    zero_point = int(getattr(attr, "zero_point", getattr(attr, "zp", 0)))

    print(
        f"  name={name},core_id={core_id}, n_dims={n_dims}, "
        f"shape={shape}, stride={stride}"
    )
    print(
        f"  aligned_size={aligned_text}, layout={layout}, dtype={dtype}"
    )
    print(
        f"  qnt_type={qnt_type} scale={scale:.6f}, zero_point={zero_point}"
    )


def query_and_print_tensor_attrs(
    rknn_lite: RKNN3Lite, n_input: int, n_output: int
) -> None:
    """Print compact tensor attributes for the active dynamic shape."""
    print("input tensors:")
    for index in range(n_input):
        attr = rknn_lite.rknn3_query(
            RKNN3QueryCmd.RKNN3_QUERY_INPUT_ATTR, index=index
        )
        if attr is None:
            print(f"  [{index}] query failed")
            continue
        print_tensor_attr(attr, index, "input")

    print("output tensors:")
    for index in range(n_output):
        attr = rknn_lite.rknn3_query(
            RKNN3QueryCmd.RKNN3_QUERY_OUTPUT_ATTR, index=index
        )
        if attr is None:
            print(f"  [{index}] query failed")
            continue
        print_tensor_attr(attr, index, "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RKNN3 dynamic-shape model test. One invocation accepts one group "
            "of input npy files and runs only the matching dynamic shape."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rknn_path", type=str, required=True, help="RKNN model path (.rknn)"
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="RKNN weight path. Default: replace .rknn with .weight",
    )
    parser.add_argument(
        "--input_npy_paths",
        type=str,
        required=True,
        help=(
            "One dynamic-shape group's input npy paths, separated by '#'. "
            "The number and order must match model inputs."
        ),
    )
    parser.add_argument(
        "--golden_npy_paths",
        type=str,
        default=None,
        help="Optional golden output npy paths, separated by '#'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory used to save output_data_N.npy (default: current directory)",
    )
    parser.add_argument(
        "--core_mask",
        type=lambda value: int(value, 0),
        default=None,
        help="NPU core mask, for example 0x01 or 0x0f. Default: use all cores",
    )
    parser.add_argument(
        "--loop_count",
        type=int,
        default=DEFAULT_LOOP_COUNT,
        help=f"Number of inference loops (default: {DEFAULT_LOOP_COUNT})",
    )
    parser.add_argument(
        "--target", type=str, default="rk1820", help="Target device (default: rk1820)"
    )
    parser.add_argument(
        "--device_id",
        type=str,
        default=None,
        help="Optional device id. Default: use the first enumerated device",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print input/output values, per-loop timing and detailed comparisons",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.loop_count <= 0:
        print("Error: --loop_count must be greater than 0", file=sys.stderr)
        return 1

    model_path = args.rknn_path
    weight_path = args.weight_path or (os.path.splitext(model_path)[0] + ".weight")
    input_files = split_paths(args.input_npy_paths)
    golden_files = split_paths(args.golden_npy_paths)

    if not os.path.isfile(model_path):
        print(f"Error: RKNN model not found: {model_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(weight_path):
        print(f"Error: RKNN weight not found: {weight_path}", file=sys.stderr)
        return 1
    if not input_files:
        print("Error: no input npy files were provided", file=sys.stderr)
        return 1

    try:
        input_arrays = load_npy_files(input_files, "input", args.verbose)
        golden_arrays = (
            load_npy_files(golden_files, "golden", args.verbose)
            if golden_files else []
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nRKNN3 Dynamic Model Test (Python)")
    rknn_lite = RKNN3Lite()

    try:
        print("--> Load RKNN model")
        ret = rknn_lite.load_rknn(
            model_path=model_path,
            weight_path=weight_path,
        )
        if ret != 0:
            print(f"Load RKNN model failed: {ret}", file=sys.stderr)
            return ret
        print("done")

        device_id = args.device_id or first_device_id(rknn_lite)
        initial_core_mask = args.core_mask if args.core_mask is not None else 0x01

        init_kwargs: Dict[str, Any] = {
            "target": args.target,
            "core_mask": initial_core_mask,
        }
        if device_id:
            init_kwargs["device_id"] = device_id

        print("--> Init runtime environment")
        ret = rknn_lite.init_runtime(**init_kwargs)
        if ret != 0:
            print(f"Init runtime environment failed: {ret}", file=sys.stderr)
            return ret
        print("done")

        dev_mem_info = rknn_lite.rknn3_query(
            RKNN3QueryCmd.RKNN3_QUERY_DEVICE_MEM_INFO
        )
        if dev_mem_info is not None:
            print(
                "Device Memory Info: "
                f"total={dev_mem_info.sys_total // (1024 * 1024)} MB, "
                f"free={dev_mem_info.sys_free // (1024 * 1024)} MB"
            )
            if args.verbose:
                for index in range(dev_mem_info.node_num):
                    node = dev_mem_info.node_mem_info[index]
                    print(
                        f"  Node {index}: total={node.total // (1024 * 1024)} MB, "
                        f"free={node.free // (1024 * 1024)} MB"
                    )

        core_num = rknn_lite.rknn3_query(
            RKNN3QueryCmd.RKNN3_QUERY_CORE_NUMBER
        )
        if core_num is None:
            print("Warning: failed to query core number, keeping current core_mask")
        else:
            core_num = int(core_num)
            print(f"Core number: {core_num}")

        # Follow rknn3_model_test.py: when no mask is supplied, reinitialize with
        # a mask containing all available cores.
        if args.core_mask is None and core_num is not None:
            final_core_mask = (1 << core_num) - 1
            print(
                f"Auto-generated core_mask: 0x{final_core_mask:x} "
                f"for {core_num} cores"
            )

            if final_core_mask != initial_core_mask:
                rknn_lite.release()
                rknn_lite = RKNN3Lite()

                ret = rknn_lite.load_rknn(
                    model_path=model_path,
                    weight_path=weight_path,
                )
                if ret != 0:
                    print(f"Reload RKNN model failed: {ret}", file=sys.stderr)
                    return ret

                reinit_kwargs: Dict[str, Any] = {
                    "target": args.target,
                    "core_mask": final_core_mask,
                }
                if device_id:
                    reinit_kwargs["device_id"] = device_id

                ret = rknn_lite.init_runtime(**reinit_kwargs)
                if ret != 0:
                    print(
                        f"Re-init runtime environment failed: {ret}",
                        file=sys.stderr,
                    )
                    return ret

        io_num = rknn_lite.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
        if io_num is None:
            print("Failed to query model IO number", file=sys.stderr)
            return 1
        n_input = int(io_num.n_input)
        n_output = int(io_num.n_output)
        print(f"Model input num: {n_input}, output num: {n_output}")

        if len(input_arrays) != n_input:
            print(
                f"Error: got {len(input_arrays)} input npy files, "
                f"but the model has {n_input} inputs",
                file=sys.stderr,
            )
            return 1

        if golden_arrays and len(golden_arrays) != n_output:
            print(
                f"Error: got {len(golden_arrays)} golden npy files, "
                f"but the model has {n_output} outputs",
                file=sys.stderr,
            )
            return 1

        shape_infos = rknn_lite.get_dynamic_shape_infos()
        if not shape_infos:
            print(
                "Error: the supplied model has no dynamic shape information",
                file=sys.stderr,
            )
            return 1

        current_shape_id = get_current_shape_id(rknn_lite, shape_infos)
        print_dynamic_shape_summary(shape_infos, current_shape_id)
        try:
            selected_entry, match_modes = match_dynamic_shape_entry(
                shape_infos, input_arrays
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        selected_shape_id = int(selected_entry.get("shape_id", 0))
        if args.verbose:
            print(f"Match modes: {match_modes}")
            for index, array in enumerate(input_arrays):
                print(
                    f"Input[{index}]: shape={list(array.shape)}, "
                    f"dtype={array.dtype}, elements={array.size}"
                )
                print_data_values(array, count=10)

        print(f"\nRunning shape ID {selected_shape_id}, loops={args.loop_count}...")
        all_outputs: Optional[List[np.ndarray]] = None
        run_times_ms: List[float] = []

        for loop in range(args.loop_count):
            start = time.perf_counter()
            outputs = rknn_lite.inference(inputs=input_arrays)
            end = time.perf_counter()

            if outputs is None:
                print(
                    f"RKNN3 inference failed at loop {loop + 1}",
                    file=sys.stderr,
                )
                return 1

            loop_ms = (end - start) * 1000.0
            run_times_ms.append(loop_ms)
            if args.verbose:
                print(f"  loop {loop + 1}: {loop_ms:.3f} ms")
            all_outputs = outputs

        print(
            f"Inference: loops={args.loop_count}, "
            f"avg={sum(run_times_ms) / len(run_times_ms):.3f} ms, "
            f"min={min(run_times_ms):.3f} ms, max={max(run_times_ms):.3f} ms"
        )
        print(f"Active shape ID set to: {selected_shape_id}")
        query_and_print_tensor_attrs(
            rknn_lite, n_input=n_input, n_output=n_output
        )

        if all_outputs is None:
            print("Error: inference produced no outputs", file=sys.stderr)
            return 1
        if len(all_outputs) != n_output:
            print(
                f"Error: inference returned {len(all_outputs)} outputs, "
                f"but the model reports {n_output}",
                file=sys.stderr,
            )
            return 1

        for index, output in enumerate(all_outputs):
            output_array = np.asarray(output)
            output_path = os.path.join(
                args.output_dir, f"output_data_{index}.npy"
            )
            np.save(output_path, output_array)
            print(
                f"Saved output[{index}]: {output_path}, "
                f"shape={list(output_array.shape)}, dtype={output_array.dtype}"
            )
            if args.verbose:
                print(f"Output[{index}] first values:")
                print_data_values(output_array, count=10)

            if not golden_arrays:
                continue

            golden = golden_arrays[index]
            output_flat = output_array.astype(np.float32, copy=False).reshape(-1)
            golden_flat = golden.astype(np.float32, copy=False).reshape(-1)

            if output_flat.size != golden_flat.size:
                print(
                    f"Warning: output[{index}] elements={output_flat.size}, "
                    f"golden[{index}] elements={golden_flat.size}; "
                    "cosine similarity uses the common prefix"
                )

            compare_count = min(output_flat.size, golden_flat.size)
            if compare_count == 0:
                print(f"Output {index} cosine similarity: skipped (empty array)")
                continue

            print(f"\nComparing first 10 values of Output {index}:")
            for value_index in range(min(10, compare_count)):
                print(
                    f"  Index[{value_index}]: "
                    f"Output={output_flat[value_index]:.6f}, "
                    f"Golden={golden_flat[value_index]:.6f}"
                )

            print(f"Comparing last 10 values of Output {index}:")
            for value_index in range(max(0, compare_count - 10), compare_count):
                print(
                    f"  Index[{value_index}]: "
                    f"Output={output_flat[value_index]:.6f}, "
                    f"Golden={golden_flat[value_index]:.6f}"
                )

            similarity = cosine_similarity(
                output_flat[:compare_count], golden_flat[:compare_count]
            )
            print(f"Output {index} cosine similarity: {similarity:.6f}")

        print("\ndone")
        return 0

    finally:
        try:
            rknn_lite.release()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())