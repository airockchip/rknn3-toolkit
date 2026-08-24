#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN3 profile ops test implemented with RKNN3Lite public APIs only.

This script intentionally does NOT call low-level rknn3_profile_ops(ctx, inputs, outputs, ...).
It uses RKNN3Lite.load_rknn / init_runtime / inference / profile_mem / profile_ops / release.

Usage compatible with the C++ test style:
  python3 profile_ops_test_lite.py <rknn_path> <weight_path> [input_npy_paths] [golden_output_npy_paths] [core_mask] [log_level] [shape_id] [device_id]

Examples:
  # random input, skip golden comparison
  python3 profile_ops_test_lite.py model.rknn model.weight "" "" 0xff 1 0

  # input and golden npy, separated by '#'
  python3 profile_ops_test_lite.py model.rknn model.weight "input0.npy#input1.npy" "golden0.npy" 0xff 1 0

Named-argument form:
  python3 profile_ops_test_lite.py \
    --rknn-path model.rknn \
    --weight-path model.weight \
    --inputs "" \
    --goldens "" \
    --core-mask 0xff \
    --log-level 1 \
    --shape-id 0 \
    --target rk1820
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rknn3lite.api import RKNN3Lite
from rknn3lite.api.rknn3_types import (
    RKNN3TensorLayout,
    RKNN3TensorType,
    dump_tensor_attr,
    rknn_dtype_to_numpy_dtype,
    rknn3_get_layout_string,
    rknn3_get_type_string,
)


DEFAULT_CORE_MASK = "0xff"
DEFAULT_LOG_LEVEL = 0
DEFAULT_SHAPE_ID = 0
DEFAULT_TARGET = "rk1820"


def split_paths(paths: Optional[str]) -> List[str]:
    if paths is None or paths == "":
        return []
    return [p for p in paths.split("#") if p]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    if n <= 0:
        return 0.0

    a = a[:n]
    b = b[:n]
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0

    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    if denom <= np.finfo(np.float32).eps:
        return 0.0
    return max(-1.0, min(1.0, dot / denom))


def attr_name(attr) -> str:
    raw = attr.name
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return bytes(raw).split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def attr_shape(attr) -> List[int]:
    return [int(attr.shape[i]) for i in range(int(attr.n_dims))]


def is_special_scalar_input(attr) -> Tuple[bool, float]:
    """Mirror the C++ test's special cases for Th/Tc/Ts/Tsr scalar tensors."""
    name = attr_name(attr)
    shape = attr_shape(attr)
    if len(shape) == 1 and shape[0] == 1:
        if "Th" in name:
            return True, 0.0
        if "Tc" in name:
            return True, 1.0
        if "Ts" in name or "Tsr" in name:
            return True, 0.0
    return False, 0.0


def infer_data_format_from_attr(attr) -> str:
    """
    Data format passed to RKNN3Lite.inference.
    For 1-D / non-image tensors, use undefined.
    """
    layout = int(attr.layout)
    ndim = int(attr.n_dims)
    if ndim < 4:
        return "undefined"
    if layout == int(RKNN3TensorLayout.RKNN3_TENSOR_NHWC):
        return "nhwc"
    if layout == int(RKNN3TensorLayout.RKNN3_TENSOR_NCHW):
        return "nchw"
    # NC1HWC2 is an internal/native layout. Feed logical NCHW to the high-level wrapper.
    if layout == int(RKNN3TensorLayout.RKNN3_TENSOR_NC1HWC2):
        return "nchw"
    return "undefined"


def logical_input_shape_from_attr(attr, data_format: str) -> List[int]:
    """
    Build a reasonable logical ndarray shape for RKNN3Lite.inference.
    Do not allocate native packed NC1HWC2 buffers here; the wrapper handles conversion.
    """
    shape = attr_shape(attr)
    layout = int(attr.layout)

    if len(shape) == 5 and layout == int(RKNN3TensorLayout.RKNN3_TENSOR_NC1HWC2):
        n, c1, h, w, c2 = shape
        return [n, c1 * c2, h, w]  # logical NCHW

    if len(shape) == 4:
        # Keep attr shape. data_format tells the wrapper how to interpret it.
        return shape

    return shape if shape else [int(getattr(attr, "n_elems", 1)) or 1]


def numpy_dtype_for_attr(attr):
    try:
        dtype_enum = RKNN3TensorType(int(attr.dtype))
    except ValueError:
        return np.float32

    # bfloat16 has no ordinary NumPy compute dtype; for wrapper input, use fp32.
    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_BFLOAT16:
        return np.float32
    return rknn_dtype_to_numpy_dtype(dtype_enum)


def make_random_array(attr, data_format: str) -> np.ndarray:
    shape = logical_input_shape_from_attr(attr, data_format)
    dtype = numpy_dtype_for_attr(attr)

    special, value = is_special_scalar_input(attr)
    if special:
        return np.full(shape, value, dtype=dtype)

    dtype_enum = None
    try:
        dtype_enum = RKNN3TensorType(int(attr.dtype))
    except ValueError:
        pass

    if dtype_enum in (
        RKNN3TensorType.RKNN3_TENSOR_FLOAT32,
        RKNN3TensorType.RKNN3_TENSOR_FLOAT16,
        RKNN3TensorType.RKNN3_TENSOR_BFLOAT16,
    ):
        return np.random.uniform(-1.0, 1.0, size=shape).astype(dtype)

    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_INT8:
        return np.random.randint(-128, 128, size=shape, dtype=np.int8)

    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_UINT8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)

    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_INT16:
        return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)

    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_UINT16:
        return np.random.randint(0, 65536, size=shape, dtype=np.uint16)

    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_INT32:
        return np.random.randint(-1000, 1001, size=shape, dtype=np.int32)

    if dtype_enum == RKNN3TensorType.RKNN3_TENSOR_UINT32:
        return np.random.randint(0, 1001, size=shape, dtype=np.uint32)

    # Safe fallback. Most models accept fp32 input through the wrapper conversion path.
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def parse_data_formats(data_format_arg: Optional[str], n_inputs: int, input_attrs=None, use_random_input=False) -> List[str]:
    """
    For user npy input, default to NCHW to match the C++ test's assumption.
    For random input, infer from attr because we are creating shape from attr.
    """
    if data_format_arg:
        items = [x.strip().lower() for x in data_format_arg.split("#") if x.strip()]
        if len(items) == 1 and n_inputs > 1:
            items = items * n_inputs
        if len(items) != n_inputs:
            raise ValueError(f"data_format count mismatch: expected {n_inputs}, got {len(items)}")
        return items

    if use_random_input and input_attrs is not None:
        return [infer_data_format_from_attr(attr) for attr in input_attrs]

    return ["nchw"] * n_inputs


def load_inputs_from_npy(input_files: Sequence[str], n_inputs: int) -> List[np.ndarray]:
    if len(input_files) != n_inputs:
        raise ValueError(f"input npy count mismatch: model expects {n_inputs}, got {len(input_files)}")

    inputs = []
    for idx, path in enumerate(input_files):
        if not os.path.exists(path):
            raise FileNotFoundError(f"input[{idx}] not found: {path}")
        arr = np.load(path)
        inputs.append(arr)
        print(f"Loaded input[{idx}]: {path}, shape={arr.shape}, dtype={arr.dtype}")
    return inputs


def make_random_inputs(input_attrs, data_formats: Sequence[str]) -> List[np.ndarray]:
    inputs = []
    for idx, attr in enumerate(input_attrs):
        arr = make_random_array(attr, data_formats[idx])
        inputs.append(arr)
        print(
            f"Generated random input[{idx}]: name={attr_name(attr)}, "
            f"shape={arr.shape}, dtype={arr.dtype}, data_format={data_formats[idx]}"
        )
    return inputs


def save_outputs(outputs: Sequence[np.ndarray], prefix: str = "rt_output") -> None:
    for i, out in enumerate(outputs):
        path = f"{prefix}{i}.npy"
        np.save(path, out)
        print(f"Saved output[{i}] to {path}, shape={out.shape}, dtype={out.dtype}")


def compare_with_goldens(outputs: Sequence[np.ndarray], golden_files: Sequence[str]) -> None:
    if not golden_files:
        print("Skipping golden comparison")
        return

    if len(golden_files) != len(outputs):
        raise ValueError(f"golden count mismatch: model outputs {len(outputs)}, got {len(golden_files)}")

    for i, (out, golden_path) in enumerate(zip(outputs, golden_files)):
        if not os.path.exists(golden_path):
            raise FileNotFoundError(f"golden[{i}] not found: {golden_path}")
        golden = np.load(golden_path)
        sim = cosine_similarity(out, golden)

        out_flat = np.asarray(out).reshape(-1)
        golden_flat = np.asarray(golden).reshape(-1)
        n = min(out_flat.size, golden_flat.size)
        print(f"\nOutput[{i}] compare with {golden_path}")
        print(f"  output shape={out.shape}, dtype={out.dtype}, elems={out_flat.size}")
        print(f"  golden shape={golden.shape}, dtype={golden.dtype}, elems={golden_flat.size}")
        print(f"  compare elems={n}")
        print(f"  cosine similarity={sim:.6f}")

        show = min(10, n)
        if show > 0:
            print("  first values:")
            for j in range(show):
                print(f"    [{j}] output={float(out_flat[j]):.6f}, golden={float(golden_flat[j]):.6f}")


def print_attrs(title: str, attrs) -> None:
    print(f"\n{title}:")
    for i, attr in enumerate(attrs):
        try:
            dump_tensor_attr(attr, prefix=f"{title.lower()}[{i}]")
        except Exception:
            print(
                f"{title}[{i}]: name={attr_name(attr)}, shape={attr_shape(attr)}, "
                f"layout={rknn3_get_layout_string(int(attr.layout))}, "
                f"dtype={rknn3_get_type_string(int(attr.dtype))}"
            )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    # Support original positional C++ style unless named args are used.
    if argv and not argv[0].startswith("-"):
        parser = argparse.ArgumentParser(description="RKNN3 profile ops test with RKNN3Lite APIs")
        parser.add_argument("rknn_path")
        parser.add_argument("weight_path")
        parser.add_argument("inputs", nargs="?", default="")
        parser.add_argument("goldens", nargs="?", default="")
        parser.add_argument("core_mask", nargs="?", default=DEFAULT_CORE_MASK)
        parser.add_argument("log_level", nargs="?", type=int, default=DEFAULT_LOG_LEVEL)
        parser.add_argument("shape_id", nargs="?", type=int, default=DEFAULT_SHAPE_ID)
        parser.add_argument("device_id", nargs="?", default=None)
        parser.add_argument("--target", default=DEFAULT_TARGET)
        parser.add_argument("--data-format", default=None, help="input data format(s): nchw/nhwc/undefined, separated by #")
        parser.add_argument("--verbose", action="store_true", default=True)
        return parser.parse_args(argv)

    parser = argparse.ArgumentParser(description="RKNN3 profile ops test with RKNN3Lite APIs")
    parser.add_argument("--rknn-path", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--inputs", default="", help="input npy paths separated by '#'. Empty means random input")
    parser.add_argument("--goldens", default="", help="golden output npy paths separated by '#'. Empty skips comparison")
    parser.add_argument("--core-mask", default=DEFAULT_CORE_MASK, help="hex core mask, default 0xff")
    parser.add_argument("--log-level", type=int, default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--shape-id", type=int, default=DEFAULT_SHAPE_ID)
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--data-format", default=None, help="input data format(s): nchw/nhwc/undefined, separated by #")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    rknn_path = getattr(args, "rknn_path")
    weight_path = getattr(args, "weight_path")
    input_files = split_paths(getattr(args, "inputs", ""))
    golden_files = split_paths(getattr(args, "goldens", ""))
    use_random_input = len(input_files) == 0

    print("RKNN3 Lite Profile Ops Test")
    print(f"rknn_path={rknn_path}")
    print(f"weight_path={weight_path}")
    print(f"target={args.target}, core_mask={args.core_mask}, log_level={args.log_level}, shape_id={args.shape_id}")
    if use_random_input:
        print("Using random input data")
    if not golden_files:
        print("Golden output not provided")

    rknn = RKNN3Lite(verbose=args.verbose)

    try:
        print("\n--> Loading model")
        ret = rknn.load_rknn(rknn_path, weight_path)
        if ret != 0:
            print("load_rknn failed")
            return ret
        print("done")

        print("\n--> Init runtime")
        ret = rknn.init_runtime(
            target=args.target,
            core_mask=int(args.core_mask, 16),
            device_id=args.device_id,
        )
        if ret != 0:
            print("init_runtime failed")
            return ret
        print("done")

        if args.shape_id != 0:
            print(f"\n--> Set dynamic shape: {args.shape_id}")
            ret = rknn.set_shape(args.shape_id)
            if ret != 0:
                print(f"set_shape({args.shape_id}) failed")
                return ret
            print("done")

        input_attrs = rknn.get_inputs_tensor_attr()
        output_attrs = rknn.get_outputs_tensor_attr()
        if input_attrs is None or output_attrs is None:
            print("failed to query input/output attrs")
            return -1

        print_attrs("Input tensors", input_attrs)
        print_attrs("Output tensors", output_attrs)

        n_inputs = len(input_attrs)
        data_formats = parse_data_formats(
            args.data_format,
            n_inputs,
            input_attrs=input_attrs,
            use_random_input=use_random_input,
        )

        if use_random_input:
            inputs = make_random_inputs(input_attrs, data_formats)
        else:
            inputs = load_inputs_from_npy(input_files, n_inputs)
            print(f"Using data_format={data_formats}")

        print("\n--> Inference")
        outputs = rknn.inference(inputs=inputs, data_format=data_formats)
        if outputs is None:
            print("inference failed")
            return -1
        print("done")

        print("\n--> Profile memory")
        ret = rknn.profile_mem()
        if ret != 0:
            print(f"profile_mem failed: ret={ret}")
            return ret
        print("done")

        print("\n--> Profile ops")
        ret = rknn.profile_ops(args.log_level)
        if ret != 0:
            print(f"profile_ops failed: ret={ret}")
            return ret
        print("done")

        print("\n--> Save outputs")
        save_outputs(outputs)

        print("\n--> Compare golden outputs")
        compare_with_goldens(outputs, golden_files)

        return 0

    finally:
        try:
            rknn.release()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))