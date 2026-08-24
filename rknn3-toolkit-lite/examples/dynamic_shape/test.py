# -*- coding: utf-8 -*-
"""RKNN3Lite dynamic-shape ResNet18 demo.

The application always passes an ordinary NHWC uint8 image:

    outputs = rknn_lite.inference(inputs=[img])

No shape_id selection, tensor reallocation or NC1HWC2 conversion is required in
user code.  RKNN3Lite matches the image shape against model metadata and switches
the runtime shape internally.
"""

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple
from pprint import pprint
import cv2
import numpy as np

from rknn3lite.api import RKNN3Lite


DEFAULT_MODEL = "resnet18.rknn"
DEFAULT_WEIGHT = "resnet18.weight"
DEFAULT_IMAGE = "dog_224x224.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RKNN3Lite automatic dynamic-shape inference demo"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .rknn")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help="Path to .weight")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Input image")
    parser.add_argument("--target", default="rk1820", help="Target platform")
    parser.add_argument("--device-id", default=None, help="Optional device id")
    parser.add_argument(
        "--core-mask",
        type=lambda value: int(value, 0),
        default=0x01,
        help="NPU core mask, for example 0x01",
    )
    parser.add_argument(
        "--run-order",
        default=None,
        help=(
            "Optional logical sizes, for example 224,160,256 or "
            "224x320,256x256. By default all model shapes are run."
        ),
    )
    parser.add_argument(
        "--repeat-first",
        action="store_true",
        help="Run the first shape again to verify switch-back memory reuse.",
    )
    parser.add_argument(
        "--labels",
        default="labels.txt",
        help="Optional ImageNet label file.",
    )
    return parser.parse_args()


def load_labels(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return [str(i) for i in range(1000)]
    with open(path, "r", encoding="utf-8") as fp:
        labels = []
        for line in fp:
            value = line.strip()
            if not value:
                continue
            labels.append(value.split(":", 1)[-1].strip())
        return labels


def parse_hw_list(value: str) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" in token:
            h_text, w_text = token.split("x", 1)
            result.append((int(h_text), int(w_text)))
        else:
            size = int(token)
            result.append((size, size))
    return result


def logical_hw(entry: Dict) -> Tuple[int, int]:
    shape = [int(v) for v in entry["inputs"][0]["shape"]]
    if len(shape) != 4:
        raise ValueError("This demo expects a 4-D image input, got {}".format(shape))
    return shape[2], shape[3]


def select_entries(shape_infos: Sequence[Dict], run_order: str) -> List[Dict]:
    if not run_order:
        return list(shape_infos)

    selected = []
    for requested_hw in parse_hw_list(run_order):
        matches = [entry for entry in shape_infos if logical_hw(entry) == requested_hw]
        if len(matches) != 1:
            raise RuntimeError(
                "Expected one shape for {}x{}, found {}".format(
                    requested_hw[0], requested_hw[1], len(matches)
                )
            )
        selected.append(matches[0])
    return selected


def print_top5(output: np.ndarray, labels: Sequence[str]) -> None:
    logits = np.asarray(output, dtype=np.float32).reshape(-1)
    indices = np.argsort(logits)[::-1][:5]

    print("----- TOP 5 -----")
    for index in indices:
        label = labels[index] if index < len(labels) else str(index)
        print(
            '[{:>3d}] score:{:.6f} class:"{}"'.format(
                int(index), float(logits[index]), label
            )
        )


def first_device_id(rknn_lite: RKNN3Lite):
    try:
        device_ids = rknn_lite.get_devices_id()
    except Exception:
        return None
    if isinstance(device_ids, (list, tuple)) and device_ids:
        return device_ids[0]
    return None


def main() -> int:
    args = parse_args()
    for path in (args.model, args.weight, args.image):
        if not os.path.isfile(path):
            print("File not found: {}".format(path), file=sys.stderr)
            return 1

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print("Failed to read image: {}".format(args.image), file=sys.stderr)
        return 1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    labels = load_labels(args.labels)

    rknn_lite = RKNN3Lite()
    try:
        print("--> Load RKNN model")
        ret = rknn_lite.load_rknn(args.model, args.weight)
        if ret != 0:
            print("load_rknn failed: {}".format(ret), file=sys.stderr)
            return ret

        device_id = args.device_id or first_device_id(rknn_lite)
        init_kwargs = {
            "target": args.target,
            "core_mask": args.core_mask,
        }
        if device_id:
            init_kwargs["device_id"] = device_id

        print("--> Init runtime")
        ret = rknn_lite.init_runtime(**init_kwargs)
        if ret != 0:
            print("init_runtime failed: {}".format(ret), file=sys.stderr)
            return ret

        shape_infos = rknn_lite.get_dynamic_shape_infos()
        if not shape_infos:
            print("The supplied model is not a multi-shape model.", file=sys.stderr)
            return 1
        if any(len(entry.get("inputs", [])) != 1 for entry in shape_infos):
            print("This demo expects a single-input image model.", file=sys.stderr)
            return 1

        for entry in shape_infos:
            print(f"\n========== Shape ID: {entry['shape_id']} ==========")
            pprint(
                entry,
                sort_dicts=False,
                width=100,
                indent=2,
            )

        entries = select_entries(shape_infos, args.run_order)
        if args.repeat_first and len(entries) > 1:
            entries.append(entries[0])

        for run_index, entry in enumerate(entries):
            h, w = logical_hw(entry)
            img = cv2.resize(image_rgb, (w, h))
            img = np.expand_dims(img, axis=0)

            print(
                "\n--> run={}, logical NCHW={}, user NHWC={}".format(
                    run_index, entry["inputs"][0]["shape"], list(img.shape)
                )
            )
            outputs = rknn_lite.inference(inputs=[img])
            if outputs is None:
                print("inference failed", file=sys.stderr)
                return 1
            print_top5(outputs[0], labels)

        return 0
    finally:
        rknn_lite.release()


if __name__ == "__main__":
    sys.exit(main())