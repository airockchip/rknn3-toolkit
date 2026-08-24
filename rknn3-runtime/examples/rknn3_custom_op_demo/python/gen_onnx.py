#!/usr/bin/env python3
"""
Generate an ONNX model with a custom operator "RknnCustomOpExam".

The custom op computes:  y = clamp(x * scale + shift, min_val, max_val)

Its attributes cover all 6 dtype tags used by the RKNN3 custom-op framework:
  i  (int)    : min_val=-3, max_val=3
  f  (float)  : scale=2.0, shift=0.5
  s  (string) : mode="linear"
  is (ints)   : strides=[1,2,3]
  fs (floats) : weights=[0.1,0.2,0.3]
  ss (strings): tags=["a","b"]

Usage:
    python3 gen_onnx.py [--output rknn_custom_op.onnx]
"""

import argparse
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto


def build_model(input_shape=(1, 3, 256, 256), output_path="rknn_custom_op.onnx"):
    """Build an ONNX model with a single RknnCustomOpExam node."""

    # Input / output info
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(input_shape))

    # Custom op node with all 6 attribute types
    node = helper.make_node(
        "RknnCustomOpExam",          # op_type (must match the custom op registered in RKNN)
        inputs=["x"],
        outputs=["y"],
        name="rknn_custom_op_0",
        domain="",             # use default domain so RKNN toolchain can parse it

        # --- dtype "i" (int) attributes ---
        min_val=-3,
        max_val=3,

        # --- dtype "f" (float) attributes ---
        scale=2.0,
        shift=0.5,

        # --- dtype "s" (string) attribute ---
        mode="linear",

        # --- dtype "is" (ints) attribute ---
        strides=[1, 2, 3],

        # --- dtype "fs" (floats) attribute ---
        weights=[0.1, 0.2, 0.3],

        # --- dtype "ss" (strings) attribute ---
        tags=["a", "b"],
    )

    graph = helper.make_graph(
        [node],
        "rknn_custom_op_graph",
        [x],
        [y],
    )

    model = helper.make_model(
        graph,
        producer_name="gen_onnx.py",
        opset_imports=[helper.make_opsetid("", 15)],
    )
    model.ir_version = 8

    # Skip onnx.checker.check_model - RknnCustomOpExam is not a standard ONNX op,
    # so the checker would reject it. RKNN toolchain handles custom ops via custom_op_lib.
    onnx.save(model, output_path)
    print(f"[gen_onnx] Model saved to {output_path}")
    print(f"[gen_onnx] Input shape:  {input_shape}")
    print(f"[gen_onnx] Input dtype:  float32")
    print(f"[gen_onnx] Attributes:")
    print(f"  min_val (i)  = -3")
    print(f"  max_val (i)  = 3")
    print(f"  scale   (f)  = 2.0")
    print(f"  shift   (f)  = 0.5")
    print(f"  mode    (s)  = 'linear'")
    print(f"  strides (is) = [1, 2, 3]")
    print(f"  weights (fs) = [0.1, 0.2, 0.3]")
    print(f"  tags    (ss) = ['a', 'b']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ONNX model with RknnCustomOpExam")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX model path (default: ../install/rknn_custom_op.onnx)")
    parser.add_argument("--shape", type=int, nargs=4, default=[1, 3, 256, 256],
                        help="Input shape as N C H W (default: 1 3 256 256)")
    args = parser.parse_args()

    if args.output is None:
        install_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "install")
        os.makedirs(install_dir, exist_ok=True)
        args.output = os.path.join(install_dir, "rknn_custom_op.onnx")

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    build_model(tuple(args.shape), output_path)
