#!/usr/bin/env python3
"""
Convert ONNX model with RknnCustomOpExam to RKNN format.

Uses rknn.reg_custom_op() to register the custom op so the RKNN toolchain
can parse it (shape inference + reference compute).  The actual on-device
execution is handled by the C plugin .so registered at runtime via
rknn3_register_custom_ops_plugins().

Usage:
    python3 convert2rknn.py --onnx_model_path ../install/rknn_custom_op.onnx
"""

import sys
import argparse
import os
import numpy as np
from rknn.api import RKNN
from rknn.api.custom_op import get_node_attr


# ------------------------------------------------------------------ #
#  Custom op Python class for RKNN toolchain                         #
#  - shape_infer: tells RKNN the output shape/dtype                  #
#  - compute:     reference implementation for toolkit verification  #
#  (The on-device implementation is in the C plugin .so)             #
# ------------------------------------------------------------------ #

class RknnCustomOpExam:
    op_type = 'RknnCustomOpExam'

    def shape_infer(self, node, in_shapes, in_dtypes):
        # Output has same shape and dtype as input
        out_shapes = in_shapes.copy()
        out_dtypes = in_dtypes.copy()
        return out_shapes, out_dtypes

    def compute(self, node, inputs):
        x = inputs[0]
        scale = get_node_attr(node, 'scale')
        shift = get_node_attr(node, 'shift')
        min_val = get_node_attr(node, 'min_val')
        max_val = get_node_attr(node, 'max_val')

        y = x * scale + shift
        y = np.clip(y, min_val, max_val)
        return [y.astype(np.float32)]


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to RKNN.')
    parser.add_argument('--onnx_model_path', type=str, default='../install/rknn_custom_op.onnx',
                        help='Path to ONNX model.')
    parser.add_argument('--platform', type=str, default='rk1820',
                        choices=['rk1820', 'rk1828', 'rk3572'],
                        help='Target platform.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output RKNN file path (default: ../install/rknn_custom_op.rknn)')
    args = parser.parse_args()

    if args.output is None:
        install_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "install")
        os.makedirs(install_dir, exist_ok=True)
        args.output = os.path.join(install_dir, "rknn_custom_op.rknn")

    return args


if __name__ == '__main__':
    args = parse_args()
    abs_path = os.path.abspath(args.output)
    directory = os.path.dirname(abs_path)
    os.makedirs(directory, exist_ok=True)

    rknn = RKNN(verbose=False)

    # Register custom op so RKNN toolchain can parse the ONNX model
    # Applicable to non onnx standard operators
    print('--> Register custom op')
    ret = rknn.reg_custom_op(RknnCustomOpExam())
    if ret != 0:
        print('Register custom op failed!')
        sys.exit(ret)
    print('done')
    output_attrs = { 'y':{'dtype': 'float16', 'layout': 'NC1HWC2'}, }

    print('--> Config model')
    config_params = {
        'mean_values': [[0, 0, 0]],
        'std_values': [[1, 1, 1]],
        'target_platform': args.platform,
        'output_attrs': output_attrs
    }

    rknn.config(**config_params)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_model_path)
    if ret != 0:
        print('Load model failed!')
        sys.exit(ret)
    print('done')

    print('--> Building model (fp, no quantization)')
    ret = rknn.build(do_quantization=False, dataset=None)
    if ret != 0:
        print('Build model failed!')
        sys.exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print('Export rknn model failed!')
        sys.exit(ret)
    print('done')
    print(f'[convert2rknn] RKNN model saved to {args.output}')
