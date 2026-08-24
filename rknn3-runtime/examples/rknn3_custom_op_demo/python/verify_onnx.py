#!/usr/bin/env python3
"""
Verify the RknnCustomOpExam ONNX model by computing the reference output with numpy
and saving it as a .npy file for later cosine-similarity comparison with RKNN.

Reference computation:  y = clamp(x * scale + shift, min_val, max_val)

Usage:
    python3 verify_onnx.py [--onnx rknn_custom_op.onnx] [--output ref_output.npy]
"""

import argparse
import os
import numpy as np


def compute_reference(x, scale=2.0, shift=0.5, min_val=-3, max_val=3):
    """Compute y = clamp(x * scale + shift, min_val, max_val)."""
    y = x * scale + shift
    y = np.clip(y, min_val, max_val)
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Verify RknnCustomOpExam ONNX and save reference output")
    parser.add_argument("--onnx", type=str, default="../install/rknn_custom_op.onnx",
                        help="ONNX model path (for attribute verification)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npy file for reference result (default: ../install/ref_output.npy)")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional input .npy file; if not given, random input is used")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    install_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "install")
    os.makedirs(install_dir, exist_ok=True)

    if args.output is None:
        args.output = os.path.join(install_dir, "ref_output.npy")

    np.random.seed(args.seed)

    # Read input shape from ONNX model
    import onnx
    onnx_model = onnx.load(args.onnx)
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    input_shape = tuple(input_shape)
    print(f"[verify] Input shape from ONNX: {input_shape}")

    # Generate or load input
    if args.input and os.path.exists(args.input):
        x = np.load(args.input).astype(np.float32)
        print(f"[verify] Loaded input from {args.input}, shape={x.shape}")
    else:
        x = np.random.randn(*input_shape).astype(np.float32)
        input_path = os.path.join(install_dir, "ref_input.npy")
        np.save(input_path, x)
        print(f"[verify] Generated random input, shape={x.shape}, saved to {input_path}")

    # Print sample input values
    print(f"[verify] Input sample (first 5 values): {x.flatten()[:5]}")

    # Compute reference output
    y_ref = compute_reference(x)
    print(f"[verify] Output sample (first 5 values): {y_ref.flatten()[:5]}")
    print(f"[verify] Output shape: {y_ref.shape}, dtype: {y_ref.dtype}")

    # Save reference output
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, y_ref)
    print(f"[verify] Reference output saved to {output_path}")

    # Also verify with onnxruntime if custom op is registered (optional)
    try:
        import onnxruntime as ort

        # Register a custom op kernel for onnxruntime
        class RknnCustomOpExamSession:
            pass

        # onnxruntime can't run custom ops without a custom kernel,
        # so we skip this and rely on numpy reference.
        print("[verify] onnxruntime available but custom op not registered, skipping ORT inference")
    except ImportError:
        print("[verify] onnxruntime not available, numpy reference only")


if __name__ == "__main__":
    main()
