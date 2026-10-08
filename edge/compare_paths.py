"""
Runs both edge inference paths on the same input and compares them.
On the M1, only the ARM path can actually run -- the CUDA path reports
clearly that it needs a GPU (see edge/cuda_path/infer.py) instead of
pretending to have a number it doesn't.

Usage: python -m edge.compare_paths
"""

from edge.arm_path import infer as arm_infer
from edge.cuda_path import infer as cuda_infer


def main():
    print("Running ARM path (M1 CPU, real)...")
    arm_latency, arm_pred, label = arm_infer.run()

    print("\nRunning CUDA path (requires GPU)...")
    cuda_latency, cuda_pred, _ = cuda_infer.run()

    print("\n--- Comparison ---")
    print(f"ARM path latency:  {arm_latency*1000:.3f} ms")
    if cuda_latency is not None:
        print(f"CUDA path latency: {cuda_latency*1000:.3f} ms")
        speedup = arm_latency / cuda_latency
        print(f"CUDA path is {speedup:.2f}x the speed of the ARM path")
        agree = "yes" if arm_pred == cuda_pred else "no"
        print(f"Both paths agree on the prediction: {agree}")
    else:
        print("CUDA path latency: not available on this machine (no GPU) -- "
              "run this same script on Colab to get the real comparison.")


if __name__ == "__main__":
    main()
