import os

import pytest

from edge.arm_path import infer as arm_infer
from edge.cuda_path import infer as cuda_infer

CHECKPOINT_EXISTS = os.path.exists(arm_infer.CHECKPOINT_PATH)


@pytest.mark.skipif(not CHECKPOINT_EXISTS, reason="run model/train.py first to produce a checkpoint")
def test_arm_path_runs_and_predicts_in_valid_range():
    latency, pred, label = arm_infer.run(n_timing_runs=3)
    assert latency > 0
    assert 0 <= pred < 10
    assert 0 <= label < 10


def test_cuda_path_reports_missing_kernel_cleanly_without_gpu():
    if os.path.exists(cuda_infer.SO_PATH):
        pytest.skip("fused_attention.so exists on this machine -- this test is for the no-GPU case")
    latency, pred, label = cuda_infer.run()
    assert latency is None and pred is None and label is None
