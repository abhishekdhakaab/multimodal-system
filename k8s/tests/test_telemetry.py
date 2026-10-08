import shutil
import subprocess

import pytest

from k8s.telemetry_collector import get_hard_cases

CLUSTER_AVAILABLE = shutil.which("kubectl") is not None and subprocess.run(
    ["kubectl", "get", "nodes"], capture_output=True
).returncode == 0


@pytest.mark.skipif(not CLUSTER_AVAILABLE, reason="no live k3d/kubectl cluster reachable")
def test_live_hard_cases_endpoint_returns_real_data():
    result = get_hard_cases("fenris-inference-stable")
    assert result["num_examples"] > 0
    assert isinstance(result["hard_case_indices"], list)
    assert all(0 <= i < result["num_examples"] for i in result["hard_case_indices"])
    assert len(result["hard_case_confidences"]) == len(result["hard_case_indices"])
    assert all(0.0 <= c <= 1.0 for c in result["hard_case_confidences"])
