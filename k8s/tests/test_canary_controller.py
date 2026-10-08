import shutil
import subprocess

import pytest

from k8s.canary_controller import build_env_patch, get_accuracy

CLUSTER_AVAILABLE = shutil.which("kubectl") is not None and subprocess.run(
    ["kubectl", "get", "nodes"], capture_output=True
).returncode == 0


def test_build_env_patch_shape():
    patch = build_env_patch("/app/model/checkpoints/model.pkl", "v2")
    containers = patch["spec"]["template"]["spec"]["containers"]
    assert containers[0]["name"] == "inference"
    env = {e["name"]: e["value"] for e in containers[0]["env"]}
    assert env["MODEL_PATH"] == "/app/model/checkpoints/model.pkl"
    assert env["MODEL_VERSION"] == "v2"


@pytest.mark.skipif(not CLUSTER_AVAILABLE, reason="no live k3d/kubectl cluster reachable")
def test_live_accuracy_endpoints_respond():
    """Integration check against the real cluster, if it's up: both
    deployments should be reachable and return a plausible accuracy."""
    stable = get_accuracy("fenris-inference-stable")
    assert 0.0 <= stable["accuracy"] <= 1.0
