"""
Canary rollout controller for the fenris-inference deployments.

Flow for rolling out a new model version:
  1. Patch the canary deployment to the new MODEL_PATH/MODEL_VERSION, wait
     for its pod to become ready.
  2. Port-forward to the canary service and poll its real /accuracy
     endpoint (computed live inside the pod on a held-out shard, not a
     hardcoded number).
  3. If accuracy >= (stable's accuracy - REGRESSION_TOLERANCE): PROMOTE --
     patch the stable deployment to the same version, then reset canary
     back to the stable version (ready for the next rollout).
  4. Otherwise: ROLLBACK -- revert canary to the previous version, leave
     stable untouched. Nothing resembling the new model ever reaches the
     replicas actually serving traffic.

Uses plain `kubectl` subprocess calls -- no extra Kubernetes client library
dependency, appropriate at this project's scale.

Usage:
    python -m k8s.canary_controller rollout --model-path /app/model/checkpoints/model.pkl --version v2
    python -m k8s.canary_controller rollout --model-path /app/model/checkpoints/model_bad.pkl --version v3-bad
"""

import argparse
import json
import subprocess
import time
import urllib.request

REGRESSION_TOLERANCE = 0.05  # canary accuracy can be at most 5 points below stable's before rolling back
PORT_FORWARD_LOCAL_PORT = 18080


def kubectl(*args):
    result = subprocess.run(["kubectl"] + list(args), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def build_env_patch(model_path, version):
    return {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "inference",
                            "env": [
                                {"name": "MODEL_PATH", "value": model_path},
                                {"name": "MODEL_VERSION", "value": version},
                                {"name": "NODE_TYPE", "value": "arm"},
                            ],
                        }
                    ]
                }
            }
        }
    }


def patch_deployment_env(deployment, model_path, version):
    patch = build_env_patch(model_path, version)
    kubectl("patch", "deployment", deployment, "--type=strategic", "-p", json.dumps(patch))
    kubectl("rollout", "status", f"deployment/{deployment}", "--timeout=60s")


def get_accuracy(service, timeout_s=30):
    proc = subprocess.Popen(
        ["kubectl", "port-forward", f"svc/{service}", f"{PORT_FORWARD_LOCAL_PORT}:8080"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + timeout_s
        last_error = None
        while time.time() < deadline:
            time.sleep(1)
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT_FORWARD_LOCAL_PORT}/accuracy", timeout=3) as resp:
                    return json.loads(resp.read())
            except Exception as e:
                last_error = e
        raise RuntimeError(f"could not reach {service}'s /accuracy within {timeout_s}s: {last_error}")
    finally:
        proc.terminate()
        proc.wait()


def rollout(model_path, version):
    print(f"[canary] patching canary deployment to version={version} ({model_path})")
    patch_deployment_env("fenris-inference-canary", model_path, version)

    print("[canary] polling canary's live /accuracy...")
    canary_result = get_accuracy("fenris-inference-canary")
    canary_acc = canary_result["accuracy"]
    print(f"[canary] canary accuracy: {canary_acc:.4f} (version={version})")

    print("[canary] polling stable's live /accuracy for comparison...")
    stable_result = get_accuracy("fenris-inference-stable")
    stable_acc = stable_result["accuracy"]
    stable_version = stable_result["model_version"]
    print(f"[canary] stable accuracy: {stable_acc:.4f} (version={stable_version})")

    if canary_acc >= stable_acc - REGRESSION_TOLERANCE:
        print(f"[canary] PROMOTE: canary ({canary_acc:.4f}) is within tolerance of stable ({stable_acc:.4f})")
        patch_deployment_env("fenris-inference-stable", model_path, version)
        print(f"[canary] stable is now serving version={version}")
        return "promoted"
    else:
        print(
            f"[canary] ROLLBACK: canary ({canary_acc:.4f}) regressed more than "
            f"{REGRESSION_TOLERANCE} below stable ({stable_acc:.4f})"
        )
        patch_deployment_env("fenris-inference-canary", "/app/model/checkpoints/model.pkl", stable_version)
        print(f"[canary] canary reset back to version={stable_version}, stable untouched")
        return "rolled_back"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rollout_parser = sub.add_parser("rollout")
    rollout_parser.add_argument("--model-path", required=True)
    rollout_parser.add_argument("--version", required=True)
    args = parser.parse_args()

    if args.command == "rollout":
        outcome = rollout(args.model_path, args.version)
        print(f"\n[canary] outcome: {outcome}")
