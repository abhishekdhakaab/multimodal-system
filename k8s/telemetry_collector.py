"""
Polls every fenris-inference service in the live cluster for its real
/hard_cases telemetry (low-confidence predictions, found by the pod
actually running inference -- not simulated) and writes the aggregated
result to a local telemetry log. This stands in for a fleet's central
telemetry ingestion -- nodes report hard cases, a central process collects
them for the data pipeline to mine from.

Usage: python -m k8s.telemetry_collector
"""

import json
import os
import subprocess
import time
import urllib.request

SERVICES = ["fenris-inference-stable", "fenris-inference-canary"]
PORT_FORWARD_LOCAL_PORT = 18081
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "telemetry", "hard_cases.json")


def get_hard_cases(service, timeout_s=20):
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
                with urllib.request.urlopen(f"http://localhost:{PORT_FORWARD_LOCAL_PORT}/hard_cases", timeout=3) as resp:
                    return json.loads(resp.read())
            except Exception as e:
                last_error = e
        raise RuntimeError(f"could not reach {service}'s /hard_cases within {timeout_s}s: {last_error}")
    finally:
        proc.terminate()
        proc.wait()


def collect():
    report = {"collected_from": [], "nodes": []}
    for service in SERVICES:
        print(f"[telemetry] polling {service}...")
        result = get_hard_cases(service)
        n_hard = len(result["hard_case_indices"])
        print(
            f"[telemetry] {service}: model_version={result['model_version']} "
            f"flagged {n_hard}/{result['num_examples']} examples as hard cases"
        )
        report["collected_from"].append(service)
        report["nodes"].append(result)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[telemetry] wrote aggregated report to {OUT_PATH}")
    return report


if __name__ == "__main__":
    collect()
