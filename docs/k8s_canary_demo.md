# Kubernetes Fleet Control Plane — Canary/Rollback Demo

## What's real here

- A real local k3d (K3s-in-Docker) cluster, 3 nodes (1 server + 2 agents),
  running on genuine ARM64 silicon (the M1) — `kubernetes.io/arch=arm64`
  confirmed on every node via `kubectl get nodes -o wide`.
- `node-type=arm` on agent-0 is real, not simulated. `node-type=gpu-simulated,
  gpu=false` on agent-1 is explicitly labeled as simulated — there's no real
  GPU node available yet (blocked on Colab results, see `PLAN.md`). The label
  itself says so, so nobody downstream mistakes it for real GPU hardware.
- A real Docker image (`fenris-inference:v1`, ~1GB) running the actual
  trained model, imported into the k3d cluster, serving real HTTP traffic
  from real Kubernetes pods.
- `/accuracy` on every pod computes real accuracy live, in-pod, on a held-
  out validation shard — not a hardcoded number returned by the API.

## The demo sequence (actually run, not just described)

```
$ python -m k8s.canary_controller rollout --model-path .../model.pkl --version v2
[canary] canary accuracy: 0.8300 (version=v2)
[canary] stable accuracy: 0.8300 (version=v1)
[canary] PROMOTE: canary (0.8300) is within tolerance of stable (0.8300)
[canary] stable is now serving version=v2

$ python -m k8s.canary_controller rollout --model-path .../model_bad.pkl --version v3-bad
[canary] canary accuracy: 0.1050 (version=v3-bad)
[canary] stable accuracy: 0.8300 (version=v2)
[canary] ROLLBACK: canary (0.1050) regressed more than 0.05 below stable (0.8300)
[canary] canary reset back to version=v2, stable untouched
```

Verified after the fact with `kubectl get deployment ... -o jsonpath=...`:
`fenris-inference-stable` stayed on `v2` throughout the bad rollout attempt
— the broken model never reached the deployment actually serving traffic.

## How the "bad" version was made realistic

`model_bad.pkl` is the same architecture, randomly initialized and never
trained (see `scripts/build_bad_checkpoint.py`) — standing in for a
training job that silently failed to converge, or a corrupted checkpoint
upload. Its live-measured accuracy (10.5%) is right at the 10-class random-
chance baseline, which is itself a nice sanity check that the measurement
pipeline is honest: an untrained model measuring in at ~10% is exactly what
should happen, not a suspiciously round or convenient number.

## What the controller actually does (no mocking)

`k8s/canary_controller.py` uses plain `kubectl` subprocess calls (no extra
Kubernetes client library) to: patch the canary Deployment's env vars,
wait for `kubectl rollout status` to confirm the new pod is actually ready,
port-forward to the live service, hit `/accuracy` over real HTTP, compare
against stable's live-measured accuracy, and only then patch (or don't
patch) the stable Deployment. Every number in the log above came from an
actual running pod at the moment the command ran.

## Honest limitations

- Only one node type (`arm`) is used for both canary and stable in this
  demo — that's the correct canary pattern (test on the same hardware class
  as production). The `gpu-simulated` node exists in the cluster and is
  schedulable, but nothing is deployed there yet since there's no real GPU
  inference path to put on it until Phase 3/4's Colab results land.
- The validation shard baked into the image is one 200-example shard
  (`shard_000.npz`), not the full held-out set — accuracy numbers here
  (83%) are close to but not identical to the full-val-set number (86.01%
  best epoch) reported in `docs/phase2_profiling_notes.md`. Both are real
  measurements, just on different-sized samples.
