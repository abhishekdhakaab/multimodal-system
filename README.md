# Fenris — Multimodal Perception Compiler for Edge Robotics Fleets

A compiler + runtime for a multimodal (camera + lidar) perception model,
built end to end: real data, a trained fusion model, a hand-written CUDA
kernel targeting a profiled bottleneck, a dual-path edge runtime (ARM vs.
GPU-accelerated), a real Kubernetes fleet control plane with canary
rollout/rollback, and a telemetry → hard-case-mining closed loop.

See `PLAN.md` for the full phase-by-phase plan, status, and checklist —
it's the project's running source of truth. This README is a quickstart
and orientation; `docs/final_report.md` is the full writeup with every
number and what's real vs. simulated.

## Quickstart

```bash
# 1. Set up the environment
./scripts/setup_env.sh
source venv/bin/activate

# 2. Build the real dataset (ModelNet10, ~473MB, no signup needed)
# see docs/dataset_rework_notes.md for how to fetch ModelNet10.zip into
# data_pipeline/raw/ -- it's gitignored (too large), not included in the repo
python -m data_pipeline.build_shards

# 3. Train the multimodal fusion model (~15 min on a laptop CPU)
python -m model.train

# 4. Run inference on the real ARM edge path (works on any machine, no GPU needed)
python -m edge.arm_path.infer

# 5. Try the Kubernetes fleet demo (needs Docker + k3d)
./scripts/setup_k3s.sh
docker build -f k8s/Dockerfile -t fenris-inference:v1 .
k3d image import fenris-inference:v1 -c fenris
kubectl apply -f k8s/manifests/stable.yaml -f k8s/manifests/canary.yaml
python -m k8s.canary_controller rollout --model-path /app/model/checkpoints/model.pkl --version v2
```

## What's real vs. simulated (read this before citing any number)

| Component | Status |
|---|---|
| Dataset | **Real** — ModelNet10, public, no signup |
| Model + training | **Real** — 86.01% val accuracy, measured |
| Profiling / bottleneck finding | **Real** — measured on this machine |
| CUDA kernel | **Written, untested on real hardware** — needs a CUDA GPU (Colab), see `scripts/colab_sync.md` |
| Benchmarking (Phase 4) | **Blocked** — needs the kernel to have actually run |
| Edge ARM path | **Real** — runs and measured on the dev machine |
| Edge CUDA path | **Written, untested** — same blocker as the kernel |
| Kubernetes cluster | **Real** — local k3d, genuinely ARM64 nodes |
| GPU fleet node | **Simulated, labeled as such** (`gpu-simulated`) — no real GPU node attached |
| Canary rollout/rollback | **Real** — actually run against the live cluster, logs in `docs/k8s_canary_demo.md` |
| Telemetry / hard-case mining | **Real** — actually run, numbers in `docs/hard_case_mining_notes.md` |

## Project structure

```
fenris/
  PLAN.md               the project plan, checklist, and current status
  data_pipeline/          ModelNet10 loading, shard building, hard-case mining
  model/                  the multimodal fusion model, training, profiling
  kernels/                CUDA kernel + XLA custom-call glue (Phase 3)
  edge/                   dual-path inference runtime (ARM vs CUDA)
  k8s/                    Kubernetes manifests, serving image, canary controller, telemetry
  benchmarks/             (Phase 4, pending Colab results)
  docs/                   detailed writeups for every phase, with real numbers
  scripts/                setup and helper scripts
```

## Key documents

- `docs/dataset_rework_notes.md` — why/how the dataset is real, not synthetic
- `docs/fusion_bug_notes.md` — a real bug found via ablation, not just tuning
- `docs/phase2_profiling_notes.md` — the bottleneck finding behind the CUDA kernel
- `docs/edge_dual_path_notes.md` — ARM vs CUDA inference path
- `docs/k8s_canary_demo.md` — the real canary rollout/rollback run
- `docs/hard_case_mining_notes.md` — the real telemetry closed-loop run
- `docs/final_report.md` — the complete writeup
