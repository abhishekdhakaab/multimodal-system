# Fenris — Multimodal Perception Compiler for Edge Robotics Fleets

> **This file is the single source of truth for this project.** At the start of every work session, read the `STATUS` block first before doing anything else. Update `STATUS` and check off tasks as they complete, in the same edit as the work itself — never batch status updates for later. If a task turns out to be wrong, blocked, or needs rescoping, edit the plan here rather than silently deviating — the plan must always reflect reality.

---

## STATUS

```
Current phase: 9 COMPLETE. Phases 0, 1, 2, 5, 6, 7, 8, 9 are all done with real, measured results. Only Phase 3 (CUDA kernel execution) and Phase 4 (benchmarking) remain, both genuinely blocked on the user running scripts/colab_sync.md on a real CUDA GPU.
Last completed task:
  - REWORK of Phases 1-2: replaced hand-drawn synthetic shapes with ModelNet10 (real public CAD dataset, no signup) after user feedback that fake data "makes people lose interest." Both modalities derived from the same real mesh per example. See docs/dataset_rework_notes.md.
  - Found and fixed a REAL BUG, not just a tuning problem: model/fusion.py had NUM_CLASSES=4 hardcoded (leftover from the old 4-shape synthetic dataset), silently capping the model's output head at 4 of ModelNet10's 10 real classes. Found via ablation (vision-only=68.9%, lidar-only=87.7%, but the fused model scored only 35% -- worse than either alone, which is what made it clearly a bug and not just undertraining). Full story in docs/fusion_bug_notes.md. Fixed by importing NUM_CLASSES from the dataset's class list everywhere, plus added a regression test (test_fusion_num_classes_matches_real_dataset) so this exact bug class can't silently reappear.
  - Also fixed real overfitting (separate from the bug above): 5x random-viewpoint data augmentation (train split only), inverse-frequency class-weighted loss (8x imbalance between chair/bathtub), L2 weight decay, LR decay at epoch 30, best-val-accuracy checkpointing, image resolution bumped 32->40px.
  - FINAL RESULT after all fixes: 86.01% best validation accuracy (10-class real object classification, ModelNet10), consistent with (and the fused model now sensibly between/around) the single-modality ablation numbers. Checkpoint saved at model/checkpoints/model.pkl.
  - Re-profiled the corrected model (docs/phase2_profiling_notes.md updated): self-attention in the vision encoder is still the dominant single op (3.16ms/block of 6.49ms vision-encoder time, out of 7.44ms total forward pass) -- bottleneck finding unchanged, just updated numbers for the new 40x40/101-token sequence length.
  - Phase 3 kernel code written (untested): kernels/reference.py (verified against real model math, diff 3.5e-10), kernels/fused_attention.cu (FlashAttention-style: K/V loaded into shared memory once per (batch,head) block, never materializes the [N,N] score matrix in global memory), kernels/custom_call.cpp (XLA custom-call glue), kernels/register.py (JAX jax.extend.ffi registration), kernels/tests/test_correctness.py (skips cleanly without a built .so), scripts/colab_sync.md (Colab build/debug steps, written expecting first-build compile errors).
  - Full test suite: 13 passed, 3 skipped (the 3 GPU-dependent kernel tests, correctly skip on CPU-only machines) -- data_pipeline (7), model (5), kernels/reference (1).
Next task: NOTHING pending on this machine. The only remaining work is Phase 3/4 execution, which requires the user to run scripts/colab_sync.md on Colab. When they report back what broke/worked and real benchmark numbers, update kernels/, docs/benchmark_results.md (Phase 4), docs/edge_dual_path_notes.md (real CUDA-path latency), docs/final_report.md's tables, and this STATUS block. Do not fabricate GPU numbers in the meantime.
Blockers: Phase 3 execution AND Phase 4 (benchmarking) remain blocked on the user's Colab session (by design). Every other phase is complete with real, measured results, including Phase 9 (zero-shot classification), added after user feedback that the project needed a genuine technical contribution beyond infra glue. docs/final_report.md should be updated to mention Phase 9's result in its headline numbers table if revisited.
Extra context for whoever resumes this: a real k3d cluster ("fenris") is currently UP on this machine with 3 nodes and 2 live Deployments (fenris-inference-stable on v2, fenris-inference-canary on v2) -- check `kubectl get pods` / `k3d cluster list` before assuming it needs to be recreated. Docker image `fenris-inference:v1` exists locally and inside the cluster (now includes /hard_cases endpoint). `data_pipeline/telemetry/hard_cases.json` and `model/checkpoints/model_finetuned.pkl` are real artifacts from the Phase 7 run, not placeholders.
Budget spent so far: $0.00 / $10.00
Last updated: 2026-10-08
```

---

## 0. What this project is (recap, so no re-deriving is needed)

A compiler + runtime that takes a multimodal perception model (camera + lidar fusion) and compiles it for heterogeneous edge hardware, with a Kubernetes control plane that manages fleet-wide model rollout, canarying, and a data pipeline that mines hard cases from telemetry to close the retraining loop.

**Hardware/budget reality (binding constraint, do not violate):**
- Primary machine: MacBook Pro M1, 32GB RAM, Apple Silicon (ARM64) — **no CUDA-capable GPU, ever.**
- All CUDA kernel development/testing/benchmarking happens remotely: Google Colab free tier (T4 GPU, $0) first; a paid rented GPU (vast.ai/RunPod spot, ~$0.15–0.40/hr) only for the final cross-hardware benchmark, capped at **$10 total**.
- The M1 itself is the real ARM edge device — not simulated. This is a feature, not a workaround.
- Kubernetes fleet = local K3s/k3d cluster on the M1 (ARM nodes, real) + optionally one remote GPU node for true heterogeneity. No real robots, no real fleet.
- **Data: ModelNet10, a real public research dataset — not hand-drawn/invented.** nuScenes/KITTI require account registration before download, which breaks full automation. An earlier version of this project used hand-drawn synthetic shapes as a stand-in, but that read as an unconvincing toy and was dropped. **ModelNet10** (real CAD models of 10 real-world object categories: bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet; used in real published work like PointNet/VoxNet) is directly downloadable with no signup (~473MB, from `3dvision.princeton.edu`). Both modalities are derived from the same real 3D mesh per example: the "lidar" point cloud is sampled directly from the real mesh surface, and the "camera" image is a 2D projection of that same real geometry from a random viewpoint — both views are grounded in a real object, not invented. Uses ModelNet10's official train/test split, not a custom one.

**Scope level (binding constraint, do not violate):** this should read as "a strong fresher's project, pushed a bit further" — not a research-lab-scale system. Concretely: small models (thousands-to-low-millions of params, not billions), simple synthetic data (not a large real-world dataset), one CUDA kernel (not a kernel library), a 3-node local cluster (not a real fleet), a plain Python polling script for the canary controller (not a custom K8s operator/CRD). If at any point a task starts requiring more than ~150-250 lines of new code to add one piece, stop and simplify the approach rather than pushing through.

**Guiding principle for implementation style:** basic, readable, human-level code. No premature abstraction, no defensive code for cases that can't happen, no gold-plating. Each phase should produce something that *runs end to end*, however small, before adding the next layer.

---

## 1. Architecture

```
[Data Pipeline]  -->  [Model + Compiler]  -->  [Edge Runtime]  -->  [Fleet Control Plane]
 synthetic shapes      JAX fusion model         dual-path:            K8s (K3s/k3d)
 generator + shard      profiled bottleneck       - CUDA path (cloud)   rollout / canary /
 hard-case mining      custom CUDA kernel        - ARM/CPU path (M1)   rollback, fed by
                        (XLA custom-call)                              telemetry -> back
                                                                        into data pipeline
```

## 2. Folder structure (already created)

```
fenris/
  PLAN.md              <- this file, the persistent memory/checklist
  data_pipeline/        synthetic data generator, shard builder, hard-case miner
  model/                 JAX fusion model (vision encoder + lidar encoder + fusion head)
  kernels/               CUDA kernel source (.cu/.cpp), XLA custom-call glue
  edge/                  dual-path runtime: cuda_path/ and arm_path/
  benchmarks/            benchmark scripts + results (latency, precision, correctness)
  k8s/                   K3s manifests, canary/rollback controller, telemetry collector
  docs/                  design notes, phase writeups, final report
  scripts/               setup/helper shell scripts (env setup, colab sync, etc.)
```

## 3. Phases

Each phase has a **Definition of Done** — a concrete, testable condition. If you stop after any phase, the system up to that point should run and demonstrate something real; later phases build on top without needing rework (if a later phase ever requires reworking an earlier one, that's a planning bug — fix the plan, don't silently patch around it).

---

### Phase 0 — Environment & scaffold
**Goal:** project skeleton, dependencies installable on M1.

- [x] Set up Python env (venv) on M1 with: `jax` (CPU backend), `numpy`, `pillow` (for drawing synthetic images), `pytest`.
- [x] Write `scripts/setup_env.sh` that reproduces the env from scratch.
- [x] Write `.gitignore` (venv, checkpoints, __pycache__, generated data/shards).
- [x] `git init` the project, first commit.

**Definition of Done:** `python -c "import jax; print(jax.devices())"` runs on M1 without error. ✅ DONE — jax 0.11.2, `[CpuDevice(id=0)]`.

---

### Phase 1 — Data pipeline (synthetic multimodal data)
**Goal:** generate a small synthetic multimodal dataset and turn it into training shards, plus a hard-case mining script (used later, in Phase 7).

- [x] `data_pipeline/generate.py`: for each of 4 shape classes (circle, square, triangle, star), generate (a) a small 2D rendered image ("camera") with random position/rotation/noise, and (b) a 3D point cloud sampled from that shape's surface with random rotation/noise ("lidar"). Save a few thousand examples total — this is intentionally small.
- [x] `data_pipeline/build_shards.py`: write (image, pointcloud, label) tuples to disk shards as simple `.npz` files — no need for a real distributed format at this scale.
- [x] `data_pipeline/hard_case_miner.py`: stub now, filled in during Phase 7 — given model predictions + confidence scores, flag low-confidence examples. Write the interface now so Phase 2's model output shape matches what this expects.
- [x] Basic test: `pytest data_pipeline/tests/` — shard round-trip (write then read back, shapes match) and a sanity check that each class's generated examples look visually distinct.

**Definition of Done:** running `python data_pipeline/generate.py && python data_pipeline/build_shards.py` produces N shards in `data_pipeline/shards/`, and a small script can load one shard and print tensor shapes. ✅ DONE — 3000 examples generated (750/class, balanced), 6 train shards + 1 val shard, 4/4 pytest tests pass. Sample star image visually confirmed correct.

---

### Phase 2 — Baseline multimodal model (JAX, runs on M1 CPU)
**Goal:** a small, real multimodal fusion model, trainable on M1 CPU (small enough to be feasible without a GPU).

- [x] `model/vision_encoder.py`: tiny ViT (2 transformer blocks, 4x4 patches, embed dim 48 — small, CPU-trainable).
- [x] `model/lidar_encoder.py`: simple point-cloud encoder (PointNet-style: per-point MLP + max-pool for a pooled embedding, plus `forward_per_point` exposing per-point features used as cross-attention tokens).
- [x] `model/fusion.py`: real multi-token cross-attention (vision CLS embedding as query, lidar per-point features as keys/values) + linear classification head (10-class real object classification, ModelNet10 — originally shipped with a hardcoded 4-class head left over from the synthetic dataset, see `docs/fusion_bug_notes.md` for the bug and fix).
- [x] `model/train.py`: training loop on shards from Phase 1, Adam optimizer (hand-written, no optax dependency), class-weighted loss, L2 weight decay, LR decay, best-val-accuracy checkpointing, runs on CPU.
- [x] Train to a sane, reported-honestly accuracy number — **86.01% best val accuracy** on real ModelNet10 data (10 classes, 19,955 augmented train examples, 908 held-out val objects), 50 epochs, ~15 min total on M1 CPU. (Superseded an earlier buggy run that scored only 35% due to the NUM_CLASSES bug above — see `docs/fusion_bug_notes.md`.)
- [x] `model/profile.py`: profiled forward pass (JIT, warmup-excluded timing), identified the bottleneck. **Documented in `docs/phase2_profiling_notes.md`.**

**Definition of Done:** model trains end-to-end on M1 CPU for at least a few epochs, produces a checkpoint, and `model/profile.py` outputs a clear bottleneck finding documented in `docs/phase2_profiling_notes.md`. ✅ DONE — vision encoder is 87.3% of forward-pass time; within it, self-attention (QKV+softmax+out-proj) is 3.16ms/block vs 0.88ms for the MLP, the clear fusion target for Phase 3. 13/16 tests passing (3 GPU-dependent kernel tests correctly skip on CPU).

---

### Phase 3 — CUDA kernel + XLA custom-call integration (remote GPU, mostly Colab)
**Goal:** a hand-written fused CUDA kernel for the bottleneck found in Phase 2, callable from JAX via XLA custom-call.

- [ ] `scripts/colab_sync.md`: short notes on the workflow (edit locally, push to git, pull in Colab, run there).
- [ ] `kernels/fused_op.cu`: the CUDA kernel implementing the fused op.
- [ ] `kernels/custom_call.cpp`: C++ glue matching XLA's custom-call ABI.
- [ ] `kernels/register.py`: JAX-side registration (`jax.extend.ffi` or equivalent) so the fused op is a normal JAX primitive, usable inside `jit`.
- [ ] `kernels/tests/test_correctness.py`: property-based test — compare kernel output vs. the plain-JAX reference op across random inputs at fp32/fp16/int8. This runs on Colab (needs GPU).
- [ ] Debug on Colab free T4 until correctness tests pass and no race conditions (`compute-sanitizer` if available in the Colab runtime).

**Definition of Done:** the fused kernel passes the correctness suite on Colab's T4, and swapping it into the Phase 2 model (run on Colab, not M1) produces identical-enough outputs to the un-fused version.

---

### Phase 4 — Benchmarking
**Goal:** honest, documented latency/precision numbers.

- [ ] `benchmarks/run_latency.py`: compare stock JAX/XLA vs. custom kernel, at batch size 1 and batched, across fp32/fp16/int8. Record p50/p99, not just mean.
- [ ] `benchmarks/run_cross_hardware.py`: run the same suite on a second GPU architecture — this is where the **$10 budget** gets spent (rented vast.ai/RunPod instance, short session, log the actual cost spent in `STATUS` above).
- [ ] `docs/benchmark_results.md`: write up the numbers, including where the custom kernel does NOT win (be honest — this is a credibility feature, not a weakness).

**Definition of Done:** a results table exists with real measured numbers from at least 2 GPU architectures, committed to `docs/benchmark_results.md`. Budget spent logged.

---

### Phase 5 — Edge dual-path runtime
**Goal:** the model runs two real ways: GPU-accelerated (cloud/server) and native ARM (the M1 itself).

- [x] `edge/cuda_path/`: inference entrypoint using the Phase 3 kernel (runs on Colab/rented GPU). Shares `model/vision_encoder.py`'s `forward(..., backend="cuda_kernel")` — one model, swappable backend, not two separate models. On the M1 it correctly reports "kernel not built, needs a GPU" instead of crashing confusingly.
- [x] `edge/arm_path/`: inference entrypoint using JAX's CPU backend — runs natively on the M1 (real, not simulated). **Measured: 0.451ms/inference, correct prediction on a real val example.**
- [x] `edge/compare_paths.py`: run both paths on the same input, compare latency and output, document the tradeoff in `docs/edge_dual_path_notes.md`.
- [x] `edge/tests/test_edge.py`: 2 tests (ARM path runs and predicts in valid range; CUDA path fails cleanly without a GPU).

**Definition of Done:** both paths run inference on the same trained checkpoint and produce matching-enough outputs; latency difference is documented honestly. ✅ PARTIALLY DONE — ARM path fully real and measured; CUDA path is written and correctly reports its own unavailability on this hardware. Full side-by-side (including the real CUDA latency number) pending Colab results, same blocker as Phase 3/4. 15/18 tests passing (3 GPU-dependent skip).

---

### Phase 6 — Kubernetes fleet control plane
**Goal:** a real, local K3s/k3d cluster that manages model rollout across simulated heterogeneous nodes.

- [x] `scripts/setup_k3s.sh`: real local k3d cluster, 1 server + 2 agents, all genuinely ARM64 (confirmed via `kubectl get nodes`, since this runs on the M1's own silicon). agent-0 labeled `node-type=arm` (real); agent-1 labeled `node-type=gpu-simulated, gpu=false` (explicitly marked simulated in the label itself — no real GPU node available yet, blocked on Colab per Phase 3/4).
- [x] `k8s/Dockerfile` + `k8s/serve.py`: real Docker image (`fenris-inference:v1`, ~1GB) bundling the actual trained checkpoint(s) and a minimal stdlib HTTP server that computes **real, live accuracy** inside the pod on a held-out shard (not a hardcoded number) — `/health` and `/accuracy` endpoints.
- [x] `k8s/manifests/stable.yaml` + `canary.yaml`: real Deployments+Services, both scheduled via `nodeSelector: node-type=arm` (standard canary practice: test on the same hardware class as production). Both verified `Ready` via `kubectl wait`.
- [x] `k8s/canary_controller.py`: real controller using plain `kubectl` subprocess calls (patch deployment env, wait for rollout status, port-forward + poll live `/accuracy` over real HTTP) — no mocking, no Kubernetes client library needed at this scale.
- [x] `scripts/build_bad_checkpoint.py`: generates a genuinely untrained (random-init) checkpoint, baked into the same image, selected via `MODEL_PATH` env var.
- [x] Test: deliberately deployed the bad checkpoint as canary — **actually ran**, not simulated.

**Definition of Done:** a demo-able sequence: deploy v1 → deploy v2 (good) canaries and promotes → deploy v3 (deliberately bad) canaries and auto-rolls-back. Recorded as a short script/log, not just claimed. ✅ DONE, real run (see `docs/k8s_canary_demo.md` for full logs): v1→v2 canary measured 0.8300 live accuracy vs stable's 0.8300 → **promoted**, stable now serving v2. v2→v3-bad canary measured **0.1050** live accuracy (untrained checkpoint, right at the 10-class random-chance baseline) vs stable's 0.8300 → **rolled back**, confirmed via `kubectl get deployment -o jsonpath` that stable never changed from v2. 17/20 tests passing (3 GPU-dependent kernel tests skip, everything else — including a live-cluster integration test — passes for real).

---

### Phase 7 — Telemetry → hard-case mining closed loop
**Goal:** close the loop — field telemetry feeds back into the data pipeline.

- [x] `k8s/serve.py`: added real `/hard_cases` endpoint (uses `data_pipeline/hard_case_miner.py`'s `find_hard_cases`, no longer just a stub) — rebuilt image, redeployed to the live cluster, verified live: flagged 15/200 examples as hard cases from a real running pod.
- [x] `k8s/telemetry_collector.py`: polled both live services (`fenris-inference-stable`, `fenris-inference-canary`) over real `kubectl port-forward` + HTTP, wrote aggregated report to `data_pipeline/telemetry/hard_cases.json`.
- [x] `model/retrain_on_hard_cases.py`: mined hard cases from the full 908-example val pool (70 found, 7.7% — consistent with the live pods' 7.5%, a nice cross-check), split 35/35 into mine/held-out-eval, brief fine-tune (8 epochs) on train shards + oversampled mined cases.

**Definition of Done:** one full loop demonstrated: inference flags a hard case → it's mined → it's added to a retraining shard → retrained checkpoint shows a measured (even if modest) change on a held-out hard-case set. ✅ DONE, real measured numbers (full story in `docs/hard_case_mining_notes.md`): held-out hard-case eval accuracy **25.71% → 37.14% (+11.43 points)**, overall val accuracy barely moved (86.01% → 86.34%, no regression). New checkpoint saved separately as `model_finetuned.pkl` (deployed checkpoint not silently swapped). 19/22 tests passing (3 GPU-dependent skip).

---

### Phase 8 — Polish & writeup
**Goal:** make the finished project legible to an interviewer.

- [x] `docs/final_report.md`: architecture, measured numbers table, real-vs-simulated table, the NUM_CLASSES bug story, lessons learned.
- [x] `README.md`: quickstart, real-vs-simulated table, project structure, links to every doc.
- [x] `scripts/demo_canary.sh`: one-shot script that resets the live cluster to v1, runs v1→v2 (promote) and v2→v3-bad (rollback), prints final deployment state. **Actually re-run as part of this phase** — reproduced the exact documented result (promote then rollback, stable left on v2 throughout).

**Definition of Done:** someone unfamiliar with the project can read `docs/final_report.md` and `README.md` and understand exactly what's real, what's simulated, and why each engineering decision was made. ✅ DONE.

---

### Phase 9 — Zero-shot classification (added after user feedback that the project needed a genuine novel contribution, not just infra glue)
**Goal:** demonstrate real generalization to classes never seen during training — not a toy, a real held-out-class experiment.

- [x] `model/build_class_embeddings.py`: real GloVe (`glove-wiki-gigaword-50`, 400K vocab) word embeddings for all 10 ModelNet10 class names, cached to `model/class_embeddings.npy` so serving/inference never needs the gensim dependency.
- [x] `model/zero_shot_head.py`: DeViSE-style embedding-matching head — projects the fused embedding into GloVe space, classifies by cosine similarity against class-name embeddings (not a fixed per-class weight vector, which is what makes unseen classes representable at all).
- [x] `model/fusion.py` refactored to expose `cross_attend()` (pre-head fused embedding) separately from `forward()` (the original softmax head) — both heads now share the same backbone.
- [x] `model/train_zero_shot.py`: held out `desk` and `night_stand` **entirely** from training (zero training images), trained on the remaining 8 classes, evaluated on all 10.
- [x] Found a real negative result and diagnosed it properly (not just reported it): raw zero-shot accuracy was 0.58%, *below* the 10% random baseline — inspected actual per-example similarity scores (not just the aggregate number) and identified the published "seen-class bias / hubness" problem (seen classes act as attractors for visually-similar unseen ones).
- [x] `model/calibrate_zero_shot.py`: implemented calibrated stacking (Chao et al., ECCV 2016), a real published correction — swept the calibration scalar and got zero-shot accuracy up to **24.42%** (2.4x random chance) at the cost of seen-class accuracy, with a balanced operating point at gamma=4.0 (65.9% seen / 19.8% zero-shot).
- [x] `model/tests/test_zero_shot.py`: shape/normalization tests for the new head.

**Definition of Done:** a genuine held-out-class generalization result, honestly reported including the failure mode, the diagnosis, and the real tradeoff curve of the fix — not a single cherry-picked number. ✅ DONE. Full writeup in `docs/zero_shot_notes.md`. 25/28 tests passing (3 GPU-dependent skip).

**Note on scope:** the user's feedback also raised sink-token sparse attention (tied to the Phase 3 kernel), MoE routing, and speculative decoding for action generation. Sparse attention is a real candidate for future work (see `docs/final_report.md`'s "what's left" section). Speculative decoding for action generation requires a VLA with real action-sequence labels, which don't exist in this project's data (ModelNet10 is static object classification) — building that would mean fabricating action data, which was explicitly flagged as a problem earlier in this project, so it was not attempted.

---

## 4. Non-negotiable honesty rules (carry through every phase)

- Never claim a number that wasn't actually measured in this project.
- Never claim hardware was used that wasn't actually used — if something is simulated (e.g. a "GPU node" that's actually a labeled CPU node), say so explicitly in `docs/`.
- If a phase's Definition of Done can't be met as scoped, shrink the scope and update this file — don't fake the result.

## 5. Stopping points

The project is designed so that stopping after **Phase 2** (data pipeline + trained multimodal model) is already a legitimate, demoable multimodal ML project. Stopping after **Phase 4** adds real compiler/CUDA depth. Stopping after **Phase 6** adds the full Kubernetes fleet story. Phase 7-8 are the closed-loop/polish layer for maximum depth. Each stopping point is a complete, working system — nothing beyond the last completed phase is required for the project to "work."
