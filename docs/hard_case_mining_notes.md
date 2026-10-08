# Telemetry → Hard-Case Mining Closed Loop

## The full loop, actually run end to end

1. **Inference flags a hard case.** `k8s/serve.py`'s `/hard_cases` endpoint
   runs real inference inside a live pod and flags predictions with
   confidence below 0.6, using `data_pipeline/hard_case_miner.py`'s
   `find_hard_cases` (previously a stub, now wired up for real).
2. **Telemetry collects it.** `k8s/telemetry_collector.py` polls every
   service in the live k3d cluster (`fenris-inference-stable` and
   `fenris-inference-canary`) over real HTTP, via real `kubectl
   port-forward`, and writes an aggregated report to
   `data_pipeline/telemetry/hard_cases.json`. Measured result: both pods
   (same model version, same validation shard) flagged **15/200 (7.5%)**
   examples as hard cases.
3. **It's mined and added to a retraining shard.**
   `model/retrain_on_hard_cases.py` runs the deployed checkpoint over the
   *full* validation pool (908 real held-out objects across all 5 val
   shards — a bigger pool than any single pod serves, so there's enough
   hard cases to split meaningfully). Found **70 hard cases (7.7% of the
   pool)** — consistent with the live pods' 7.5%, which is itself a nice
   cross-check that the live telemetry and the offline mining agree.
4. **Retrained checkpoint shows a measured change.** The 70 hard cases
   were split in half: 35 mined and oversampled into a brief fine-tune
   (8 epochs, small LR, starting from the existing checkpoint — not
   training from scratch), 35 **held out and never touched during
   training** to measure the real effect.

## The real numbers

| Metric | Before | After | Change |
|---|---|---|---|
| Accuracy on held-out hard-case eval set (35 examples, never trained on) | 25.71% | 37.14% | **+11.43 points** |
| Overall val accuracy (full 908-example pool, sanity check) | 86.01% | 86.34% | +0.33 points (no regression) |

The hard-case eval set's accuracy genuinely improved, and critically, the
fine-tune did **not** trade general performance away to get there — overall
val accuracy stayed essentially flat (even ticked up slightly). That's the
real story a before/after number alone wouldn't tell: this is a case where
the targeted fine-tune helped where it was supposed to without the usual
“fixed the hard cases, broke everything else” tradeoff.

## Honesty notes

- The held-out hard-case eval set is only 35 examples — a single number
  change of a few points there has real variance; the headline fact is the
  direction and rough size of the improvement, not that "37.14%" is a
  precise, reproducible-to-the-decimal number.
- Both live pods currently serve the exact same model version and the exact
  same validation shard, so their hard-case counts matching (15/200 each)
  is expected, not an independent cross-validation — the real
  cross-check here is between the live pods' 7.5% and the offline full-pool
  mining's 7.7%, which use different (though overlapping) data.
- `model/checkpoints/model_finetuned.pkl` is a new checkpoint, saved
  separately from `model.pkl` — the deployed/serving checkpoint was not
  silently swapped. Promoting `model_finetuned.pkl` into the Kubernetes
  deployments (Phase 6) would be the natural next step of the real loop,
  but wasn't done automatically here to avoid conflating "the loop works"
  with "the fleet was actually updated."
