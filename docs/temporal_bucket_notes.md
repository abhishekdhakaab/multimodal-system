# Temporal Bucket/Group Memory — the KV-Cache Idea, Applied to Multi-View Recognition

## What this covers

The one remaining keyword from earlier feedback: "bucket/group
representation of old tokens," the real idea behind KV-cache eviction in
long-context LLM serving (H2O, StreamingLLM — keep recent context exact,
compress older context into fewer summary tokens so memory doesn't grow
unboundedly). This model is single-frame, so there was no sequence to
bucket until this experiment built one.

## The data — real objects, simulated motion, stated honestly

A robot circling a real object captures multiple views over time. There's
no real such video dataset available here, so this is **simulated motion
around a real, static 3D object** — 8 frames per object, each a different
random viewpoint of the *same real ModelNet10 mesh* (reusing the existing
rendering pipeline, just called 8 times per object with different seeds
instead of once). Every individual frame is a real object; the sequence
structure is a reasonable simulation of multi-view capture, not fabricated
identity. 400 train objects, 150 val objects (`model/generate_temporal_sequences.py`).

Each frame is run through the **frozen, already-trained** single-frame
backbone to get its 96-dim fused embedding — the backbone itself is not
retrained here, only a new temporal-aggregation layer is trained on top
of its (cached) outputs.

## The bucketing scheme

8 frames, oldest to newest: the most recent 2 are kept as individual
tokens (full resolution); the older 6 are averaged into 2 groups of 3 —
**8 frames compressed to 4 memory tokens**, regardless of how long a real
deployment's history might grow. (`model/temporal_head.py::bucket_sequence`)

## The experiment — three conditions, each trained fairly

A real methodological trap was caught and fixed before reporting
anything: the first run trained one model on bucketed memory only, then
evaluated it on BOTH bucketed and full memory — meaning "full memory"
was being tested out-of-distribution for a model that never saw 8
individual tokens during training. Fixed by training two separate
models, each trained AND evaluated on its own matching memory regime:

| Condition | Accuracy |
|---|---|
| (a) No memory — latest frame only, existing single-frame model | 82.00% |
| (b) **Bucketed memory** — 8 frames → 4 tokens | **89.33%** |
| (c) Full memory — all 8 frames, no compression | 87.33% |

## The genuinely interesting result

Temporal memory helps a lot (82.00% → 87-89%) — multiple views of the
same object really do carry more information than one. But the more
interesting finding: **bucketed memory (89.33%) matches or slightly beats
full, uncompressed memory (87.33%)**, even in a fair, matched comparison
where distribution mismatch was ruled out.

**Honest interpretation, not overclaimed:** with only 400 training
sequences, the full-memory model has more tokens (and effectively more
degrees of freedom in its attention pattern) to fit with limited data,
while the bucketed model's coarser, lower-variance representation may
act as an implicit regularizer. This is a plausible explanation, not a
proven one — the honest claim from this experiment is **"bucketing
didn't cost anything here, and plausibly helped, in a small-data
regime"**, not "bucketing is universally better than full memory,"
which would need a larger-scale study to actually establish.

## Honest limitations

- Small scale: 400 train / 150 val sequences. A larger study might show
  full memory pulling ahead once there's enough data to use its extra
  capacity without overfitting.
- The backbone is frozen — only the temporal-attention layer is trained.
  A fully fine-tuned version might behave differently.
- "Motion around an object" is simulated via independent random
  viewpoints per frame, not a physically continuous camera trajectory —
  stated plainly, not disguised as real video.
