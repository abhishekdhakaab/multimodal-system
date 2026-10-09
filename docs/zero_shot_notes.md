# Zero-Shot Classification via Real Word Embeddings

## Motivation

The original model has a fixed 10-way softmax head — structurally incapable
of recognizing any object class it wasn't trained on. This replaces that
with a DeViSE-style (Frome et al., 2013) embedding-matching head: the
fused vision+lidar embedding is projected into real GloVe word-embedding
space, and classification is nearest-neighbor search against class-name
embeddings, computed by cosine similarity. Because the lookup table is
just embeddings, not learned per-class weights, it can include classes
the model has **never seen a single training image of**.

**Real data used, no fabrication:** GloVe (`glove-wiki-gigaword-50`, a
public pretrained embedding set, 400K vocab, downloaded via `gensim`) for
the class-name embeddings, and 2 of ModelNet10's 10 real classes
(`desk`, `night_stand`) held out entirely from training — not synthetic
stand-ins, real objects the model genuinely never trained on.

## Experiment 1: raw zero-shot result

Trained on the 8 seen classes (17,955 augmented examples), evaluated on
all 10 (seen classes normally, held-out classes zero-shot):

| Metric | Result |
|---|---|
| Seen-class accuracy (8 trained classes) | 96.06% |
| **Zero-shot accuracy (2 never-trained classes)** | **0.58%** |
| Random baseline (10-way) | 10.00% |

The raw zero-shot number is *worse than random chance* — not a subtle
underperformance, a near-total failure. Worth investigating rather than
reporting as "zero-shot doesn't work here."

## Diagnosis: seen-class bias ("hubness")

Inspected actual per-example similarity scores for held-out examples
(not just the aggregate accuracy number):

```
true=night_stand  top3=[('dresser', 5.62), ('desk', -1.16), ('sofa', -1.92)]
true=night_stand  top3=[('table', 6.93), ('chair', 5.77), ('night_stand', 5.48)]
true=desk         top3=[('sofa', 6.57), ('chair', 2.86), ('dresser', 2.37)]
```

The true class is sometimes close in similarity score (`night_stand` at
5.48 vs the winning `table` at 6.93 in one example) but never wins the
argmax. This is the documented **seen-class bias / hubness problem** in
zero-shot learning: because training only ever pushes the projection to
produce high similarity in the 8 seen classes' directions, those classes
act as "hubs" that win by default for any visually-similar unseen input
(`dresser`, visually and semantically close to `night_stand`, is a
repeat offender above) — not a code bug, a known, published failure mode.

## Experiment 2: calibrated stacking correction

Applied calibrated stacking (Chao et al., ECCV 2016): subtract a single
scalar `gamma` from every seen-class similarity score before argmax, to
level the playing field for unseen classes.

| gamma | seen accuracy | zero-shot accuracy |
|---|---|---|
| 0.0 | 96.06% | 0.58% |
| 2.0 | 90.62% | 9.30% |
| 4.0 | 65.90% | 19.77% |
| 6.0 | 23.23% | 23.84% |
| **8.0+** | ~0% | **24.42%** |

Two honest readings of this table:
- **Best balanced operating point** (harmonic mean of seen/zero-shot
  accuracy): `gamma=4.0`, seen=65.9%, zero-shot=19.8% — a real, usable
  tradeoff point if both matter.
- **Zero-shot ceiling observed**: `gamma≈8`, zero-shot accuracy plateaus
  at **24.42%** — 2.4x random chance on 2 classes the model never saw a
  single training image of — but seen-class accuracy has collapsed to
  near zero at that point, since the correction is a single global
  scalar with no per-class nuance.

## Honesty notes

- `gamma` here is swept and reported directly against the real zero-shot
  accuracy, which is a diagnostic exploration of the technique, not a
  properly cross-validated hyperparameter search (that needs a held-out
  set of *labeled unseen* examples distinct from the real evaluation set,
  which this project's data size doesn't comfortably support without
  compromising the eval set's integrity).
- The headline zero-shot number depends entirely on which operating point
  you pick — there is no single "zero-shot accuracy" for this model, only
  a tradeoff curve. Reporting just one number without the curve would be
  misleading.
- This whole investigation (bias discovery via per-example inspection,
  not just an aggregate metric, then a real published mitigation,
  measured honestly) is the actual point of this experiment — not the
  specific accuracy number.
