"""
Stub for Phase 7. Given model predictions and their confidence scores,
flags low-confidence examples as "hard cases" worth adding to a retraining
shard. Not wired up to anything yet — the interface is fixed now so Phase 2's
model output shape (per-class probabilities) matches what this expects.
"""


def find_hard_cases(probs, labels, threshold=0.6):
    """
    probs: [N, num_classes] float array of predicted class probabilities
    labels: [N] int array of true labels (used only to report whether the
        low-confidence prediction was also wrong, for logging purposes)
    threshold: examples where the predicted class's probability is below
        this are flagged as hard cases

    Returns: list of indices into probs/labels that are hard cases.
    """
    import numpy as np

    pred_class = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    hard_idx = np.where(confidence < threshold)[0]
    return hard_idx.tolist()
