import os

import pytest

from model.retrain_on_hard_cases import load_all_val, SHARDS_DIR

VAL_SHARDS_DIR = os.path.join(SHARDS_DIR, "val")


@pytest.mark.skipif(not os.path.isdir(VAL_SHARDS_DIR), reason="run data_pipeline/build_shards.py first")
def test_load_all_val_combines_all_shards():
    images, points, labels = load_all_val()
    assert images.shape[0] == points.shape[0] == labels.shape[0]
    assert images.shape[0] > 0
