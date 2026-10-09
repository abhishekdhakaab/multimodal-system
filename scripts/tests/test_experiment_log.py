from scripts.experiment_log import load_log

REQUIRED_KEYS = {"phase", "experiment", "change", "metric", "result", "notes"}


def test_log_loads_and_every_row_has_required_fields():
    log = load_log()
    assert len(log) > 0
    for row in log:
        assert REQUIRED_KEYS.issubset(row.keys())
        assert isinstance(row["result"], (int, float))
