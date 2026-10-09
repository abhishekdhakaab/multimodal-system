"""
Minimal structured experiment tracking for this project -- not a full
MLflow/W&B setup, but a real, append-only, machine-readable log
(docs/experiments_log.json) instead of results scattered across
markdown prose and terminal output. Every real number quoted in this
project's docs traces back to a row here.

Usage:
    python -m scripts.experiment_log                 # print as a table
    python -m scripts.experiment_log --add            # append a new entry (see add_entry)
"""

import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments_log.json")


def load_log():
    with open(LOG_PATH, "r") as f:
        return json.load(f)


def add_entry(phase, experiment, change, metric, result, notes):
    """Appends one row and rewrites the log file. Call this from any
    training/eval script going forward instead of only printing to stdout."""
    log = load_log()
    log.append(
        {
            "phase": phase,
            "experiment": experiment,
            "change": change,
            "metric": metric,
            "result": result,
            "notes": notes,
        }
    )
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"logged: {experiment} -> {metric}={result}")


def print_table():
    log = load_log()
    print(f"{'phase':>8}  {'experiment':<55} {'metric':<20} {'result':>8}")
    print("-" * 100)
    for row in log:
        print(f"{str(row['phase']):>8}  {row['experiment']:<55} {row['metric']:<20} {row['result']:>8}")


if __name__ == "__main__":
    print_table()
