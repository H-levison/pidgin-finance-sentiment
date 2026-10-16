"""Summarise the multi-seed final runs as mean and range.

A single training run of a transformer on 3,500 examples moves by a point or two
of F1 between seeds purely from initialisation and shuffling order. Reporting one
number hides that. This reads the seeded rows out of experiments.csv and reports
mean, min and max so the spread is visible.

    python src/seed_summary.py
    python src/seed_summary.py --pattern afro-xlmr-base-weighted-seed --split test
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import RESULTS_DIR

METRICS = ("accuracy", "macro_f1", "weighted_f1", "f1_negative", "f1_neutral", "f1_positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="seed")
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    path = RESULTS_DIR / "experiments.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet.")

    frame = pd.read_csv(path)
    frame = frame.sort_values("run_at").drop_duplicates(
        subset=["experiment", "split"], keep="last"
    )
    subset = frame[
        frame["experiment"].str.contains(args.pattern, na=False)
        & (frame["split"] == args.split)
    ]
    if subset.empty:
        raise SystemExit(
            f"No rows matching pattern '{args.pattern}' on split '{args.split}'."
        )

    lines = [
        f"# Final results across {len(subset)} seeds ({args.split} split)",
        "",
        f"Runs: {', '.join(sorted(subset['experiment']))}",
        "",
        "| metric | mean | min | max | spread |",
        "|---|---|---|---|---|",
    ]
    for metric in METRICS:
        values = subset[metric].astype(float)
        lines.append(
            f"| {metric} | {values.mean():.3f} | {values.min():.3f} | "
            f"{values.max():.3f} | {values.max() - values.min():.3f} |"
        )

    lines += [
        "",
        f"Seed-to-seed spread on macro F1 is "
        f"{subset['macro_f1'].astype(float).max() - subset['macro_f1'].astype(float).min():.3f}. "
        "Any difference between models smaller than this is not a real difference.",
        "",
        "Note: 3 seeds, not the 5 used by the published AfriSenti baselines.",
        "",
    ]

    summary = "\n".join(lines)
    (RESULTS_DIR / "seed_summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
