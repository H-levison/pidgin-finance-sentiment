"""Measure how reliable the finance test labels are.

A test set labelled by one person is a weakness, and the honest way to handle it
is to measure how much the labels can be trusted rather than hope nobody asks.

Cohen's kappa is used rather than raw agreement because raw agreement flatters
an unbalanced label set: if 70% of posts are negative, two annotators guessing
negative every time would agree 70% of the time while adding no information.
Kappa measures agreement above what chance would produce.

Rough reading of kappa, following Landis and Koch (1977): below 0.20 is slight,
0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, above 0.80 almost
perfect. Report the number and the interpretation, not just the number.

Expects a second file with the same `id` column and that person's labels:

    data/annotation/second_annotator.csv   columns: id,label

    python src/annotation_agreement.py
    python src/annotation_agreement.py --second data/annotation/my_second_pass.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from config import LABELS, REPO_ROOT, RESULTS_DIR

ANNOTATION_DIR = REPO_ROOT / "data" / "annotation"
SHEET = ANNOTATION_DIR / "to_label.csv"
SECOND = ANNOTATION_DIR / "second_annotator.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--second", default=str(SECOND))
    args = parser.parse_args(argv)

    second_path = REPO_ROOT / args.second if not str(args.second).startswith("/") else args.second

    if not SHEET.exists():
        raise SystemExit(f"{SHEET} is missing.")
    if not pd.io.common.file_exists(str(second_path)):
        raise SystemExit(
            f"{second_path} is missing.\n\n"
            "Ask a second Pidgin speaker to label the rows marked 'yes' in the\n"
            "double column of to_label.csv, using data/annotation/GUIDELINES.md\n"
            "and nothing else, then save their labels as a CSV with columns\n"
            "id,label."
        )

    first = pd.read_csv(SHEET, dtype=str).fillna("")
    second = pd.read_csv(second_path, dtype=str).fillna("")

    for frame in (first, second):
        frame["id"] = frame["id"].astype(str).str.strip()
        frame["label"] = frame["label"].str.strip().str.lower()

    merged = first[["id", "text", "label"]].merge(
        second[["id", "label"]], on="id", suffixes=("_first", "_second")
    )
    merged = merged[
        merged["label_first"].isin(LABELS) & merged["label_second"].isin(LABELS)
    ]

    if merged.empty:
        raise SystemExit("No rows where both annotators gave a usable label.")

    kappa = cohen_kappa_score(merged["label_first"], merged["label_second"])
    raw = float((merged["label_first"] == merged["label_second"]).mean())

    if kappa < 0.20:
        reading = "slight"
    elif kappa < 0.41:
        reading = "fair"
    elif kappa < 0.61:
        reading = "moderate"
    elif kappa < 0.81:
        reading = "substantial"
    else:
        reading = "almost perfect"

    matrix = confusion_matrix(
        merged["label_first"], merged["label_second"], labels=list(LABELS)
    )

    lines = [
        "# Annotation reliability",
        "",
        f"- Posts labelled by both annotators: {len(merged)}",
        f"- Raw agreement: {raw:.1%}",
        f"- Cohen's kappa: {kappa:.3f} ({reading})",
        "",
        "## Where they disagreed",
        "",
        "Rows are annotator 1, columns annotator 2.",
        "",
        "| | " + " | ".join(LABELS) + " |",
        "|---" * (len(LABELS) + 1) + "|",
    ]
    for label, row in zip(LABELS, matrix):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")

    disagreements = merged[merged["label_first"] != merged["label_second"]]
    if len(disagreements):
        lines += ["", "## Disagreement examples", ""]
        for _, row in disagreements.head(10).iterrows():
            lines.append(
                f"- `{row['label_first']}` vs `{row['label_second']}`: "
                f"{row['text'][:110]}"
            )

    summary = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "annotation_agreement.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
