"""Turn the labelled annotation sheet into the finance test set.

Reads data/annotation/to_label.csv once the label column is filled in, drops the
excluded rows, checks the labels are valid, and writes
data/processed/finance_test.csv in the same format as the AfriSenti splits.

It also prints the numbers the report needs about the test set: how many posts
were collected, how many survived, the exclusion reasons, and the class balance.

    python src/build_finance_test.py
"""

from __future__ import annotations

import sys

import pandas as pd

from config import LABEL2ID, LABELS, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR

SHEET = REPO_ROOT / "data" / "annotation" / "to_label.csv"
OUT = PROCESSED_DIR / "finance_test.csv"


def main():
    if not SHEET.exists():
        raise SystemExit(
            f"{SHEET} is missing. Run src/make_annotation_sheet.py first."
        )

    sheet = pd.read_csv(SHEET, dtype=str).fillna("")
    sheet["label"] = sheet["label"].str.strip().str.lower()

    total = len(sheet)
    unlabelled = sheet[sheet["label"] == ""]
    if len(unlabelled):
        print(f"Warning: {len(unlabelled)} of {total} rows still have no label.")

    valid = set(LABELS) | {"exclude"}
    bad = set(sheet.loc[sheet["label"] != "", "label"]) - valid
    if bad:
        raise SystemExit(
            f"Unrecognised labels in the sheet: {sorted(bad)}. "
            f"Use one of {sorted(valid)}."
        )

    excluded = sheet[sheet["label"] == "exclude"]
    kept = sheet[sheet["label"].isin(LABELS)].copy()

    if kept.empty:
        raise SystemExit("No usable labels yet.")

    kept["label_id"] = kept["label"].map(LABEL2ID)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    kept[["text", "label", "label_id"]].to_csv(OUT, index=False, encoding="utf-8")

    counts = kept["label"].value_counts()
    lines = [
        "# Finance and crypto test set",
        "",
        f"- Posts collected and screened: {total}",
        f"- Excluded by the guidelines: {len(excluded)} "
        f"({100 * len(excluded) / total:.0f}%)",
        f"- Still unlabelled: {len(unlabelled)}",
        f"- Usable test posts: {len(kept)}",
        "",
        "| label | count | share |",
        "|---|---|---|",
    ]
    for label in LABELS:
        count = int(counts.get(label, 0))
        lines.append(f"| {label} | {count} | {100 * count / len(kept):.1f}% |")

    if len(excluded):
        reasons = excluded["notes"].str.strip().replace("", "no reason recorded")
        lines += ["", "## Exclusion reasons", ""]
        for reason, count in reasons.value_counts().items():
            lines.append(f"- {reason}: {count}")

    summary = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "finance_test_stats.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Wrote {len(kept)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
