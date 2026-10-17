"""Build the worklist of posts the two annotators labelled differently.

Adjudication is a third pass over the disagreements only. Posts where the two
annotators already agree are settled and are not revisited, because reopening
them would let the discussion drift and would quietly turn two independent
passes into one negotiated one.

The sheet is ordered by the kind of disagreement rather than by post id, so the
pair working through it settles one question at a time: all the
neutral-against-positive cases together, then negative-against-neutral, and so
on. Deciding a rule once and applying it to a run of similar posts is faster and
more consistent than arguing each post from scratch.

    python src/make_adjudication_sheet.py
"""

from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from config import REPO_ROOT

ANNOTATION_DIR = REPO_ROOT / "data" / "annotation"
SHEET = ANNOTATION_DIR / "to_label.csv"
SECOND = ANNOTATION_DIR / "second_annotator.csv"
OUT = ANNOTATION_DIR / "adjudication.csv"


def main():
    if not SECOND.exists():
        raise SystemExit(f"{SECOND} is missing.")

    first = pd.read_csv(SHEET, dtype=str).fillna("")
    second = pd.read_csv(SECOND, dtype=str).fillna("")
    for frame in (first, second):
        frame["id"] = frame["id"].astype(str)

    merged = first[["id", "text", "label"]].merge(
        second[["id", "label"]], on="id", suffixes=("_1", "_2")
    )
    merged = merged[(merged.label_1 != "") & (merged.label_2 != "")]

    disagree = merged[merged.label_1 != merged.label_2].copy()
    disagree["disagreement"] = disagree.apply(
        lambda r: " vs ".join(sorted([r.label_1, r.label_2])), axis=1
    )

    # Group the same kind of disagreement together: settle the rule once, then
    # apply it down the run.
    order = [pair for pair, _ in Counter(disagree.disagreement).most_common()]
    disagree["group"] = disagree.disagreement.map({p: i for i, p in enumerate(order)})
    disagree = disagree.sort_values(["group", "id"]).drop(columns="group")

    disagree["agreed"] = ""
    disagree["reason"] = ""
    disagree[["id", "disagreement", "label_1", "label_2", "agreed", "reason", "text"]].to_csv(
        OUT, index=False, encoding="utf-8"
    )

    agreed_count = int((merged.label_1 == merged.label_2).sum())
    print(f"both annotators labelled : {len(merged)}")
    print(f"  already agreed         : {agreed_count} (left alone)")
    print(f"  to adjudicate          : {len(disagree)}")
    print()
    for pair, n in Counter(disagree.disagreement).most_common():
        print(f"  {pair:<24} {n:>3}")
    print(f"\nwrote {OUT}")
    print("Fill the 'agreed' column with: negative, neutral, positive, exclude")
    print("or 'unresolved' if you genuinely cannot settle it.")
    print("Put the rule you applied in 'reason' - that becomes the guideline fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
