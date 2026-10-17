"""Build the finance test set from two independent passes plus adjudication.

The set is assembled from two sources:

  - posts both annotators labelled the same way, which stand unchanged;
  - posts they labelled differently, resolved in a joint third pass recorded in
    data/annotation/adjudication.csv.

Posts the two already agreed on are never revisited. Reopening them would turn
two independent passes into one negotiated pass and would destroy the agreement
measurement that justifies the set in the first place.

Writes data/processed/finance_test.csv and data/annotation/adjudicated_gold.csv,
and prints the figures the report quotes.

    python src/build_adjudicated_set.py
"""

from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from config import LABEL2ID, LABELS, PROCESSED_DIR, REPO_ROOT

ANNOTATION_DIR = REPO_ROOT / "data" / "annotation"
SHEET = ANNOTATION_DIR / "to_label.csv"
SECOND = ANNOTATION_DIR / "second_annotator.csv"
ADJUDICATION = ANNOTATION_DIR / "adjudication.csv"
GOLD = ANNOTATION_DIR / "adjudicated_gold.csv"


def load():
    for path in (SHEET, SECOND, ADJUDICATION):
        if not path.exists():
            raise SystemExit(f"{path} is missing.")
    frames = []
    for path in (SHEET, SECOND, ADJUDICATION):
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame["id"] = frame["id"].astype(str)
        frames.append(frame)
    return frames


def main():
    first, second, adjudication = load()
    adjudication["agreed"] = adjudication["agreed"].str.strip().str.lower()

    both = first[["id", "text", "label"]].merge(
        second[["id", "label"]], on="id", suffixes=("_1", "_2")
    )
    both = both[(both.label_1 != "") & (both.label_2 != "")]

    unanimous = both[both.label_1 == both.label_2].copy()
    unanimous["final"] = unanimous.label_1
    unanimous["source"] = "unanimous"

    settled = adjudication[adjudication.agreed != "unresolved"][["id", "agreed"]]
    settled = settled.rename(columns={"agreed": "final"})
    settled = settled.merge(first[["id", "text"]], on="id")
    settled["source"] = "adjudicated"

    gold = pd.concat(
        [unanimous[["id", "text", "final", "source"]],
         settled[["id", "text", "final", "source"]]],
        ignore_index=True,
    ).sort_values("id", key=lambda c: c.astype(int))

    unresolved = int((adjudication.agreed == "unresolved").sum())
    usable = gold[gold.final.isin(LABELS)].copy()

    GOLD.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(GOLD, index=False, encoding="utf-8")

    out = usable[["text", "final"]].rename(columns={"final": "label"})
    out["label_id"] = out["label"].map(LABEL2ID)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DIR / "finance_test.csv", index=False, encoding="utf-8")

    # How far the settled labels sit from each annotator's own. A figure close to
    # 100% for one of them would mean that annotator simply prevailed, which is
    # not adjudication.
    check = gold.merge(both[["id", "label_1", "label_2"]], on="id")
    match_1 = float((check.final == check.label_1).mean())
    match_2 = float((check.final == check.label_2).mean())

    print(f"labelled by both annotators : {len(both)}")
    print(f"  unanimous                 : {len(unanimous)}")
    print(f"  resolved by adjudication  : {len(settled)}")
    print(f"  left unresolved           : {unresolved}")
    print(f"  majority said exclude     : {int((gold.final == 'exclude').sum())}")
    print(f"usable test posts           : {len(usable)}")
    print(f"  {dict(Counter(usable.final))}")
    print(f"  neutral share             : {100 * (usable.final == 'neutral').mean():.1f}%")
    print()
    print(f"settled labels match annotator 1: {100 * match_1:.1f}%")
    print(f"settled labels match annotator 2: {100 * match_2:.1f}%")
    print()
    print(f"wrote {GOLD}")
    print(f"wrote {PROCESSED_DIR / 'finance_test.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
