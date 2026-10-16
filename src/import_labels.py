"""Merge labels collected in the web labeller back into the annotation sheet.

The labelling interface writes one document per (annotator, post) into the
artifact's shared store. Those documents are exported to a directory of JSON
files, and this reads them into data/annotation/to_label.csv so the rest of the
pipeline can carry on unchanged.

If more than one annotator is present, the first argument picks whose labels
become the sheet's own `label` column; the others are written to
`label_<annotator>` columns so the agreement script can use them.

    python src/import_labels.py <directory of exported json>
    python src/import_labels.py <dir> --primary u_UmlIoJ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from config import REPO_ROOT

SHEET = REPO_ROOT / "data" / "annotation" / "to_label.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir")
    parser.add_argument("--primary", default=None)
    args = parser.parse_args(argv)

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        raise SystemExit(f"{export_dir} does not exist")

    records = []
    for path in export_dir.rglob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and doc.get("postId") and doc.get("label"):
            records.append(doc)

    if not records:
        raise SystemExit(f"No label documents found under {export_dir}")

    by_annotator = defaultdict(dict)
    for record in records:
        by_annotator[record.get("annotator", "unknown")][str(record["postId"])] = record

    counts = {a: len(v) for a, v in by_annotator.items()}
    print(f"{len(records)} records from {len(by_annotator)} annotator(s):")
    for annotator, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        tally = Counter(r["label"] for r in by_annotator[annotator].values())
        print(f"  {annotator[:26]}  {n:>4} labels  {dict(tally)}")

    primary = args.primary or max(counts, key=counts.get)
    print(f"\nprimary annotator: {primary[:26]}")

    sheet = pd.read_csv(SHEET, dtype=str).fillna("")
    sheet["id"] = sheet["id"].astype(str)

    primary_labels = by_annotator[primary]
    sheet["label"] = sheet["id"].map(lambda i: primary_labels.get(i, {}).get("label", ""))
    sheet["notes"] = sheet["id"].map(
        lambda i: primary_labels.get(i, {}).get("notes", "") or ""
    )

    for annotator, labels in by_annotator.items():
        if annotator == primary:
            continue
        column = f"label_{annotator[:12]}"
        sheet[column] = sheet["id"].map(lambda i: labels.get(i, {}).get("label", ""))
        overlap = int(((sheet["label"] != "") & (sheet[column] != "")).sum())
        print(f"  second annotator column {column}: {overlap} posts labelled by both")

    sheet.to_csv(SHEET, index=False, encoding="utf-8")

    filled = int((sheet["label"] != "").sum())
    print(f"\nwrote {filled} of {len(sheet)} labels into {SHEET.name}")
    print(f"unlabelled rows: {len(sheet) - filled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
