"""Build a training split with extra neutral examples from other Pidgin corpora.

The neutral class fails because the AfriSenti training split holds 51 neutral
examples against roughly 3,400 negative and positive ones. Class weighting does
not fix that: weighting moves a decision boundary, it does not add evidence.
This script tests the obvious alternative by adding whatever labelled neutral
Pidgin text is publicly available.

NaijaSenti, usually named as the other Pidgin sentiment corpus, is not useful
here: its pcm files are byte-for-byte identical to AfriSenti's. Two smaller
corpora do add examples, both CC BY 4.0:

    WAZOBIALABS/nigerian-pidgin-voice-text   500 rows,  77 neutral
    WAZOBIALABS/nigerian-pidgin-eval         230 rows,  11 neutral

Together that is 88 neutral examples against the 51 already present.

Two caveats that belong in the report. These are transcribed speech, not tweets,
so register and length differ from the training data. And the eval corpus is
distributed as an evaluation set, so it is added to training here only to test
the data-scarcity hypothesis; it is not used to report any score.

    python src/augment_neutral.py
    python src/augment_neutral.py --neutral-only   # add only the neutral rows
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
import urllib.request

import pandas as pd

from config import LABEL2ID, LABELS, PROCESSED_DIR, SEED

SOURCES = [
    ("WAZOBIALABS/nigerian-pidgin-voice-text", "wazobia_pidgin_v07_500.csv"),
    ("WAZOBIALABS/nigerian-pidgin-eval", "wazobia_pidgin_eval_v02.csv"),
]
TEXT_COLUMN = "pidgin_text"
LABEL_COLUMN = "sentiment"


def fetch(repo, filename):
    url = (
        f"https://huggingface.co/datasets/{repo}/resolve/main/"
        f"{urllib.parse.quote(filename)}"
    )
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def load_source(repo, filename):
    frame = pd.read_csv(io.BytesIO(fetch(repo, filename)))
    if TEXT_COLUMN not in frame or LABEL_COLUMN not in frame:
        raise SystemExit(f"{repo}: expected columns {TEXT_COLUMN!r}, {LABEL_COLUMN!r}")
    out = frame[[TEXT_COLUMN, LABEL_COLUMN]].rename(
        columns={TEXT_COLUMN: "text", LABEL_COLUMN: "label"}
    )
    out["text"] = out["text"].astype(str).str.lower().str.strip()
    out["label"] = out["label"].astype(str).str.lower().str.strip()
    out = out[out["label"].isin(LABELS)]
    out = out[out["text"].str.split().str.len() >= 3]
    out["source"] = repo.split("/")[-1]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neutral-only", action="store_true",
                        help="add only neutral rows, leaving the other classes alone")
    parser.add_argument("--base", default="train_dedup")
    parser.add_argument("--out", default="train_augmented")
    args = parser.parse_args(argv)

    base_path = PROCESSED_DIR / f"{args.base}.csv"
    if not base_path.exists():
        raise SystemExit(f"{base_path} is missing. Run src/prepare_data.py first.")
    base = pd.read_csv(base_path)
    print(f"{args.base}: {len(base)} rows, "
          f"{dict(base['label'].value_counts())}")

    extra_frames = []
    for repo, filename in SOURCES:
        frame = load_source(repo, filename)
        print(f"  {repo.split('/')[-1]}: {len(frame)} usable rows, "
              f"{dict(frame['label'].value_counts())}")
        extra_frames.append(frame)

    extra = pd.concat(extra_frames, ignore_index=True)
    extra = extra.drop_duplicates(subset="text")

    if args.neutral_only:
        extra = extra[extra["label"] == "neutral"]

    # Never add anything that appears in dev or test, for the same reason the
    # AfriSenti overlap was removed in the first place.
    held_out = set()
    for split in ("dev", "test"):
        path = PROCESSED_DIR / f"{split}.csv"
        if path.exists():
            held_out |= set(pd.read_csv(path)["text"])
    before = len(extra)
    extra = extra[~extra["text"].isin(held_out)]
    extra = extra[~extra["text"].isin(set(base["text"]))]
    if before != len(extra):
        print(f"  dropped {before - len(extra)} rows already present elsewhere")

    combined = pd.concat(
        [base[["text", "label"]], extra[["text", "label"]]], ignore_index=True
    )
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    combined["label_id"] = combined["label"].map(LABEL2ID)
    out_path = PROCESSED_DIR / f"{args.out}.csv"
    combined.to_csv(out_path, index=False)

    counts = combined["label"].value_counts()
    print(f"\n{args.out}: {len(combined)} rows")
    for label in LABELS:
        base_n = int((base["label"] == label).sum())
        now = int(counts.get(label, 0))
        print(f"  {label:<9} {base_n:>5} -> {now:<5} ({now - base_n:+d})")
    print(f"\nwrote {out_path}")
    print("Train on it with:")
    print(f"  python src/finetune.py --epochs 6 --model Davlan/afro-xlmr-base "
          f"--class-weight --train {args.out} --tag aug")
    return 0


if __name__ == "__main__":
    sys.exit(main())
