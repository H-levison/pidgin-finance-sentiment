"""Audit and clean the AfriSenti pcm splits, then write them out as CSV.

Two jobs here. The first is to describe the data honestly: class balance,
duplicates, overlap between splits, text lengths. The second is light cleaning.
The cleaning stays minimal because AfriSenti is already lowercased, stripped of
punctuation and anonymised (mentions replaced with @user, URLs removed), so
aggressive normalisation would only throw away signal.

The audit turned up two problems in the published release that drive the rest of
the project:

1. Train/test leakage. 1,013 training tweets appear verbatim in the test set and
   291 appear in dev, and there are 512 duplicate tweets inside train itself.
   A model trained on the release as-is is partly memorising its own test set.

2. Inconsistent gold labels. On the tweets shared between train and test, the
   two splits disagree on the label 12% of the time. Those tweets cannot all be
   classified correctly no matter how good the model is, which puts a ceiling on
   the achievable score.

So two training files get written rather than one:

  train_full.csv   the release as distributed, whitespace-normalised only. Used
                   so results stay comparable to published AfriSenti numbers,
                   which were computed on these splits.
  train_dedup.csv  duplicates removed and any tweet that also appears in dev or
                   test dropped. Smaller and harder, but the score it produces
                   is an honest estimate of generalisation.

Dev and test are never modified. They are the official evaluation sets and
editing them would make the results incomparable to prior work.

    python src/prepare_data.py
"""

from __future__ import annotations

import csv
import json
import re
import sys

import pandas as pd

from config import LABEL2ID, LABELS, PROCESSED_DIR, RAW_DIR, RESULTS_DIR, SEED

SPLIT_FILES = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"}

# Anything outside basic Latin: mostly emoji and zero-width joiners.
NON_ASCII = re.compile(r"[^\x00-\x7F]")
WHITESPACE = re.compile(r"\s+")


def load_split(name):
    """Read one TSV. QUOTE_NONE because tweets contain stray quote characters
    that pandas would otherwise treat as field delimiters."""
    path = RAW_DIR / SPLIT_FILES[name]
    frame = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    frame = frame.rename(columns={"tweet": "text"})
    if set(frame.columns) != {"text", "label"}:
        raise ValueError(f"{path.name}: unexpected columns {list(frame.columns)}")
    return frame


def normalise(text):
    """Collapse runs of whitespace and trim. Nothing else."""
    return WHITESPACE.sub(" ", str(text)).strip()


def describe(frame, name):
    """The numbers that go into the dataset section of the report."""
    counts = frame["label"].value_counts()
    tokens = frame["text"].str.split().str.len()
    return {
        "split": name,
        "rows": int(len(frame)),
        "labels": {label: int(counts.get(label, 0)) for label in LABELS},
        "label_pct": {
            label: round(100 * counts.get(label, 0) / len(frame), 2) for label in LABELS
        },
        "duplicate_texts": int(frame["text"].duplicated().sum()),
        "tokens_mean": round(float(tokens.mean()), 1),
        "tokens_median": int(tokens.median()),
        "tokens_p95": int(tokens.quantile(0.95)),
        "tokens_max": int(tokens.max()),
        "with_non_ascii": int(frame["text"].str.contains(NON_ASCII).sum()),
        "with_user_tag": int(frame["text"].str.contains("@user", regex=False).sum()),
        "with_http": int(frame["text"].str.contains("http", regex=False).sum()),
        "vocab_size": int(len(set(" ".join(frame["text"]).split()))),
    }


def audit_overlap(train, dev, test):
    """Quantify the leakage and the label disagreement it exposes."""
    dev_texts, test_texts = set(dev["text"]), set(test["text"])

    shared = train.merge(test, on="text", suffixes=("_train", "_test"))
    if len(shared):
        agreement = float((shared["label_train"] == shared["label_test"]).mean())
        disagreements = shared[shared["label_train"] != shared["label_test"]]
        examples = [
            {
                "text": row["text"][:120],
                "label_in_train": row["label_train"],
                "label_in_test": row["label_test"],
            }
            for _, row in disagreements.head(10).iterrows()
        ]
    else:
        agreement, examples = 1.0, []

    return {
        "duplicate_texts_within_train": int(train["text"].duplicated().sum()),
        "train_texts_also_in_dev": int(train["text"].isin(dev_texts).sum()),
        "train_texts_also_in_test": int(train["text"].isin(test_texts).sum()),
        "dev_texts_also_in_test": int(dev["text"].isin(test_texts).sum()),
        "test_rows_reachable_from_train_pct": round(
            100 * test["text"].isin(set(train["text"])).mean(), 2
        ),
        "label_agreement_on_train_test_overlap": round(agreement, 4),
        "contradictory_label_count": int(round((1 - agreement) * len(shared))),
        "contradictory_examples": examples,
    }


def markdown_table(stats):
    """Stats as a markdown table, ready to paste into the report."""
    rows = [
        ("Rows", "rows"),
        ("Negative", "labels.negative"),
        ("Neutral", "labels.neutral"),
        ("Positive", "labels.positive"),
        ("Neutral %", "label_pct.neutral"),
        ("Duplicate texts", "duplicate_texts"),
        ("Mean tokens", "tokens_mean"),
        ("Median tokens", "tokens_median"),
        ("95th pct tokens", "tokens_p95"),
        ("Vocabulary size", "vocab_size"),
        ("Contains emoji", "with_non_ascii"),
    ]

    def get(entry, dotted):
        value = entry
        for key in dotted.split("."):
            value = value[key]
        return value

    lines = [
        "| | " + " | ".join(entry["split"] for entry in stats) + " |",
        "|---" * (len(stats) + 1) + "|",
    ]
    for title, key in rows:
        cells = " | ".join(str(get(entry, key)) for entry in stats)
        lines.append(f"| {title} | {cells} |")
    return "\n".join(lines)


def write_csv(frame, name):
    out = frame[["text", "label"]].copy()
    out["label_id"] = out["label"].map(LABEL2ID)
    out.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    return len(out)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}
    for name in SPLIT_FILES:
        frame = load_split(name)
        frame["text"] = frame["text"].map(normalise)
        frame["label"] = frame["label"].str.strip().str.lower()

        unknown = set(frame["label"]) - set(LABELS)
        if unknown:
            raise ValueError(f"{name}: unexpected labels {unknown}")

        # Rows that lost all content after normalising carry no signal.
        frames[name] = frame[frame["text"].str.len() > 0].copy()

    train, dev, test = frames["train"], frames["dev"], frames["test"]
    overlap = audit_overlap(train, dev, test)

    # The honest training set: no duplicates, nothing that appears in dev or test.
    held_out = set(dev["text"]) | set(test["text"])
    train_dedup = train.drop_duplicates(subset="text")
    train_dedup = train_dedup[~train_dedup["text"].isin(held_out)].copy()

    # Size-matched control. train_full is 47% larger than train_dedup, so a model
    # trained on it benefits from both the leakage AND the extra rows, and the
    # two cannot be separated. This subsample keeps the leakage but matches
    # train_dedup's row count, so the difference between train_full_sub and
    # train_dedup isolates the leakage effect.
    train_full_sub = train.sample(n=len(train_dedup), random_state=SEED)

    stats = [
        describe(train, "train_full"),
        describe(train_full_sub, "train_full_sub"),
        describe(train_dedup, "train_dedup"),
        describe(dev, "dev"),
        describe(test, "test"),
    ]

    sizes = {
        "train_full": write_csv(train, "train_full"),
        "train_full_sub": write_csv(train_full_sub, "train_full_sub"),
        "train_dedup": write_csv(train_dedup, "train_dedup"),
        "dev": write_csv(dev, "dev"),
        "test": write_csv(test, "test"),
    }

    report = {
        "source": "AfriSenti (SemEval-2023 Task 12), Nigerian Pidgin (pcm)",
        "source_url": "https://github.com/afrisenti-semeval/afrisent-semeval-2023",
        "licence": "CC BY-NC-SA 4.0",
        "label_order": list(LABELS),
        "split_sizes": sizes,
        "per_split": stats,
        "overlap_audit": overlap,
    }
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(report, indent=2))

    summary = "\n".join(
        [
            "# Dataset statistics",
            "",
            markdown_table(stats),
            "",
            "## Overlap audit (on the release as distributed)",
            "",
            f"- Duplicate tweets inside train: {overlap['duplicate_texts_within_train']}",
            f"- Train tweets also in dev: {overlap['train_texts_also_in_dev']}",
            f"- Train tweets also in test: {overlap['train_texts_also_in_test']}",
            f"- Dev tweets also in test: {overlap['dev_texts_also_in_test']}",
            f"- Share of test reachable verbatim from train: "
            f"{overlap['test_rows_reachable_from_train_pct']}%",
            f"- Gold label agreement on the train/test overlap: "
            f"{overlap['label_agreement_on_train_test_overlap']:.1%} "
            f"({overlap['contradictory_label_count']} tweets labelled "
            f"differently in the two splits)",
            "",
        ]
    )
    (RESULTS_DIR / "data_stats.md").write_text(summary, encoding="utf-8")

    print(markdown_table(stats))
    print()
    print("Overlap audit:")
    for key, value in overlap.items():
        if key != "contradictory_examples":
            print(f"  {key}: {value}")
    print("\nContradictory gold labels, first few:")
    for example in overlap["contradictory_examples"][:5]:
        print(
            f"  train={example['label_in_train']:<8} "
            f"test={example['label_in_test']:<8} {example['text'][:70]}"
        )
    print(f"\nWrote {len(sizes)} CSVs to {PROCESSED_DIR}")
    print(f"Wrote audit to {RESULTS_DIR / 'data_stats.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
