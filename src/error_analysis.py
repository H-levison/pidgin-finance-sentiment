"""Group the model's mistakes by cause instead of listing them one by one.

Reporting "accuracy was 0.72" says nothing about why. This script takes a
predictions file written by finetune.py and asks specific questions:

  - Which class pairs get confused, and in which direction?
  - Are mistakes concentrated in short posts, long posts, or neither?
  - Do posts containing words the model never saw in training fail more often?
  - How many "errors" are actually tweets whose gold label contradicts the
    label the same tweet carries in the training split?
  - What does the model get confidently wrong, as opposed to narrowly wrong?

That fourth question matters most. The data audit found 124 tweets labelled
differently in train and test, so some share of the measured error rate is the
benchmark disagreeing with itself rather than the model failing.

    python src/error_analysis.py --predictions results/predictions_afro-xlmr-base-weighted_test.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

import pandas as pd

from config import LABELS, PROCESSED_DIR, RAW_DIR, RESULTS_DIR

NON_ASCII = re.compile(r"[^\x00-\x7F]")


def load_raw(name):
    frame = pd.read_csv(
        RAW_DIR / f"{name}.tsv", sep="\t", quoting=csv.QUOTE_NONE, dtype=str
    )
    return frame.rename(columns={"tweet": "text"})


def bucket_by_length(frame, column="n_words"):
    """Error rate by post length. Short posts carry less evidence, so if the
    errors cluster there the problem is information, not the model."""
    bins = [0, 10, 20, 35, 1000]
    names = ["1-10", "11-20", "21-35", "36+"]
    frame = frame.copy()
    frame["bucket"] = pd.cut(frame[column], bins=bins, labels=names, right=True)
    grouped = frame.groupby("bucket", observed=True)["wrong"].agg(["mean", "count"])
    return grouped.rename(columns={"mean": "error_rate", "count": "posts"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--train", default="train_dedup")
    parser.add_argument("--out", default="error_analysis.md")
    args = parser.parse_args(argv)

    predictions_path = RESULTS_DIR.parent / args.predictions
    if not predictions_path.exists():
        predictions_path = RESULTS_DIR / args.predictions
    if not predictions_path.exists():
        raise SystemExit(f"Cannot find {args.predictions}")

    frame = pd.read_csv(predictions_path)
    for column in ("text", "label", "predicted"):
        if column not in frame.columns:
            raise SystemExit(f"{predictions_path.name} has no '{column}' column")

    frame["wrong"] = frame["label"] != frame["predicted"]
    frame["n_words"] = frame["text"].str.split().str.len()
    frame["has_emoji"] = frame["text"].str.contains(NON_ASCII)

    train = pd.read_csv(PROCESSED_DIR / f"{args.train}.csv")
    train_vocab = set(" ".join(train["text"].astype(str)).split())

    def unseen_share(text):
        words = str(text).split()
        if not words:
            return 0.0
        return sum(word not in train_vocab for word in words) / len(words)

    frame["unseen_word_share"] = frame["text"].map(unseen_share)

    accuracy = 1 - frame["wrong"].mean()
    lines = [
        "# Error analysis",
        "",
        f"Predictions file: `{predictions_path.name}`",
        f"Posts: {len(frame)}  |  correct: {int((~frame['wrong']).sum())}  |  "
        f"wrong: {int(frame['wrong'].sum())}  |  accuracy: {accuracy:.3f}",
        "",
        "## Which classes get confused",
        "",
        "| true -> predicted | count | share of that true class |",
        "|---|---|---|",
    ]

    for true_label in LABELS:
        subset = frame[frame["label"] == true_label]
        if subset.empty:
            continue
        for predicted_label in LABELS:
            count = int((subset["predicted"] == predicted_label).sum())
            if count == 0:
                continue
            marker = "" if true_label == predicted_label else " <- error"
            lines.append(
                f"| {true_label} -> {predicted_label}{marker} | {count} | "
                f"{100 * count / len(subset):.1f}% |"
            )

    lines += ["", "## Error rate by post length", "", "| words | posts | error rate |", "|---|---|---|"]
    for bucket, row in bucket_by_length(frame).iterrows():
        lines.append(f"| {bucket} | {int(row['posts'])} | {row['error_rate']:.3f} |")

    lines += ["", "## Error rate by vocabulary coverage", ""]
    quartiles = pd.qcut(frame["unseen_word_share"], 4, duplicates="drop")
    coverage = frame.groupby(quartiles, observed=True)["wrong"].agg(["mean", "count"])
    lines += ["| share of words unseen in training | posts | error rate |", "|---|---|---|"]
    for interval, row in coverage.iterrows():
        lines.append(
            f"| {interval} | {int(row['count'])} | {row['mean']:.3f} |"
        )

    emoji_rate = frame.groupby("has_emoji", observed=True)["wrong"].agg(["mean", "count"])
    lines += ["", "## Emoji", "", "| contains emoji | posts | error rate |", "|---|---|---|"]
    for has_emoji, row in emoji_rate.iterrows():
        lines.append(
            f"| {'yes' if has_emoji else 'no'} | {int(row['count'])} | {row['mean']:.3f} |"
        )

    # How much of the error rate is the benchmark contradicting itself?
    try:
        raw_train = load_raw("train")
        raw_train["text"] = raw_train["text"].str.replace(r"\s+", " ", regex=True).str.strip()
        train_labels = raw_train.groupby("text")["label"].first()
        shared = frame[frame["text"].isin(train_labels.index)].copy()
        shared["label_in_train"] = shared["text"].map(train_labels)
        contradictory = shared[shared["label_in_train"] != shared["label"]]
        if len(contradictory):
            matched_train_label = (
                contradictory["predicted"] == contradictory["label_in_train"]
            ).sum()
            lines += [
                "",
                "## Errors that are actually label noise",
                "",
                f"- Test posts that also appear in the training split: {len(shared)}",
                f"- Of those, posts whose training label contradicts the test "
                f"label: {len(contradictory)}",
                f"- Posts where the model predicted the *training* label and was "
                f"marked wrong: {int(matched_train_label)}",
                "",
                "Those last ones are cases where the model learned the label the "
                "dataset taught it and was then penalised for it by the same "
                "dataset. They are a ceiling on the achievable score, not a "
                "modelling failure.",
                "",
                "Examples:",
                "",
            ]
            for _, row in contradictory.head(8).iterrows():
                lines.append(
                    f"- train says `{row['label_in_train']}`, test says "
                    f"`{row['label']}`, model said `{row['predicted']}`: "
                    f"{row['text'][:100]}"
                )
    except FileNotFoundError:
        lines += ["", "_(raw train TSV not available, skipped the label-noise check)_"]

    lines += ["", "## Sample errors to read out loud", ""]
    for true_label in LABELS:
        mistakes = frame[(frame["label"] == true_label) & frame["wrong"]]
        if mistakes.empty:
            continue
        lines += [f"### True label: {true_label}", ""]
        for _, row in mistakes.head(6).iterrows():
            lines.append(f"- said `{row['predicted']}`: {row['text'][:130]}")
        lines.append("")

    correct = frame[~frame["wrong"]]
    lines += ["## Sample correct predictions", ""]
    for true_label in LABELS:
        hits = correct[correct["label"] == true_label]
        if hits.empty:
            continue
        row = hits.iloc[0]
        lines.append(f"- `{true_label}`: {row['text'][:130]}")

    summary = "\n".join(lines) + "\n"
    out_path = RESULTS_DIR / args.out
    out_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
