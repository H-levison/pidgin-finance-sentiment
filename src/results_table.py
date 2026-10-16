"""Turn the experiment log into the tables that go in the report.

experiments.csv is append-only, so re-running a configuration leaves both rows
in it. This keeps the most recent row per (experiment, split) pair and writes
two markdown tables: one for dev, one for test.

Published pcm numbers from Muhammad et al. (2023b) Table 7 are printed
alongside, but only the accuracy column, because that table reports accuracy
rather than weighted F1 and comparing across metrics would be meaningless.

    python src/results_table.py
    python src/results_table.py --split test
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import RESULTS_DIR

EXPERIMENTS = RESULTS_DIR / "experiments.csv"

# Muhammad et al. (2023b), AfriSenti EMNLP, Table 7, page 8.
# Accuracy on the pcm test split, averaged over 5 runs.
PUBLISHED_PCM_ACCURACY = {
    "AfriBERTa-large": 62.3,
    "XLM-R-base": 63.8,
    "AfroXLMR-base": 67.6,
    "mDeBERTaV3-base": 66.2,
    "XLM-T-base": 66.6,
    "XLM-R-large": 67.2,
    "AfroXLMR-large": 68.7,
}

# Order the rows so the table reads as an argument rather than a dump.
ORDER = [
    "tfidf-word-unweighted",
    "tfidf-word-balanced",
    "tfidf-char-balanced",
    "bilstm-random-weighted",
    "bilstm-w2v-unweighted",
    "bilstm-w2v-weighted",
    "afriberta_base-weighted",
    "roberta-base-weighted",
    "xlm-roberta-base-weighted",
    "afro-xlmr-base-unweighted",
    "afro-xlmr-base-weighted",
    "tfidf-word-balanced-leaky",
    "afro-xlmr-base-weighted-leaky",
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default=None,
        help="dev, test, finance_test; all of them if omitted",
    )
    args = parser.parse_args(argv)

    if not EXPERIMENTS.exists():
        raise SystemExit(f"{EXPERIMENTS} does not exist yet.")

    frame = pd.read_csv(EXPERIMENTS)
    # Keep only the latest run of each configuration.
    frame = frame.sort_values("run_at").drop_duplicates(
        subset=["experiment", "split"], keep="last"
    )

    rank = {name: i for i, name in enumerate(ORDER)}
    frame["order"] = frame["experiment"].map(lambda n: rank.get(n, len(ORDER)))
    frame = frame.sort_values(["order", "experiment"])

    splits = [args.split] if args.split else ["dev", "test", "finance_test"]
    sections = []

    for split in splits:
        subset = frame[frame["split"] == split]
        if subset.empty:
            continue
        lines = [
            f"## {split} set",
            "",
            "| model | accuracy | macro F1 | weighted F1 | F1 neg | F1 neu | F1 pos |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in subset.iterrows():
            leaky = "leaky" in str(row["experiment"])
            label = str(row["experiment"])
            if leaky:
                label += " *"
            lines.append(
                f"| {label} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | "
                f"{row['weighted_f1']:.3f} | {row['f1_negative']:.3f} | "
                f"{row['f1_neutral']:.3f} | {row['f1_positive']:.3f} |"
            )
        if subset["experiment"].str.contains("leaky").any():
            lines += [
                "",
                "\\* trained on the release as distributed, with the 1,013 tweets "
                "that also appear in the test split left in. Not a fair "
                "comparison with the rows above it, which is the point.",
            ]
        sections.append("\n".join(lines))

    published = [
        "## Published pcm results for context",
        "",
        "Muhammad et al. (2023b), Table 7. **Accuracy** on the pcm test split, "
        "averaged over 5 runs. Compare these only against our accuracy column, "
        "and only against our test rows.",
        "",
        "| model | accuracy |",
        "|---|---|",
    ]
    for name, accuracy in PUBLISHED_PCM_ACCURACY.items():
        published.append(f"| {name} | {accuracy / 100:.3f} |")

    document = "\n\n".join(["# Results", *sections, "\n".join(published)]) + "\n"
    out_path = RESULTS_DIR / "results_table.md"
    out_path.write_text(document, encoding="utf-8")
    print(document)
    print(f"Written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
