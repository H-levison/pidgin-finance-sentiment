"""Evaluate a trained checkpoint on the finance test set, next to AfriSenti test.

This answers the project's second research question: does a model trained on
general Pidgin tweets still work when people are talking about money?

Both test sets are scored in the same run so the comparison is apples to apples
- same checkpoint, same preprocessing, same metrics. Reporting a finance score
on its own would say nothing, because there would be no way to tell a domain
effect from the model simply being mediocre.

Two caveats that belong in the report, not buried here:

1. The two test sets have different class balances, so weighted F1 is not
   directly comparable between them. Macro F1 and the per-class breakdown are.
2. The finance set is small and single-annotator, so a few points of difference
   is noise. Only a large gap means anything.

    python src/evaluate_finance.py --checkpoint afro-xlmr-base-weighted
    python src/evaluate_finance.py --checkpoint afro-xlmr-base-weighted --skip-afrisenti
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ID2LABEL, LABELS, MODEL_DIR, PROCESSED_DIR, RESULTS_DIR
from evaluation import (
    print_summary,
    record_experiment,
    save_confusion_matrix,
    save_metrics,
    score,
)
from finetune import TweetDataset, make_collate, predict


def load_split(name):
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        if name == "finance_test":
            raise SystemExit(
                f"{path} is missing.\n\n"
                "Build it first:\n"
                "  1. paste collected posts into data/annotation/collected.txt\n"
                "  2. python src/make_annotation_sheet.py\n"
                "  3. fill in the label column by hand, following "
                "data/annotation/GUIDELINES.md\n"
                "  4. python src/build_finance_test.py"
            )
        raise SystemExit(f"{path} is missing. Run: python src/prepare_data.py")
    return pd.read_csv(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="afro-xlmr-base-weighted",
        help="directory name under PIDGIN_MODEL_DIR, or a full path / Hub id",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--skip-afrisenti",
        action="store_true",
        help="score only the finance set (faster, but loses the comparison)",
    )
    args = parser.parse_args(argv)

    path = MODEL_DIR / args.checkpoint
    model_source = str(path) if path.exists() else args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source).to(device)
    model.eval()

    # Guard against a checkpoint whose label order differs from config.LABELS.
    # Silently mismatched labels would make every per-class number wrong.
    checkpoint_labels = [
        model.config.id2label[i] for i in range(len(model.config.id2label))
    ]
    if [label.lower() for label in checkpoint_labels] != list(LABELS):
        raise SystemExit(
            f"Checkpoint label order {checkpoint_labels} does not match "
            f"config.LABELS {list(LABELS)}. Refusing to score against the wrong "
            "label mapping."
        )

    print(f"checkpoint: {model_source}")
    print(f"device: {device}")

    collate = make_collate(tokenizer)

    finance = load_split("finance_test")
    finance_words = finance["text"].str.split().str.len()
    low, high = int(finance_words.min()), int(finance_words.max())

    targets = []
    if not args.skip_afrisenti:
        targets.append(("test", "test", None))
        # Length-matched control. The finance posts come from a forum and the
        # AfriSenti posts from Twitter, so the finance set is longer on average.
        # Without this control a drop on the finance set could be explained by
        # length alone, and the domain claim would not survive scrutiny. This
        # restricts AfriSenti test to the same word-count range, so the
        # remaining difference is attributable to domain and platform rather
        # than to how much text the model was given.
        targets.append(("test", "test_lengthmatched", (low, high)))
    targets.append(("finance_test", "finance_test", None))

    print(f"finance set word counts: {low}-{high} "
          f"(mean {finance_words.mean():.1f})")

    summary = {}
    for file_name, label, word_range in targets:
        frame = load_split(file_name)
        if word_range is not None:
            lengths = frame["text"].str.split().str.len()
            frame = frame[(lengths >= word_range[0]) & (lengths <= word_range[1])].copy()
            print(f"  {label}: {len(frame)} of the AfriSenti test posts fall in "
                  f"{word_range[0]}-{word_range[1]} words")
            if len(frame) < 100:
                print("  WARNING: small matched subset, treat this comparison "
                      "as indicative only")
        loader = DataLoader(
            TweetDataset(frame, tokenizer, args.max_length),
            batch_size=args.batch_size,
            collate_fn=collate,
        )
        predictions, truth = predict(model, loader, device)
        metrics = score(truth, predictions)
        summary[label] = metrics

        print_summary(args.checkpoint, label, metrics)
        save_metrics(args.checkpoint, label, metrics)
        record_experiment(
            args.checkpoint,
            label,
            metrics,
            f"eval only, n={len(frame)}, checkpoint={args.checkpoint}",
        )
        save_confusion_matrix(
            truth,
            predictions,
            RESULTS_DIR / "figures" / f"cm_{args.checkpoint}_{label}.png",
            f"{args.checkpoint} ({label})",
        )
        frame.assign(predicted=[ID2LABEL[int(p)] for p in predictions]).to_csv(
            RESULTS_DIR / f"predictions_{args.checkpoint}_{label}.csv", index=False
        )

    if "test" in summary and "finance_test" in summary:
        general = summary["test"]
        matched = summary.get("test_lengthmatched")
        finance = summary["finance_test"]

        lines = [
            "# Domain transfer: general Pidgin tweets to finance and crypto posts",
            "",
            f"Checkpoint: `{args.checkpoint}`",
            "",
            "The middle column is the control. The finance posts come from a "
            "forum and are longer than AfriSenti's tweets, so the comparison "
            "that matters is **finance against length-matched**, not finance "
            "against the full test set. Comparing against the full set would "
            "confound topic with post length.",
            "",
            "| metric | AfriSenti test (all) | AfriSenti length-matched | "
            "finance test | domain effect |",
            "|---|---|---|---|---|",
        ]
        for key, title in (
            ("accuracy", "accuracy"),
            ("macro_f1", "macro F1"),
            ("weighted_f1", "weighted F1"),
        ):
            if matched:
                effect = finance[key] - matched[key]
                lines.append(
                    f"| {title} | {general[key]:.3f} | {matched[key]:.3f} | "
                    f"{finance[key]:.3f} | {effect:+.3f} |"
                )
            else:
                lines.append(
                    f"| {title} | {general[key]:.3f} | n/a | {finance[key]:.3f} | "
                    f"{finance[key] - general[key]:+.3f} |"
                )

        lines += ["", "Per class F1:", "",
                  "| class | AfriSenti (all) | length-matched | finance |",
                  "|---|---|---|---|"]
        for label in LABELS:
            matched_cell = (
                f"{matched['per_class'][label]['f1']:.3f}" if matched else "n/a"
            )
            lines.append(
                f"| {label} | {general['per_class'][label]['f1']:.3f} | "
                f"{matched_cell} | {finance['per_class'][label]['f1']:.3f} |"
            )

        lines += [
            "",
            "Three caveats that belong in the report rather than buried here:",
            "",
            "1. The sets have different class balances, so **weighted F1 is not "
            "comparable across columns**. Macro F1 and the per-class rows are.",
            "2. The finance set is small and labelled by one annotator, so a "
            "few points of difference is noise, not a finding.",
            "3. Length is controlled; platform is not. The finance posts are "
            "forum replies and the AfriSenti posts are tweets, so part of any "
            "remaining gap may be register rather than subject matter. Say so.",
            "",
        ]
        out = RESULTS_DIR / "domain_transfer.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print("\n" + "\n".join(lines))
        print(f"Written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
