"""TF-IDF + logistic regression baseline.

Why this baseline: it is the simplest thing that can work on this task, it needs
no pretraining, and it trains in seconds. If a fine-tuned transformer cannot beat
it, that is worth knowing and reporting rather than hiding.

Two feature choices are compared because Pidgin spelling is not standardised.
The same word shows up as "dey", "de" and "dy", and tweets are full of
lengthened forms like "yaaaaa". Word n-grams treat all of those as separate
features; character n-grams share evidence between them. Testing both answers
whether sub-word features actually help on this data.

Class weighting is the third switch. With neutral at roughly 1.4% of the
training rows, an unweighted model has very little reason to ever predict it.

    python src/baseline.py                      # word features, no weighting
    python src/baseline.py --features char      # character n-grams
    python src/baseline.py --class-weight balanced
    python src/baseline.py --eval-test          # only once, at the end
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import PROCESSED_DIR, RESULTS_DIR, SEED
from evaluation import (
    print_summary,
    record_experiment,
    save_confusion_matrix,
    save_metrics,
    score,
)


def load(split):
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python src/prepare_data.py")
    return pd.read_csv(path)


def build_pipeline(features, class_weight, max_features, C):
    """TF-IDF features feeding a multinomial logistic regression.

    min_df=2 drops terms that appear in only one tweet. On a corpus this small
    those are mostly typos and they add thousands of useless dimensions.
    """
    if features == "word":
        vectoriser = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=max_features,
        )
    elif features == "char":
        vectoriser = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=max_features,
        )
    else:
        raise ValueError(f"unknown features: {features}")

    classifier = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight=class_weight,
        random_state=SEED,
    )
    return Pipeline([("tfidf", vectoriser), ("clf", classifier)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=("word", "char"), default="word")
    parser.add_argument(
        "--train",
        choices=("train_dedup", "train_full", "train_full_sub"),
        default="train_dedup",
        help="train_full keeps the leakage in the original release",
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="'balanced' re-weights the loss by inverse class frequency",
    )
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=1.0, help="inverse regularisation")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="also score the test set. Keep this off while tuning.",
    )
    args = parser.parse_args(argv)

    class_weight = None if args.class_weight == "none" else "balanced"
    weight_tag = "unweighted" if class_weight is None else "balanced"
    name = f"tfidf-{args.features}-{weight_tag}"
    if args.train == "train_full":
        name = f"{name}-leaky"

    train = load(args.train)
    dev = load("dev")

    pipeline = build_pipeline(args.features, class_weight, args.max_features, args.C)
    pipeline.fit(train["text"], train["label_id"])

    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"{name}: trained on {len(train):,} rows, {vocab_size:,} features")

    notes = (
        f"train={args.train}, features={args.features}, "
        f"class_weight={weight_tag}, C={args.C}"
    )

    splits = [("dev", dev)]
    if args.eval_test:
        splits.append(("test", load("test")))

    for split_name, frame in splits:
        predictions = pipeline.predict(frame["text"])
        metrics = score(frame["label_id"], predictions)
        print_summary(name, split_name, metrics)
        save_metrics(name, split_name, metrics)
        record_experiment(name, split_name, metrics, notes)
        save_confusion_matrix(
            frame["label_id"],
            predictions,
            RESULTS_DIR / "figures" / f"cm_{name}_{split_name}.png",
            f"{name} ({split_name})",
        )

    print(f"\nLogged to {RESULTS_DIR / 'experiments.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
