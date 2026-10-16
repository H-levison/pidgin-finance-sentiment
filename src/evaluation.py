"""Scoring, plots and the experiment log, shared by every model script.

Every model in this project reports the same numbers through these functions so
the rows in results/experiments.csv are actually comparable. Doing this once
also means the label order can only come from config.LABELS.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import LABELS, RESULTS_DIR

EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"


def score(y_true, y_pred):
    """Accuracy plus macro and weighted F1, and the per-class breakdown.

    Both F1 averages are reported on purpose. Weighted F1 is the metric the
    SemEval-2023 Task 12 leaderboard used, so it is what published numbers can
    be compared against. Macro F1 treats the three classes equally, which is
    what exposes the neutral class collapsing - with only ~1.4% of training rows
    labelled neutral, weighted F1 can look healthy while neutral is never
    predicted at all.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 4
        ),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
    }


def save_confusion_matrix(y_true, y_pred, path, title):
    """Counts, with the row-normalised share annotated underneath each cell."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(
        matrix,
        row_totals,
        out=np.zeros(matrix.shape, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            colour = "white" if shares[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{shares[i, j]:.0%}",
                ha="center",
                va="center",
                color=colour,
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, label="share of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def record_experiment(name, split, metrics, notes=""):
    """Append one row to results/experiments.csv, the table used in the report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": name,
        "split": split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "f1_negative": metrics["per_class"]["negative"]["f1"],
        "f1_neutral": metrics["per_class"]["neutral"]["f1"],
        "f1_positive": metrics["per_class"]["positive"]["f1"],
        "notes": notes,
    }
    write_header = not EXPERIMENTS_CSV.exists()
    with EXPERIMENTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def save_metrics(name, split, metrics):
    path = RESULTS_DIR / f"metrics_{name}_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def print_summary(name, split, metrics):
    print(f"\n{name}  [{split}]")
    print(f"  accuracy     {metrics['accuracy']:.4f}")
    print(f"  macro F1     {metrics['macro_f1']:.4f}")
    print(f"  weighted F1  {metrics['weighted_f1']:.4f}")
    for label in LABELS:
        entry = metrics["per_class"][label]
        print(
            f"  {label:<9} P {entry['precision']:.3f}  R {entry['recall']:.3f}  "
            f"F1 {entry['f1']:.3f}  (n={entry['support']})"
        )
