"""Paths, label order and seed shared by every stage of the pipeline.

Keeping the label order in one place matters: if data prep and evaluation
disagreed on which index means "neutral", the per-class scores would be wrong
without anything visibly breaking.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# Checkpoints are big, so allow them to live outside the synced project folder.
MODEL_DIR = Path(os.environ.get("PIDGIN_MODEL_DIR", REPO_ROOT / "models"))

LABELS = ("negative", "neutral", "positive")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

SEED = 42
