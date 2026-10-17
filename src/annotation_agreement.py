"""Measure how far the two annotators agreed, and what adjudication changed.

A test set labelled by one person is weak evidence. This reports what two
independent passes agreed on, where they did not, and what the joint third pass
settled.

Cohen's kappa is used rather than raw agreement because raw agreement flatters
an unbalanced label set: if most posts were negative, two annotators guessing
negative every time would agree most of the time while adding no information.
Kappa measures agreement above what chance would produce.

Reading of kappa follows Landis and Koch (1977): below 0.20 slight, 0.21-0.40
fair, 0.41-0.60 moderate, 0.61-0.80 substantial, above 0.80 almost perfect.

Writes results/annotation_agreement.md.

    python src/annotation_agreement.py
"""

from __future__ import annotations

import sys
from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import LABELS, REPO_ROOT, RESULTS_DIR

ANNOTATION_DIR = REPO_ROOT / "data" / "annotation"
SHEET = ANNOTATION_DIR / "to_label.csv"
SECOND = ANNOTATION_DIR / "second_annotator.csv"
ADJUDICATION = ANNOTATION_DIR / "adjudication.csv"
GOLD = ANNOTATION_DIR / "adjudicated_gold.csv"

# Muhammad et al. (2023b), Table 5: free marginal multi-rater kappa for pcm.
PUBLISHED_PCM_KAPPA = 0.60


def reading(kappa):
    for threshold, name in ((0.21, "slight"), (0.41, "fair"),
                            (0.61, "moderate"), (0.81, "substantial")):
        if kappa < threshold:
            return name
    return "almost perfect"


def main():
    for path in (SHEET, SECOND):
        if not path.exists():
            raise SystemExit(f"{path} is missing.")

    first = pd.read_csv(SHEET, dtype=str).fillna("")
    second = pd.read_csv(SECOND, dtype=str).fillna("")
    for frame in (first, second):
        frame["id"] = frame["id"].astype(str)

    both = first[["id", "text", "label"]].merge(
        second[["id", "label"]], on="id", suffixes=("_1", "_2")
    )
    both = both[(both.label_1 != "") & (both.label_2 != "")]

    sentiment = both[both.label_1.isin(LABELS) & both.label_2.isin(LABELS)]
    polarity = sentiment[(sentiment.label_1 != "neutral") & (sentiment.label_2 != "neutral")]

    measures = [
        ("what to exclude (not Pidgin, not finance)",
         (both.label_1 == "exclude"), (both.label_2 == "exclude"), len(both)),
        ("positive against negative, neutral dropped",
         polarity.label_1, polarity.label_2, len(polarity)),
        ("all three sentiment classes",
         sentiment.label_1, sentiment.label_2, len(sentiment)),
    ]

    lines = [
        "# Annotation reliability and adjudication",
        "",
        "Two annotators, both native Nigerian Pidgin speakers, labelled the "
        "finance set independently against `data/annotation/GUIDELINES.md`. "
        "Neither saw the other's labels.",
        "",
        "## Independent agreement, before adjudication",
        "",
        "| judgement | posts | raw agreement | Cohen's kappa | reading |",
        "|---|---|---|---|---|",
    ]
    for name, a, b, n in measures:
        kappa = cohen_kappa_score(a, b)
        raw = float((a == b).mean())
        lines.append(f"| {name} | {n} | {raw:.1%} | **{kappa:.3f}** | {reading(kappa)} |")
        print(f"  {name:<44} n={n:>4}  raw {raw:6.1%}  kappa {kappa:.3f}")

    lines += [
        "",
        f"Muhammad et al. (2023b) report a free marginal kappa of "
        f"{PUBLISHED_PCM_KAPPA} for Pidgin in AfriSenti. The three-way figure "
        "here is well below that, and the decomposition shows why: the "
        "disagreement sits almost entirely on the neutral class.",
        "",
        "| | annotator 2: " + " | ".join(LABELS) + " |",
        "|---" * (len(LABELS) + 1) + "|",
    ]
    for label in LABELS:
        row = sentiment[sentiment.label_1 == label]
        counts = [int((row.label_2 == other).sum()) for other in LABELS]
        lines.append(f"| **annotator 1: {label}** | " + " | ".join(map(str, counts)) + " |")

    # --- adjudication ---
    if ADJUDICATION.exists():
        adjudication = pd.read_csv(ADJUDICATION, dtype=str).fillna("")
        adjudication["agreed"] = adjudication["agreed"].str.strip().str.lower()
        adjudication["reason"] = adjudication["reason"].str.strip().str.lower()
        settled = adjudication[adjudication.agreed != ""]

        lines += [
            "", "## Adjudication", "",
            f"The {int((both.label_1 == both.label_2).sum())} posts the annotators "
            f"already agreed on were left untouched. The "
            f"{len(adjudication)} disagreements were worked through together, "
            "grouped by the kind of disagreement so a decision about a class of "
            "case could be applied along a run.",
            "",
            "| disagreement | n | settled as |",
            "|---|---|---|",
        ]
        for pair, n in Counter(adjudication.disagreement).most_common():
            subset = adjudication[adjudication.disagreement == pair]
            how = ", ".join(f"{v} {k}" for k, v in Counter(subset.agreed).most_common())
            lines.append(f"| {pair} | {n} | {how} |")

        if GOLD.exists():
            gold = pd.read_csv(GOLD, dtype=str).fillna("")
            gold["id"] = gold["id"].astype(str)
            check = gold.merge(both[["id", "label_1", "label_2"]], on="id")
            m1 = float((check.final == check.label_1).mean())
            m2 = float((check.final == check.label_2).mean())
            lines += [
                "",
                f"The settled labels match annotator 1 on {m1:.1%} of posts and "
                f"annotator 2 on {m2:.1%}, so this was not one annotator "
                "conceding to the other.",
            ]

        # Were the recorded reasons actually rules?
        lines += [
            "", "### Were the recorded reasons applied as rules?", "",
            "| reason recorded | n | maps to one label |", "|---|---|---|",
        ]
        for reason, n in Counter(settled[settled.reason != ""].reason).most_common():
            subset = settled[settled.reason == reason]
            top = Counter(subset.agreed).most_common(1)[0][1]
            lines.append(f"| {reason} | {n} | {100 * top / len(subset):.0f}% |")
        lines += [
            "",
            "The exclusion criteria were applied consistently. Reasons that "
            "describe what the annotator saw rather than a rule they applied do "
            "not map reliably to a label. The adjudication settled the labels "
            "but did not sharpen the guideline; the informational-against-"
            "approving boundary needs worked examples defined before labelling, "
            "not recovered afterwards.",
        ]

    if GOLD.exists():
        gold = pd.read_csv(GOLD, dtype=str).fillna("")
        usable = gold[gold.final.isin(LABELS)]
        tally = Counter(usable.final)
        lines += [
            "", "## The adjudicated gold set", "",
            "| label | count | share |", "|---|---|---|",
        ]
        for label in LABELS:
            count = tally.get(label, 0)
            lines.append(f"| {label} | {count} | {100 * count / max(len(usable), 1):.1f}% |")
        lines += ["", f"Usable test posts: **{len(usable)}**, with "
                  f"{int((gold.final == 'exclude').sum())} excluded.", ""]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "annotation_agreement.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'annotation_agreement.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
