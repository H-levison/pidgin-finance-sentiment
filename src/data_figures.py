"""Figures for the dataset section of the report.

One figure, because there is one thing the dataset section has to show: the
neutral class is almost absent from training but is a tenth of the test set.
Everything else about the data is better as a table (results/data_stats.md) than
as a chart.

Design choices worth defending if asked:

Sentiment is a **polarity** scale, not a set of unrelated categories, so the
colours use a diverging pair - red and blue poles with a grey midpoint for
neutral - rather than three arbitrary hues. Red/blue was picked over red/green
because red/green is the single worst choice for colour-blind readers.

The palette was checked with a validator rather than by eye:
  negative #e34948, neutral #52514e, positive #2a78d6
passes the lightness band, adjacent-pair colour-blind separation (worst dE 10.4,
protanopia), the normal-vision floor (21.8) and 3:1 contrast against the
surface. The one check it fails is the chroma floor on #52514e, and that is
deliberate: a diverging midpoint is supposed to read as grey. A categorical
palette failing that check would be a bug; here it is the requirement.

Bars are also directly labelled, so the figure is readable in greyscale and
identity never rests on colour alone.

    python src/data_figures.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import LABELS, PROCESSED_DIR, RESULTS_DIR

# Diverging pair with a grey midpoint; see the module docstring.
COLOURS = {"negative": "#e34948", "neutral": "#52514e", "positive": "#2a78d6"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SPLITS = [
    ("train_dedup", "train\n(deduplicated)"),
    ("dev", "dev"),
    ("test", "test"),
]


def load_shares():
    rows = []
    for name, display in SPLITS:
        path = PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"{path} is missing. Run: python src/prepare_data.py")
        frame = pd.read_csv(path)
        counts = frame["label"].value_counts()
        for label in LABELS:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "split": display,
                    "label": label,
                    "count": count,
                    "share": 100 * count / len(frame),
                }
            )
    return pd.DataFrame(rows)


def main():
    data = load_shares()

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    displays = [display for _, display in SPLITS]
    n_series = len(LABELS)
    group_width = 0.76
    bar_width = group_width / n_series

    for series_index, label in enumerate(LABELS):
        subset = data[data["label"] == label].set_index("split").loc[displays]
        # 2px-equivalent gap between adjacent bars so fills never touch.
        offsets = [
            i - group_width / 2 + bar_width * (series_index + 0.5)
            for i in range(len(displays))
        ]
        bars = ax.bar(
            offsets,
            subset["share"],
            width=bar_width * 0.88,
            label=label,
            color=COLOURS[label],
            zorder=3,
        )
        # Direct labels: keeps the figure readable in greyscale and in print.
        for bar, share, count in zip(bars, subset["share"], subset["count"]):
            ax.annotate(
                f"{share:.1f}%\nn={count:,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK,
            )

    ax.set_xticks(range(len(displays)), displays, color=INK, fontsize=9)
    ax.set_ylabel("share of split (%)", color=INK, fontsize=9)
    # Headroom so the legend sits in clear space above the tallest bar
    # and its direct label, rather than colliding with the title.
    ax.set_ylim(0, 88)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.tick_params(axis="x", length=0)

    # Recessive chrome: horizontal hairlines only, behind the data.
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.set_title(
        "Neutral is 1.5% of training data but 10.4% of the test set",
        color=INK,
        fontsize=11,
        loc="left",
        pad=10,
    )

    legend = ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.0),
    )
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)

    print(data.pivot(index="split", columns="label", values="share").round(1))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
