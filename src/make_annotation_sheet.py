"""Turn collected forum posts into a sheet ready for labelling.

Collect posts into a plain text file, one post per line, at
data/annotation/collected.txt. Then run this. It normalises the text the same
way the AfriSenti data was normalised, drops duplicates and posts that are too
short, shuffles the order, and writes a CSV with empty label and notes columns.

Shuffling matters. If the sheet stays in the order the posts were collected,
every post from one angry thread sits together and the labels drift towards
whatever mood that thread was in.

The sheet also carries a `double` column marking a random 50 posts for the
reliability check described in data/annotation/GUIDELINES.md.

    python src/make_annotation_sheet.py
    python src/make_annotation_sheet.py --double 50 --min-words 4
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

from config import REPO_ROOT, SEED

ANNOTATION_DIR = REPO_ROOT / "data" / "annotation"
COLLECTED = ANNOTATION_DIR / "collected.txt"
SHEET = ANNOTATION_DIR / "to_label.csv"

WHITESPACE = re.compile(r"\s+")
URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
# U+FFFD is what decoding the forum HTML with errors="replace" leaves behind
# where a byte could not be decoded. It carries no meaning and would show up as
# a black diamond in the labelling interface.
BROKEN_CHAR = re.compile("�+")

# Rough Pidgin markers, used only to report how much of the collection looks
# like Pidgin. Nothing is dropped on this basis - that is the annotator's call
# under the guidelines.
PIDGIN_MARKERS = (
    "dey", "na ", "wetin", "abeg", "don ", "wahala", "oga", "sabi", "shey",
    "comot", "waka", "chop", "yarn", "gist", "jare", "abi", "nawa", "pikin",
    "sef", "ehn", "biko", "wey", "dem ", "e be", "no be", "make you",
)


def normalise(text):
    """Match the AfriSenti preprocessing so the two test sets are comparable.

    AfriSenti is lowercased, has URLs removed and mentions replaced with @user.
    Applying the same steps here means a score difference between the two test
    sets reflects the domain, not the preprocessing.
    """
    text = BROKEN_CHAR.sub(" ", str(text))
    text = URL.sub("", text)
    text = MENTION.sub("@user", text)
    text = text.lower()
    return WHITESPACE.sub(" ", text).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument(
        "--double",
        type=int,
        default=50,
        help="how many posts to mark for the second-annotator check",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    if not COLLECTED.exists():
        COLLECTED.write_text(
            "# One post per line. Lines starting with # are ignored.\n"
            "# Paste posts from the Bitcointalk Nigeria board and Nairaland\n"
            "# crypto threads here, then run src/make_annotation_sheet.py\n",
            encoding="utf-8",
        )
        print(f"Created {COLLECTED}. Paste posts into it, one per line, then rerun.")
        return 0

    lines = [
        line.strip()
        for line in COLLECTED.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        print(f"{COLLECTED} has no posts in it yet.")
        return 1

    frame = pd.DataFrame({"text": [normalise(line) for line in lines]})
    collected_count = len(frame)

    frame = frame[frame["text"].str.split().str.len() >= args.min_words]
    too_short = collected_count - len(frame)

    frame = frame.drop_duplicates(subset="text")
    duplicates = collected_count - too_short - len(frame)

    frame = frame.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    looks_pidgin = frame["text"].apply(
        lambda text: any(marker in f" {text} " for marker in PIDGIN_MARKERS)
    )

    frame.insert(0, "id", range(1, len(frame) + 1))
    frame["label"] = ""
    frame["notes"] = ""
    frame["source"] = ""
    frame["double"] = ""
    frame.loc[frame.index[: args.double], "double"] = "yes"

    if SHEET.exists():
        backup = SHEET.with_suffix(".csv.bak")
        SHEET.replace(backup)
        print(f"Existing sheet moved to {backup.name} rather than overwritten.")

    frame.to_csv(SHEET, index=False, encoding="utf-8")

    print(f"Collected posts read:      {collected_count}")
    print(f"  dropped, under {args.min_words} words: {too_short}")
    print(f"  dropped as duplicates:   {duplicates}")
    print(f"Posts in the sheet:        {len(frame)}")
    print(f"  with Pidgin markers:     {int(looks_pidgin.sum())} "
          f"({100 * looks_pidgin.mean():.0f}%)")
    print(f"  marked for double check: {min(args.double, len(frame))}")
    print(f"\nSheet written to {SHEET}")
    print("Fill in the label column: positive, negative, neutral, or exclude.")
    print("Rules are in data/annotation/GUIDELINES.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
