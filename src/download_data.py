"""Download the AfriSenti Nigerian Pidgin (pcm) sentiment splits.

The files come from the official SemEval-2023 Task 12 repository:
    https://github.com/afrisenti-semeval/afrisent-semeval-2023

I fetch the TSVs from GitHub rather than the Hugging Face mirror
(HausaNLP/AfriSenti-Twitter) because that mirror ships a dataset loading script
and no parquet files, and datasets >= 4.0 refuses to run remote loading scripts.
Same data, fewer moving parts.

AfriSenti is released under CC BY-NC-SA 4.0. It is used here for non-commercial
coursework and cited in the report. The raw files are not committed to this
repository, so run this script before anything else.

    python src/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request

from config import RAW_DIR

BASE_URL = (
    "https://raw.githubusercontent.com/afrisenti-semeval/"
    "afrisent-semeval-2023/main/data/pcm"
)
FILES = ("train.tsv", "dev.tsv", "test.tsv")


def download(filename):
    dest = RAW_DIR / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {filename}: already here ({dest.stat().st_size:,} bytes)")
        return

    url = f"{BASE_URL}/{filename}"
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    dest.write_bytes(payload)
    print(f"  {filename}: saved {len(payload):,} bytes")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AfriSenti pcm into {RAW_DIR}")
    for filename in FILES:
        download(filename)
    return 0


if __name__ == "__main__":
    sys.exit(main())
