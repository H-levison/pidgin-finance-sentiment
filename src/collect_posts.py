"""Collect candidate Pidgin finance posts from the Bitcointalk Nigeria board.

This replaces copying posts by hand. It is in the repository so the collection
step is reproducible and auditable like every other stage, rather than being an
undocumented pile of text.

What it does NOT do is decide anything about sentiment. It gathers candidates
and filters out the obvious noise; every surviving post is still read and
labelled by a human under data/annotation/GUIDELINES.md. The filters here are
deliberately crude, because a filter that was clever about sentiment would be
biasing the test set towards whatever the filter thinks sentiment looks like.

Being a good citizen about this:

  - bitcointalk.org/robots.txt contains no Disallow rules (checked 16 Oct 2026),
    so fetching these public pages is permitted.
  - One request every REQUEST_DELAY seconds, single threaded.
  - Pages are cached on disk, so re-running costs nothing and the board is not
    hit twice for the same page.
  - Usernames are never recorded. The posts are used as language samples and
    author identity is irrelevant to the task.
  - Quotes of other posts and signature adverts are stripped, so each row is
    one person's own words.

Usage:
    python src/collect_posts.py                  # uses data/annotation/threads.txt
    python src/collect_posts.py --max-pages 5
    python src/collect_posts.py --no-finance-filter   # keep non-finance posts too
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

from config import REPO_ROOT

ANNOTATION_DIR = REPO_ROOT / "data" / "annotation"
THREADS_FILE = ANNOTATION_DIR / "threads.txt"
COLLECTED = ANNOTATION_DIR / "collected.txt"
CACHE_DIR = REPO_ROOT / "data" / "raw" / "html_cache"

REQUEST_DELAY = 1.5  # seconds between requests; robots.txt sets no crawl-delay
USER_AGENT = "Mozilla/5.0 (compatible; student research project; contact via GitHub)"
POSTS_PER_PAGE = 20

# Nigerian Pidgin markers, split by how much a single hit is worth.
#
# An earlier version matched these as plain substrings and was badly wrong:
# " na" matched " naira", "im " matched "him ", "dat " matched nothing useful.
# The result was a "Pidgin" set full of standard English. Two fixes: match on
# word boundaries, and separate markers that are almost never standard English
# from ones that are ordinary English words too.

# Rarely or never standard English. One of these is enough.
STRONG_MARKERS = (
    "dey", "wetin", "abeg", "wahala", "sabi", "shey", "comot", "wey", "oga",
    "pikin", "jare", "nawa", "biko", "naim", "sef", "abi", "una", "abeg",
    "waka", "gbadun", "wahalla", "ehn", "ooo", "oo", "kpata", "yawa",
    "sha", "wan", "chop", "yarn", "waka", "gist", "oyibo", "ur own",
)

# Ordinary English words that are also Pidgin function words. A post needs at
# least two distinct ones of these before it counts on their strength alone.
WEAK_MARKERS = (
    "na", "don", "fit", "nor", "dat", "dis", "im", "e", "go", "make", "say",
    "no", "get", "take", "put", "plenty", "small", "well",
)

STRONG_RE = re.compile("(?<![a-z])(" + "|".join(sorted(set(STRONG_MARKERS))) + ")(?![a-z])")
WEAK_RE = re.compile("(?<![a-z])(" + "|".join(sorted(set(WEAK_MARKERS))) + ")(?![a-z])")

# Topic filter. Broad on purpose: it is better to collect a post that turns out
# not to be about money and exclude it by hand than to silently miss one.
FINANCE_WORDS = (
    "bitcoin", "btc", "crypto", "coin", "money", "naira", "cash", "invest",
    "p2p", "exchange", "wallet", "scam", "trade", "trading", "bank", "profit",
    "loss", "hold", "hodl", "payment", "withdraw", "deposit", "fund", "price",
    "pump", "dump", "market", "altcoin", "stablecoin", "usdt", "transfer",
    "charge", "fee", "dollar", "rate", "loan", "debt", "save", "savings",
)

WHITESPACE = re.compile(r"\s+")
TOPIC_ID = re.compile(r"topic=(\d+)")


def harvest_board(board, pages):
    """Collect topic ids from the board index.

    Breadth beats depth here. Twenty posts from one thread would all be replies
    to the same prompt, so the test set would inherit that thread's topic and
    mood. Scanning many threads shallowly gives a more varied sample.
    """
    found = []
    for page in range(pages):
        url = f"https://bitcointalk.org/index.php?board={board}.{page * 40}"
        try:
            html, _ = fetch(url)
        except Exception as error:
            print(f"  board page {page}: {error}")
            break
        soup = BeautifulSoup(html, "html.parser")
        before = len(found)
        for anchor in soup.find_all("a", href=True):
            match = re.search(r"topic=(\d+)\.0$", anchor["href"])
            if match and match.group(1) not in found:
                found.append(match.group(1))
        print(f"  board page {page}: {len(found) - before} new topics "
              f"({len(found)} total)")
    return found


def read_threads():
    """Thread URLs or bare topic ids, one per line, # for comments."""
    if not THREADS_FILE.exists():
        raise SystemExit(
            f"{THREADS_FILE} does not exist. Create it with one thread URL per line."
        )
    threads = []
    for line in THREADS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = TOPIC_ID.search(line)
        threads.append(match.group(1) if match else line)
    return threads


def fetch(url, attempts=3):
    """Fetch a page, caching it on disk so reruns do not hit the server.

    Retries on transient network errors. An earlier version let a single read
    timeout abort an entire crawl, losing every page gathered up to that point.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    cached = CACHE_DIR / f"{key}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace"), True

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                html = response.read().decode("utf-8", "replace")
            cached.write_text(html, encoding="utf-8")
            time.sleep(REQUEST_DELAY)
            return html, False
        except Exception as error:  # timeouts, resets, DNS blips
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise OSError(f"{url}: {last_error}")


def page_count(soup):
    """Largest page offset linked from the page navigation, plus one page."""
    offsets = [0]
    for anchor in soup.find_all("a", href=True):
        match = re.search(r"topic=\d+\.(\d+)", anchor["href"])
        if match:
            offsets.append(int(match.group(1)))
    return max(offsets) // POSTS_PER_PAGE + 1


def extract_posts(soup):
    """One entry per post, with quoted text and signature adverts removed."""
    posts = []
    for node in soup.find_all("div", class_="post"):
        # Remove quotes first: a quote is someone else's words, and leaving it in
        # would attribute another person's sentiment to this post.
        for junk in node.find_all(class_=["quote", "quoteheader", "signature"]):
            junk.decompose()
        text = WHITESPACE.sub(" ", node.get_text(" ", strip=True)).strip()
        if text:
            posts.append(text)
    return posts


def looks_pidgin(text, min_weak=3):
    """One strong marker, or several distinct weak ones.

    Word boundaries matter: substring matching made " na" fire on " naira" and
    "im " on "him ", which passed plain English through as Pidgin.
    """
    lowered = text.lower()
    if STRONG_RE.search(lowered):
        return True
    return len(set(WEAK_RE.findall(lowered))) >= min_weak


def about_money(text):
    lowered = text.lower()
    return any(word in lowered for word in FINANCE_WORDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=4,
                        help="pages to read per thread (20 posts per page)")
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--no-pidgin-filter", action="store_true")
    parser.add_argument("--no-finance-filter", action="store_true")
    parser.add_argument("--harvest-board", type=int, metavar="PAGES",
                        help="first collect topic ids from board 275 index pages")
    args = parser.parse_args(argv)

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.harvest_board:
        print(f"Harvesting topic ids from board 275, {args.harvest_board} pages")
        harvested = harvest_board(275, args.harvest_board)
        existing_lines = THREADS_FILE.read_text(encoding="utf-8") if THREADS_FILE.exists() else ""
        new = [t for t in harvested if t not in existing_lines]
        with THREADS_FILE.open("a", encoding="utf-8") as handle:
            handle.write("\n# --- harvested from the board index ---\n")
            for topic in new:
                handle.write(f"https://bitcointalk.org/index.php?topic={topic}.0\n")
        print(f"  added {len(new)} new threads to {THREADS_FILE.name}\n")

    threads = read_threads()
    print(f"{len(threads)} threads listed, up to {args.max_pages} pages each\n")

    seen, kept = set(), []
    counts = {"fetched": 0, "cached": 0, "posts": 0, "too_short": 0,
              "too_long": 0, "not_pidgin": 0, "not_finance": 0, "duplicate": 0}

    for topic in threads:
        base = f"https://bitcointalk.org/index.php?topic={topic}"
        try:
            html, was_cached = fetch(f"{base}.0")
        except Exception as error:
            print(f"  topic {topic}: FAILED ({error}) - skipping")
            continue
        counts["cached" if was_cached else "fetched"] += 1

        soup = BeautifulSoup(html, "html.parser")
        total_pages = min(page_count(soup), args.max_pages)
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True)[:60] if title_tag else topic

        thread_kept = 0
        for page in range(total_pages):
            if page == 0:
                page_soup = soup
            else:
                try:
                    html, was_cached = fetch(f"{base}.{page * POSTS_PER_PAGE}")
                except Exception:
                    break
                counts["cached" if was_cached else "fetched"] += 1
                page_soup = BeautifulSoup(html, "html.parser")

            for text in extract_posts(page_soup):
                counts["posts"] += 1
                words = len(text.split())
                if words < args.min_words:
                    counts["too_short"] += 1
                    continue
                if words > args.max_words:
                    counts["too_long"] += 1
                    continue
                if not args.no_pidgin_filter and not looks_pidgin(text):
                    counts["not_pidgin"] += 1
                    continue
                if not args.no_finance_filter and not about_money(text):
                    counts["not_finance"] += 1
                    continue
                fingerprint = text.lower()[:160]
                if fingerprint in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(fingerprint)
                kept.append(text)
                thread_kept += 1

        print(f"  [{thread_kept:>3} kept] {title} ({total_pages}p)")

    existing = []
    if COLLECTED.exists():
        existing = [
            line.strip()
            for line in COLLECTED.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    already = {line.lower()[:160] for line in existing}
    fresh = [post for post in kept if post.lower()[:160] not in already]

    header = (
        "# Candidate Pidgin finance posts from the Bitcointalk Nigeria board.\n"
        "# Collected with src/collect_posts.py. Quotes and signatures stripped,\n"
        "# usernames not recorded. Still to be labelled by hand under\n"
        "# data/annotation/GUIDELINES.md - nothing here is labelled yet.\n"
    )
    COLLECTED.write_text(
        header + "\n".join(existing + fresh) + "\n", encoding="utf-8"
    )

    print(f"\npages fetched {counts['fetched']}, served from cache {counts['cached']}")
    print(f"posts seen                {counts['posts']}")
    print(f"  dropped, too short      {counts['too_short']}")
    print(f"  dropped, too long       {counts['too_long']}")
    print(f"  dropped, not Pidgin     {counts['not_pidgin']}")
    print(f"  dropped, not finance    {counts['not_finance']}")
    print(f"  dropped, duplicate      {counts['duplicate']}")
    print(f"kept this run             {len(kept)}")
    print(f"new (not already present) {len(fresh)}")
    print(f"total in {COLLECTED.name}: {len(existing) + len(fresh)}")
    print("\nNext: read them, then run python src/make_annotation_sheet.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
