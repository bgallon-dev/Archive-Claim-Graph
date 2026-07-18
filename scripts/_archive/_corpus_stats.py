"""Quick composition analysis of the Tutorstructure newspaper corpus.

Categorizes every article file by filename, then reads a stratified sample to
report source type, word-count distribution (full page vs article), headline
presence, and per-year counts.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = Path(r"C:\Users\Benjamin\Desktop\Tutorstructure\corpus")
_ARTICLE_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_")
_YEAR_RE = re.compile(r"_(\d{4})-\d{2}-\d{2}_")

files = [f for f in CORPUS.rglob("*.json") if not f.name.startswith(".") and _ARTICLE_RE.search(f.name)]
print(f"total article files: {len(files)}")

# Cheap categorization by filename prefix (no reads).
prefix = Counter("ca_" if f.name.startswith("ca_") else ("digit_" if f.name[:1].isdigit() else "other_") for f in files)
print(f"by filename prefix: {dict(prefix)}")

# Per-year counts (from filename date).
by_year: Counter = Counter()
for f in files:
    m = _YEAR_RE.search(f.name)
    if m:
        by_year[m.group(1)] += 1
print("by year:", dict(sorted(by_year.items())))

# Stratified sample: every Nth file so we span all years/papers.
N = max(1, len(files) // 500)
sample = files[::N][:500]
print(f"\nreading sample of {len(sample)} (stride {N}) ...")

src = Counter()
has_headline = 0
words: list[int] = []
words_by_src: dict[str, list[int]] = defaultdict(list)
for f in sample:
    try:
        a = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    s = a.get("source", "?")
    src[s] += 1
    if (a.get("headline") or "").strip():
        has_headline += 1
    wc = (a.get("scoring") or {}).get("word_count")
    if not isinstance(wc, int):
        wc = len(str(a.get("article", "")).split())
    words.append(wc)
    words_by_src[s].append(wc)

print(f"\nsource field: {dict(src)}")
print(f"headline present: {has_headline}/{len(sample)} ({100*has_headline/max(1,len(sample)):.0f}%)")
if words:
    words.sort()
    pct = lambda p: words[min(len(words) - 1, int(p * len(words)))]
    print(f"word_count: min={words[0]} p25={pct(.25)} median={statistics.median(words)} p75={pct(.75)} p90={pct(.9)} max={words[-1]}")
    full_pages = sum(1 for w in words if w >= 1500)
    print(f"'full page'-sized (>=1500 words): {full_pages}/{len(words)} ({100*full_pages/len(words):.0f}%)")
for s, ws in words_by_src.items():
    if ws:
        print(f"  source={s}: n={len(ws)} median_words={int(statistics.median(ws))} max={max(ws)}")
