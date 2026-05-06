"""
categorize_newspaper_extraction.py — Classify each under-extracted newspaper
doc into one of four failure modes so we can prioritize a real fix.

Categories (assigned by score-then-fallback rules; per-doc scores also written
to the CSV so you can re-tune thresholds offline):

  column_flattened : multi-column source PDF where the parser linearized
                rows across columns instead of reading column-by-column.
                Paragraphs end up syntactically corrupt with words from two
                columns interleaved. Detected by either the known-broken
                doc_id prefix (`pnw_forum_*`) or anomalously high words-
                per-sentence ratios. Re-extracting won't help — the input
                is corrupt at the parser layer.
  non_prose_dilution : narrative-style document (e.g. Turnbull annual
                reports) with substantial bundled non-prose content —
                waterfowl census tables OCR'd as flat number-strings,
                classified ad pages bound in. The prose IS being extracted
                correctly; the apparent deficit comes from char_count
                inflated by table/ad content the extractor correctly skips.
  list        : numbered/tabular content (prize lists, allotment rosters).
                Extractor correctly produces few claims; deficit flag is a
                false positive.
  classified  : real-estate / sale ads. Multiple distinct sale items packed
                into one paragraph with prices and brief descriptors. Real
                opportunity ONLY if you want sentence-level extraction
                (each ad = one commercial_transaction claim).
  market_data : financial / commodity price columns. Short paragraphs
                dominated by digits and currency. Tabular data, not prose.
  petition    : court petitions and formal addressed documents ("To the
                Honorable...", "the undersigned petitioners hereby..."). The
                addressing line + formal templating produces 0-claim
                paragraphs that aren't real signal.
  legal       : city ordinance / bond / assessment boilerplate. Mostly
                repetitive legal prose with limited additional claim density.
  ocr_damage  : prose with OCR letter↔digit substitutions ("Gr0cers", "Jh0W",
                "S2S"). Detected via either body-token rate OR a mangled
                title. Real extraction opportunity if OCR can be repaired.
  schedule    : calendar / agenda / program of scheduled items. Each line
                announces an event but paragraph-level extraction sees one
                topic per paragraph and under-claims.
  fragment    : very short stubs (<1500 chars and few paragraphs). Too thin
                for the extractor to find signal in regardless of content.
  continuation: jump-page continuation of an article whose first half is on
                a different newspaper page. Detected via OCR-fragmented
                "Continued from page X" headers that render as letter-by-
                letter tokens ("G O N TIN u E D r o P A G F. Q N F."). The
                visible content is mid-article, often without context, so
                the extractor has thin grounding.
  page        : full newspaper-page docs ("The Spokane Press — YYYY-MM-DD
                — ppN") that span multiple unrelated articles in a single
                "doc". Per-doc claim density isn't directly comparable to
                single-article docs — different content unit.
  other       : doesn't match the above — the genuine prose articles where
                extraction may be underperforming. After the content-type
                filters above this bucket should be small.

Usage:
    python tools/categorize_newspaper_extraction.py [options]

Options:
    --input FILE    Source CSV (default: under_extracted_docs.csv).
    --output FILE   Output CSV (default: newspaper_categorized.csv).

Requires NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (loaded from .env).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gemynd.shared.env import load_dotenv
from gemynd.shared.settings import Settings
from gemynd.ingest.graph.writer import _build_driver_kwargs

try:
    from neo4j import GraphDatabase
except ImportError:
    sys.exit("neo4j package is not installed. Install with: pip install neo4j")


PARAGRAPHS_QUERY = """
MATCH (d:Document {doc_id: $doc_id})
      -[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
      -[:HAS_PARAGRAPH]->(p:Paragraph)
RETURN p.clean_text AS text, coalesce(p.char_count, 0) AS chars
"""

# A token like "Gr0cers" / "Jh0W" / "S2S" — letter-digit-letter run inside a
# word. Strong signal of digit-for-letter OCR substitution. Excludes pure
# numeric tokens, years, and trailing-digit identifiers ("page2", "100b").
_OCR_MIXED_RE = re.compile(r"\b[A-Za-z]+\d[A-Za-z]+\w*\b")
_DIGIT_LEAD_LETTER_RE = re.compile(r"\b\d{1,2}[A-Za-z]{2,}\b")  # "1nstall", "70be"
_NUMBERED_LIST_RE = re.compile(r"^\s*\d{1,4}[\-\—\.\)]+\s*[A-Z]", re.MULTILINE)
_LEGAL_MARKERS = re.compile(
    r"\b(section|ordinance|shall be|hereby|whereas|assessed?|"
    r"bonds?|act of|levied|assessment district|the city of)\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)
# Time-of-day patterns: "2:45 p. m.", "1:30 PM", "from 9 to 11", "2 to 3 p.m."
_TIME_RE = re.compile(
    r"\b\d{1,2}\s*[:.]\s*\d{2}\s*[ap]\.?\s*m\.?",
    re.IGNORECASE,
)
# Classified-ad signals: dollar amounts in body text. We measure density
# rather than presence — every news doc mentions money occasionally; a
# classifieds page has dollar signs every few sentences.
_DOLLAR_RE = re.compile(r"\$\s?\d{1,5}(?:[,.]\d{2,3})?")
# Sale / rent / trade language clusters in classifieds.
_SALE_TERMS_RE = re.compile(
    r"\b(for sale|for rent|will trade|to trade|cash|easy terms|"
    r"mortgage|owner|small payment down|monthly payments|"
    r"bargain|good buy|ranches|acreage)\b",
    re.IGNORECASE,
)
# Petition / formal-address markers — opening templating that occupies
# whole paragraphs without producing claims.
_PETITION_RE = re.compile(
    r"\b(the honorable|undersigned petitioners?|hereby respectfully|"
    r"hereby petition|before the court|we[, ]+the (undersigned|people)|"
    r"come now the|to the (above|court|honorable))\b",
    re.IGNORECASE,
)
# Market-data signals: multiple price-or-quote tokens. "75c", "$1.20",
# "Bid 99", "Spot", "Closed at" — any combination of these patterns
# indicates a commodity/stock-quote column.
_MARKET_TERMS_RE = re.compile(
    r"\b(bid|asked|spot|closed at|opening|wheat|corn|oats|barley|"
    r"hogs|cattle|copper|silver|bullion)\b",
    re.IGNORECASE,
)
_QUOTE_PRICE_RE = re.compile(r"\b\d+\s?\d?[/\.]\d{1,2}\b|\b\d+c\b")


_ALLCAPS_INTRUSION_RE = re.compile(r"\b[A-Z]{2,}[a-z][A-Z]+\b")
_NEWSPAPER_PAGE_RE = re.compile(
    r"^The Spokane Press.*\bpp\d+\s*$",
    re.IGNORECASE,
)
# Doc-ID prefixes confirmed to have parser-broken column flattening. See
# project_pnw_forum_segmentation memory for verification details.
_KNOWN_COLUMN_FLATTENED_PREFIXES = ("pnw_forum_",)
# Sentence-end punctuation used in the words-per-sentence heuristic.
_SENTENCE_END_RE = re.compile(r"[.!?]+")


def _ocr_mangled(text: str) -> bool:
    """True if the text has at least one letter-digit-letter, digit-leading,
    or allcaps-with-lowercase-intrusion OCR-substitution token. Used for
    title-level mangling where even one token like "Cp0KANE" or "MAGhATES"
    is a strong signal."""
    if not text:
        return False
    return bool(
        _OCR_MIXED_RE.search(text)
        or _DIGIT_LEAD_LETTER_RE.search(text)
        or _ALLCAPS_INTRUSION_RE.search(text)
    )


def _is_continuation_title(title: str) -> bool:
    """True if title looks like an OCR-shattered 'Continued from Page X'
    header — five-or-more single-character tokens, typically because the
    OCR engine emitted each glyph of a stylized header as a separate word.
    """
    if not title:
        return False
    tokens = [t.strip(".,;:!?@-") for t in title.split() if t.strip(".,;:!?@-")]
    single_char_tokens = sum(1 for t in tokens if len(t) == 1)
    return single_char_tokens >= 5


def _score_doc(paragraphs: list[dict], title: str = "", doc_id: str = "") -> dict:
    """Return per-doc heuristic scores. All proportions in [0, 1]."""
    if not paragraphs:
        return {"list": 0.0, "legal": 0.0, "ocr_damage": 0.0, "para_n": 0,
                "title_mangled": False, "ocr_mixed_tokens": 0,
                "numbered_list_lines": 0, "legal_paragraphs": 0,
                "total_chars": 0,
                "non_prose_dilution": 0.0, "column_flattened": 0.0,
                "is_known_column_flattened": False, "words_per_sentence": 0.0,
                "numeric_paragraphs": 0}

    full_text = "\n\n".join(p["text"] or "" for p in paragraphs)
    total_chars = max(len(full_text), 1)
    total_word_tokens = max(len(re.findall(r"\b[A-Za-z]\w*\b", full_text)), 1)

    # List signal: numbered roster lines + high digit/dash density.
    numbered_lines = len(_NUMBERED_LIST_RE.findall(full_text))
    digit_density = sum(1 for ch in full_text if ch.isdigit()) / total_chars
    dash_density = full_text.count("-") / total_chars
    list_score = (
        min(numbered_lines / max(len(paragraphs), 1), 1.0) * 0.6
        + min(digit_density / 0.10, 1.0) * 0.25
        + min(dash_density / 0.02, 1.0) * 0.15
    )

    # Legal signal: ordinance/bond/assessment markers per paragraph.
    legal_paras = sum(
        1 for p in paragraphs if _LEGAL_MARKERS.search(p["text"] or "")
    )
    legal_score = legal_paras / max(len(paragraphs), 1)

    # OCR damage signal: mid-word digit-letter mixing in body.
    ocr_mixed = len(_OCR_MIXED_RE.findall(full_text)) + len(
        _DIGIT_LEAD_LETTER_RE.findall(full_text)
    )
    ocr_body_score = ocr_mixed / total_word_tokens

    # Title-level mangling — separate from body because headlines are
    # rendered in fancy fonts that OCR garbles even when the body is clean.
    title_mangled = _ocr_mangled(title or "")
    is_continuation = _is_continuation_title(title or "")
    is_newspaper_page = bool(
        _NEWSPAPER_PAGE_RE.match(re.sub(r"\s+", " ", (title or "").strip()))
    )

    # Schedule signal: paragraphs containing both a weekday and a time-of-day
    # token. Calendar listings and conference agendas hit this consistently.
    schedule_paras = sum(
        1 for p in paragraphs
        if _WEEKDAY_RE.search(p["text"] or "") and _TIME_RE.search(p["text"] or "")
    )
    schedule_score = schedule_paras / max(len(paragraphs), 1)

    # Classified signal: density of dollar amounts + sale-term clusters per kchar.
    dollar_count = len(_DOLLAR_RE.findall(full_text))
    sale_term_count = len(_SALE_TERMS_RE.findall(full_text))
    classified_score = (
        min((dollar_count / max(total_chars / 1000, 1)) / 3.0, 1.0) * 0.6
        + min((sale_term_count / max(total_chars / 1000, 1)) / 2.0, 1.0) * 0.4
    )

    # Petition signal: presence of petition markers in any paragraph + the
    # title style. Petitions tend to be short docs with formal addressing.
    petition_paras = sum(
        1 for p in paragraphs if _PETITION_RE.search(p["text"] or "")
    )
    petition_score = petition_paras / max(len(paragraphs), 1)

    # Market data signal: market terms + density of quote-price tokens.
    # Capped to short docs because Turnbull narrative reports mention
    # "wheat", "cattle", "corn" etc. in agricultural-impact prose, and
    # measurement-style tokens like "12.5" / "5c" trip the price regex
    # millions of times in long ecological reports without it being a
    # market column. Real market columns are 1-3k chars.
    market_term_count = len(_MARKET_TERMS_RE.findall(full_text))
    quote_count = len(_QUOTE_PRICE_RE.findall(full_text))
    if total_chars > 8000:
        market_data_score = 0.0
    else:
        market_data_score = (
            min((market_term_count / max(total_chars / 1000, 1)) / 2.0, 1.0) * 0.5
            + min((quote_count / max(total_chars / 1000, 1)) / 3.0, 1.0) * 0.5
        )

    # Non-prose dilution signal: a paragraph counts as "non-prose" if it's
    # either heavily tabular (combined digit + structural-punctuation
    # density >= 0.15) or classified-ad style (contains dollar amounts and
    # sale terms). Score is the FRACTION OF DOC CHARS in non-prose
    # paragraphs, not paragraph count — a single 5k-char census-table
    # paragraph counts much more than five 100-char snippets.
    def _tabular_density(text: str) -> float:
        if not text:
            return 0.0
        n_struct = sum(1 for ch in text if ch.isdigit() or ch in ";:|/")
        return n_struct / len(text)

    def _is_non_prose(text: str) -> bool:
        if not text:
            return False
        if _tabular_density(text) >= 0.15:
            return True
        # Classified-style: has dollar amount AND a sale term in same paragraph.
        if _DOLLAR_RE.search(text) and _SALE_TERMS_RE.search(text):
            return True
        return False

    non_prose_chars = sum(
        len(p["text"] or "") for p in paragraphs if _is_non_prose(p["text"] or "")
    )
    numeric_paragraphs = sum(
        1 for p in paragraphs if _is_non_prose(p["text"] or "")
    )
    non_prose_dilution_score = non_prose_chars / max(total_chars, 1)
    # Only fire on substantial docs — small docs with table-like content
    # are usually classifieds (already a separate category).
    if total_chars < 20_000:
        non_prose_dilution_score = 0.0

    # Column-flattened signal: rely primarily on the known-broken doc_id
    # prefix list. The words-per-sentence heuristic produces too many false
    # positives on legitimate Edwardian-era news prose, so we use it only
    # as a soft secondary signal that requires very high values to fire.
    is_known_column_flattened = any(
        doc_id.startswith(p) for p in _KNOWN_COLUMN_FLATTENED_PREFIXES
    )
    sentence_count = max(len(_SENTENCE_END_RE.findall(full_text)), 1)
    words_per_sentence = total_word_tokens / sentence_count
    if is_known_column_flattened:
        column_flattened_score = 1.0
    elif words_per_sentence >= 80:
        # Genuinely unreadable — even the longest 1900s prose rarely tops
        # 60 words/sentence; >=80 is structurally broken.
        column_flattened_score = 0.7
    else:
        column_flattened_score = 0.0

    return {
        "list": round(list_score, 3),
        "legal": round(legal_score, 3),
        "ocr_damage": round(ocr_body_score, 3),
        "schedule": round(schedule_score, 3),
        "classified": round(classified_score, 3),
        "petition": round(petition_score, 3),
        "market_data": round(market_data_score, 3),
        "title_mangled": title_mangled,
        "is_continuation": is_continuation,
        "is_newspaper_page": is_newspaper_page,
        "non_prose_dilution": round(non_prose_dilution_score, 3),
        "column_flattened": round(column_flattened_score, 3),
        "is_known_column_flattened": is_known_column_flattened,
        "words_per_sentence": round(words_per_sentence, 1),
        "numeric_paragraphs": numeric_paragraphs,
        "para_n": len(paragraphs),
        "total_chars": len(full_text),
        "ocr_mixed_tokens": ocr_mixed,
        "dollar_count": dollar_count,
        "sale_term_count": sale_term_count,
        "petition_paragraphs": petition_paras,
        "market_term_count": market_term_count,
        "quote_count": quote_count,
        "numbered_list_lines": numbered_lines,
        "legal_paragraphs": legal_paras,
        "schedule_paragraphs": schedule_paras,
    }


def categorize(scores: dict) -> str:
    """Pick the dominant category. Order matters — root-cause categories
    (parser-broken / non-prose-dilution) run first since those decisions
    determine whether re-extraction would help at all. Content-type filters
    run next, then recall-opportunity categories last."""
    # Root-cause categories: input is structurally compromised.
    if scores["column_flattened"] >= 0.60 or scores["is_known_column_flattened"]:
        return "column_flattened"
    # Threshold tuned empirically: 5% non-prose chars cleanly separates
    # the diluted Turnbull narratives (Tbl 1963 May-Aug at 9.6%, Tbl 1956
    # Sept-Dec at 9.1%, Tbl 1954 Sept-Dec at 8.6%) from the rest of the
    # corpus, where every other doc scores < 4%.
    if scores["non_prose_dilution"] >= 0.05:
        return "non_prose_dilution"
    # Unambiguous content types.
    if scores["list"] >= 0.45:
        return "list"
    if scores["schedule"] >= 0.30:
        return "schedule"
    if scores["classified"] >= 0.50:
        return "classified"
    if scores["market_data"] >= 0.45:
        return "market_data"
    if scores["petition"] >= 0.20:  # one-in-five paragraphs is enough
        return "petition"
    # Continuation pages: OCR-shattered "Continued from Page X" header means
    # we're seeing the back half of an article without context, not a
    # standalone piece. Treated as content-type artifact, not a recall miss.
    if scores["is_continuation"]:
        return "continuation"
    # Newspaper page-level docs are mixed-content multi-article pages, not
    # a comparable content unit to single-article docs.
    if scores["is_newspaper_page"]:
        return "page"
    # Recall-opportunity categories.
    if scores["ocr_damage"] >= 0.02 or scores["title_mangled"]:
        return "ocr_damage"
    if scores["total_chars"] < 1500 and scores["para_n"] < 6:
        return "fragment"
    if scores["legal"] >= 0.35:
        return "legal"
    return "other"


def fetch_paragraphs(settings: Settings, doc_id: str) -> list[dict]:
    driver_kwargs = _build_driver_kwargs(
        uri=settings.neo4j_uri,
        trust_mode=settings.neo4j_trust,
        ca_cert_path=settings.neo4j_ca_cert,
    )
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        **driver_kwargs,
    )
    with driver.session(database=settings.neo4j_database) as session:
        rows = [dict(r) for r in session.run(PARAGRAPHS_QUERY, doc_id=doc_id)]
    driver.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="under_extracted_docs.csv")
    parser.add_argument("--output", default="newspaper_categorized.csv")
    args = parser.parse_args()

    load_dotenv(_REPO_ROOT / ".env")
    settings = Settings.from_env()
    if not settings.neo4j_uri:
        sys.exit("NEO4J_URI not set.")

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    with in_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["institution_id"] == "spokane_newspaper"]
    print(f"Loaded {len(rows)} flagged newspaper docs from {in_path}")

    out_rows: list[dict] = []
    cat_counts: Counter = Counter()
    for i, row in enumerate(rows, start=1):
        paras = fetch_paragraphs(settings, row["doc_id"])
        scores = _score_doc(
            paras, title=row.get("title", ""), doc_id=row.get("doc_id", ""),
        )
        category = categorize(scores)
        cat_counts[category] += 1
        out_rows.append({
            "doc_id": row["doc_id"],
            "title": row["title"],
            "report_year": row["report_year"],
            "char_count": row["char_count"],
            "claim_count": row["claim_count"],
            "deficit_ratio": row["deficit_ratio"],
            "category": category,
            "list_score": scores["list"],
            "legal_score": scores["legal"],
            "ocr_damage_score": scores["ocr_damage"],
            "schedule_score": scores["schedule"],
            "classified_score": scores["classified"],
            "petition_score": scores["petition"],
            "market_data_score": scores["market_data"],
            "title_mangled": scores["title_mangled"],
            "is_continuation": scores["is_continuation"],
            "is_newspaper_page": scores["is_newspaper_page"],
            "non_prose_dilution_score": scores["non_prose_dilution"],
            "column_flattened_score": scores["column_flattened"],
            "is_known_column_flattened": scores["is_known_column_flattened"],
            "words_per_sentence": scores["words_per_sentence"],
            "numeric_paragraphs": scores["numeric_paragraphs"],
            "paragraph_count": scores["para_n"],
            "ocr_mixed_tokens": scores["ocr_mixed_tokens"],
            "dollar_count": scores["dollar_count"],
            "sale_term_count": scores["sale_term_count"],
            "petition_paragraphs": scores["petition_paragraphs"],
            "market_term_count": scores["market_term_count"],
            "quote_count": scores["quote_count"],
            "numbered_list_lines": scores["numbered_list_lines"],
            "legal_paragraphs": scores["legal_paragraphs"],
            "schedule_paragraphs": scores["schedule_paragraphs"],
        })
        if i % 25 == 0 or i == len(rows):
            print(f"  ... {i}/{len(rows)}")

    out_path = Path(args.output)
    fieldnames = list(out_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print()
    print(f"Wrote {len(out_rows)} rows to {out_path}")
    print()
    print("Category distribution:")
    for cat, n in cat_counts.most_common():
        pct = n / len(out_rows) * 100
        print(f"  {cat:12s} {n:>4d}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
