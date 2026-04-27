"""Cross-corpus NER enrichment pass.

Scans newspaper paragraph text for mentions of GLO patent grantees and writes
REFERS_TO / POSSIBLY_REFERS_TO relationships from new Mention nodes to the
existing Person entities.  This bridges the corpus isolation in the ingest
pipeline without re-running the full extraction stack.

Typical usage (via the CLI driver)::

    from neo4j import GraphDatabase
    from gemynd.ingest.enrichment.cross_corpus_ner import CrossCorpusNERPass

    driver = GraphDatabase.driver(uri, auth=(user, password))
    ner = CrossCorpusNERPass(driver, resources_dir=Path("gemynd/resources"))
    ner.run(dry_run=False, output_report=Path("matches.csv"))
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from gemynd.core.ids import make_entity_id, make_mention_id, make_run_id
from gemynd.core.models import EntityRecord, EntityResolutionRecord, MentionRecord
from gemynd.core.resolver import normalize_name, similarity_score

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR corruption detection (mirrors context_assembler.py)
# ---------------------------------------------------------------------------

_OCR_GARBAGE_RE = re.compile(r"[\^\x00-\x08\x0b\x0c\x0e-\x1f]")
_OCR_SPACED_CHARS_RE = re.compile(r"(?<!\S)(?:[A-Za-z] ){5,}")


def _is_ocr_corrupted(text: str) -> bool:
    if _OCR_GARBAGE_RE.search(text):
        return True
    if _OCR_SPACED_CHARS_RE.search(text):
        return True
    return False


# Collapse runs of space-separated single letters within a span.
# Applied to individual candidate spans, not the whole paragraph.
# "E D W A R D" → "EDWARD", "C O P E L A N D" → "COPELAND"
_SPAN_SPACED_RE = re.compile(r"\b([A-Za-z]) (?=[A-Za-z](?: |$))")


def _collapse_spaced_chars(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _SPAN_SPACED_RE.sub(r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------

# Capitalized phrases: 1–4 title-case tokens.  We intentionally allow
# single-token matches (e.g. "Copeland") because patent grantee names may
# appear in a newspaper without a first name.
# Require exactly 2–4 title-case tokens so single common words don't enter
# scoring.  Two-token minimum eliminates "Lake", "His", "They" etc.
_CANDIDATE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

# Title-case words that must not appear in any token of a person-name candidate.
# Covers: articles/prepositions, pronouns, months, days, honorifics,
# occupational titles, and common place-name prefixes.
_STOPWORDS: frozenset[str] = frozenset({
    # Articles / prepositions / conjunctions
    "the", "and", "but", "for", "nor", "yet", "so",
    "of", "in", "on", "at", "by", "to", "an", "a",
    "if", "as", "or", "no", "not",
    "that", "this", "those", "these", "with", "from",
    # Pronouns
    "he", "she", "it", "we", "you", "they",
    "his", "her", "its", "our", "your", "their",
    # Months / days
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Honorifics
    "mr", "mrs", "ms", "miss", "sir", "dr", "prof", "rev", "hon",
    # Occupational titles common in newspapers
    "officer", "attorney", "coroner", "councilman", "councilwoman", "organizer",
    "judge", "mayor", "sheriff", "senator", "representative", "captain", "general",
    "colonel", "detective", "inspector", "commissioner", "superintendent",
    "secretary", "treasurer", "president", "director", "manager", "editor",
    "reporter", "correspondent", "clerk", "deputy", "constable", "marshal",
    # Place-name prefixes
    "street", "avenue", "road", "block", "county", "state", "city",
    "san", "fort", "mount", "saint", "lake", "camp",
    "spokane", "washington",
    # Common abbreviations and noise
    "st", "co", "ave", "inc", "ltd",
    # Common sentence-initial words that happen to be title-case
    "ten", "nine", "eight", "seven", "six", "five", "four", "three", "two", "one",
    # Common verbs that start sentences in newspaper text
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "when", "where", "which", "what", "who", "how", "why",
    # Additional occupational titles and descriptors found in newspaper text
    "agent", "chief", "engineer", "professor", "doctor",
    "race", "horse",
})


def _extract_candidates(text: str) -> list[tuple[str, int, int]]:
    """Return (surface_form, start, end) tuples for capitalized-phrase candidates.

    Requires 2+ title-case tokens so that single capitalized words (sentence
    starts, place name components, common nouns) never enter scoring.
    """
    results: list[tuple[str, int, int]] = []
    for m in _CANDIDATE_RE.finditer(text):
        surface = m.group(1)
        # Skip spans that contain newlines (OCR line-break artifacts)
        if "\n" in surface:
            continue
        norm = normalize_name(surface)
        tokens = norm.split()
        if len(tokens) < 2 or any(t in _STOPWORDS for t in tokens):
            continue
        results.append((surface, m.start(), m.end()))
    return results


# ---------------------------------------------------------------------------
# Name expansion (abbreviations + OCR variants)
# ---------------------------------------------------------------------------

def _abbreviation_forms(name: str) -> list[str]:
    """Generate abbreviated surface forms from a canonical full name.

    For "Edward B Copeland" produces:
      "Edward Copeland", "E Copeland", "E B Copeland"

    Last-name-only forms are intentionally excluded: single-word matches
    against common surnames produce too many false positives in newspaper text.
    """
    parts = name.split()
    if len(parts) < 2:
        return []
    forms: list[str] = []
    first, *middle, last = parts
    # First + Last (drop middle)
    forms.append(f"{first} {last}")
    # First initial + Last only
    forms.append(f"{first[0]} {last}")
    # First initial + middle initial(s) + Last
    if middle:
        mid_initials = " ".join(p[0] for p in middle)
        forms.append(f"{first[0]} {mid_initials} {last}")
    return forms


def _build_expanded_records(
    grantees: list[EntityRecord],
    matrix: "OCRConfusionMatrix",  # type: ignore[name-defined]  # imported lazily
    min_ocr_prob: float,
) -> list[EntityRecord]:
    """Expand each grantee into canonical + OCR-variant + abbreviation records."""
    from scripts.generate_ocr_variants import generate_variants  # local import to avoid circular deps

    expanded: list[EntityRecord] = []
    seen_norms: set[tuple[str, str]] = set()

    def _add(entity_id: str, display: str) -> None:
        norm = normalize_name(display)
        # Only register multi-token forms — single tokens create too many
        # false positives via the substring rule in similarity_score.
        if " " not in norm:
            return
        key = (entity_id, norm)
        if key in seen_norms:
            return
        seen_norms.add(key)
        expanded.append(EntityRecord(
            entity_id=entity_id,
            entity_type="Person",
            name=display,
            normalized_form=norm,
        ))

    for g in grantees:
        # Canonical form
        _add(g.entity_id, g.name)

        # Abbreviation forms
        abbrevs = _abbreviation_forms(g.name)
        for abbrev in abbrevs:
            _add(g.entity_id, abbrev)

        # OCR variants of the canonical name and first+last
        for src in [g.name] + abbrevs[:1]:
            for variant, prob in generate_variants(src, matrix, min_prob=min_ocr_prob):
                _add(g.entity_id, variant)

    return expanded


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_REFERS_TO_THRESHOLD = 0.85
_POSSIBLY_REFERS_TO_THRESHOLD = 0.65
_UNIQUENESS_GAP = 0.05


@dataclass(slots=True)
class _Match:
    surface_form: str
    collapsed_form: str
    normalized_form: str
    start: int
    end: int
    entity_id: str
    entity_name: str
    score: float
    relation_type: str
    ocr_suspect: bool


def _last_token_anchor(cand_norm: str, entity_norm: str, min_ratio: float = 0.84) -> bool:
    """Return True if the last token of candidate matches the last token of entity.

    This prevents first-name-only similarity from creating false positives:
    "George Frost" should not match "George Ross" just because "George" is exact.
    Single-character tokens (initials like "a", "b") are treated as wildcards.

    Uses raw SequenceMatcher ratio (NOT similarity_score) to avoid the substring
    boost: "anderson"/"sanders" share "ander" as a substring and would pass
    similarity_score at 0.88, but their raw SequenceMatcher ratio is 0.800, which
    is below the 0.84 threshold. Likewise "ander"/"sanders" = 0.833 < 0.84.
    """
    import difflib
    cand_tokens = cand_norm.split()
    entity_tokens = entity_norm.split()
    if not cand_tokens or not entity_tokens:
        return False
    last_cand = cand_tokens[-1]
    last_entity = entity_tokens[-1]
    if len(last_cand) <= 1 or len(last_entity) <= 1:
        return True  # Single-char initials don't constrain
    return difflib.SequenceMatcher(a=last_cand, b=last_entity).ratio() >= min_ratio





def _score_candidates(
    candidates: list[tuple[str, int, int]],
    expanded_records: list[EntityRecord],
    ocr_suspect: bool,
) -> list[_Match]:
    """Score each candidate span against expanded_records and return matches."""
    matches: list[_Match] = []
    for surface, start, end in candidates:
        # Try collapsed form first (handles spaced-char OCR in the span)
        collapsed = _collapse_spaced_chars(surface)
        norm = normalize_name(collapsed)
        is_ocr_via_collapse = (collapsed.lower() != surface.lower())

        scored: list[tuple[EntityRecord, float]] = []
        for entity in expanded_records:
            # Last-token anchor: require the surname (last word) to match well.
            # This prevents "George Frost" from matching "George Ross".
            # Bypass for spaced-char collapsed forms: "EDWARDCOPELAND" collapses to
            # a single-token norm where the anchor can't find a meaningful surname.
            if not is_ocr_via_collapse and not _last_token_anchor(norm, entity.normalized_form):
                continue
            raw_score = similarity_score(norm, entity.normalized_form)
            if raw_score < _POSSIBLY_REFERS_TO_THRESHOLD:
                continue
            scored.append((entity, raw_score))

        if not scored:
            continue
        scored.sort(key=lambda row: row[1], reverse=True)
        top_entity, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0

        if top_score >= _REFERS_TO_THRESHOLD and (top_score - second_score) >= _UNIQUENESS_GAP:
            rel = "REFERS_TO"
        elif top_score >= _POSSIBLY_REFERS_TO_THRESHOLD:
            rel = "POSSIBLY_REFERS_TO"
        else:
            continue

        # First-initial consistency: if the matching entity uses only an initial
        # for its first name (e.g. "A Johnson") and the candidate has a full first
        # name (e.g. "Sam Johnson"), the first letter must agree.  A mismatch
        # ("sam"[0]='s' ≠ 'a') caps to POSSIBLY_REFERS_TO — these are different
        # people who share a surname, not the same person seen twice.
        # Skipped for spaced-char collapsed forms where the first token carries
        # no reliable first-name signal.
        if rel == "REFERS_TO" and not is_ocr_via_collapse:
            entity_tokens = top_entity.normalized_form.split()
            cand_tokens = norm.split()
            entity_first = entity_tokens[0] if entity_tokens else ""
            cand_first = cand_tokens[0] if cand_tokens else ""
            if len(entity_first) == 1 and len(cand_first) > 1 and cand_first[0] != entity_first:
                rel = "POSSIBLY_REFERS_TO"

        matches.append(_Match(
            surface_form=surface,
            collapsed_form=collapsed,
            normalized_form=norm,
            start=start,
            end=end,
            entity_id=top_entity.entity_id,
            entity_name=top_entity.name,
            score=round(top_score, 4),
            relation_type=rel,
            ocr_suspect=ocr_suspect or is_ocr_via_collapse,
        ))
    return matches


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CrossCorpusNERPass:
    """Enrichment pass that links newspaper mentions to patent Person entities."""

    # Land patent grantees are connected via synthetic Paragraph nodes that are
    # NOT linked back to their Document via HAS_PARAGRAPH (patents have HAS_PAGE
    # only).  Query through the claim_type instead, which is always 'land_grant'
    # for patent-builder-generated claims.
    _GRANTEE_QUERY = """
        MATCH (c:Claim {claim_type: 'land_grant'})-[:SUBJECT_OF_CLAIM]->(p:Person)
        WHERE (p.deleted_at IS NULL OR p.deleted_at = '')
        RETURN DISTINCT
            p.entity_id       AS entity_id,
            p.name            AS name,
            p.normalized_form AS normalized_form
    """

    # Paragraphs are not linked to Documents via HAS_PARAGRAPH in this graph;
    # they are standalone nodes whose paragraph_id embeds the parent doc_id as
    # the prefix before ':paragraph:'.  We recover the relationship by splitting.
    _PARAGRAPH_QUERY = """
        MATCH (par:Paragraph)
        WITH par, split(par.paragraph_id, ':paragraph:')[0] AS doc_id
        MATCH (d:Document {doc_id: doc_id})
        WHERE d.institution_id = 'spokane_newspaper'
          AND (d.deleted_at IS NULL OR d.deleted_at = '')
        RETURN
            par.paragraph_id  AS paragraph_id,
            par.clean_text    AS clean_text,
            par.raw_ocr_text  AS raw_ocr_text
    """

    _MERGE_MENTION = """
        MERGE (m:Mention {mention_id: $mention_id})
        SET m += $props
    """

    _MERGE_CONTAINS = """
        MATCH (par:Paragraph {paragraph_id: $paragraph_id})
        MATCH (m:Mention {mention_id: $mention_id})
        MERGE (par)-[:CONTAINS_MENTION]->(m)
    """

    _MERGE_REFERS = """
        MATCH (m:Mention {mention_id: $mention_id})
        MATCH (e:Person {entity_id: $entity_id})
        MERGE (m)-[r:REFERS_TO]->(e)
        SET r.match_score = $match_score, r.run_id = $run_id
    """

    _MERGE_POSSIBLY = """
        MATCH (m:Mention {mention_id: $mention_id})
        MATCH (e:Person {entity_id: $entity_id})
        MERGE (m)-[r:POSSIBLY_REFERS_TO]->(e)
        SET r.match_score = $match_score, r.run_id = $run_id
    """

    def __init__(
        self,
        driver: Any,
        resources_dir: Path | None = None,
        review_store: str | None = None,
        min_ocr_prob: float = 0.08,
    ) -> None:
        self._driver = driver
        self._resources_dir = resources_dir
        self._review_store = review_store
        self._min_ocr_prob = min_ocr_prob

    def _load_grantees(self) -> list[EntityRecord]:
        with self._driver.session() as session:
            result = session.run(self._GRANTEE_QUERY)
            records = []
            for row in result:
                records.append(EntityRecord(
                    entity_id=row["entity_id"],
                    entity_type="Person",
                    name=row["name"] or "",
                    normalized_form=row["normalized_form"] or normalize_name(row["name"] or ""),
                ))
        _log.info("Loaded %d patent grantees as seed set", len(records))
        return records

    def _iter_paragraphs(self) -> Iterator[dict[str, str | None]]:
        with self._driver.session() as session:
            result = session.run(self._PARAGRAPH_QUERY)
            for row in result:
                yield {
                    "paragraph_id": row["paragraph_id"],
                    "clean_text": row["clean_text"],
                    "raw_ocr_text": row["raw_ocr_text"],
                }

    def _scan_paragraph(
        self,
        para: dict[str, str | None],
        expanded_records: list[EntityRecord],
    ) -> list[_Match]:
        """Return resolved matches for one paragraph, trying clean then raw text."""
        matches: list[_Match] = []

        clean = para["clean_text"] or ""
        raw = para["raw_ocr_text"] or ""

        if clean and not _is_ocr_corrupted(clean):
            candidates = _extract_candidates(clean)
            matches = _score_candidates(candidates, expanded_records, ocr_suspect=False)

        if not matches and raw and raw != clean:
            # Secondary pass: collapse spaced-char OCR in the raw text, then scan
            collapsed_raw = _collapse_spaced_chars(raw)
            if collapsed_raw != raw and not _is_ocr_corrupted(collapsed_raw):
                candidates = _extract_candidates(collapsed_raw)
                matches = _score_candidates(candidates, expanded_records, ocr_suspect=True)

        return matches

    def _write_batch(
        self,
        session: Any,
        batch: list[tuple[MentionRecord, EntityResolutionRecord]],
        run_id: str,
    ) -> None:
        for mention, resolution in batch:
            props = mention.node_props()
            session.run(self._MERGE_MENTION, mention_id=mention.mention_id, props=props)
            session.run(
                self._MERGE_CONTAINS,
                paragraph_id=mention.paragraph_id,
                mention_id=mention.mention_id,
            )
            cypher = self._MERGE_REFERS if resolution.relation_type == "REFERS_TO" else self._MERGE_POSSIBLY
            session.run(
                cypher,
                mention_id=mention.mention_id,
                entity_id=resolution.entity_id,
                match_score=resolution.match_score,
                run_id=run_id,
            )

    def run(
        self,
        dry_run: bool = False,
        output_report: Path | None = None,
        batch_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Execute the enrichment pass.

        Returns a list of match dicts (suitable for dry-run inspection or
        writing to a CSV report via *output_report*).
        """
        from scripts.generate_ocr_variants import OCRConfusionMatrix
        from gemynd.shared.resource_loader import load_ocr_correction_map

        known_errors: dict[str, str] = {}
        if self._resources_dir:
            try:
                known_errors = load_ocr_correction_map(self._resources_dir)
            except Exception:
                _log.warning("Could not load ocr_corrections from %s", self._resources_dir)

        matrix = OCRConfusionMatrix.from_sources(
            known_errors=known_errors,
            review_store_path=self._review_store,
        )

        grantees = self._load_grantees()
        if not grantees:
            _log.warning("No patent grantees found — nothing to enrich")
            return []

        expanded_records = _build_expanded_records(grantees, matrix, self._min_ocr_prob)
        _log.info(
            "Expanded to %d name forms (including OCR variants and abbreviations)",
            len(expanded_records),
        )

        run_id = make_run_id("cross_corpus_ner")
        all_report_rows: list[dict[str, Any]] = []
        pending_batch: list[tuple[MentionRecord, EntityResolutionRecord]] = []
        para_count = 0

        def _flush(session: Any) -> None:
            if pending_batch and not dry_run:
                self._write_batch(session, pending_batch, run_id)
            pending_batch.clear()

        with self._driver.session() as write_session:
            for para in self._iter_paragraphs():
                para_count += 1
                if para_count % 500 == 0:
                    _log.info("Scanned %d paragraphs, %d matches so far", para_count, len(all_report_rows))

                paragraph_id: str = para["paragraph_id"]
                matches = self._scan_paragraph(para, expanded_records)

                for match in matches:
                    mention = MentionRecord(
                        mention_id=make_mention_id(
                            run_id,
                            paragraph_id,
                            match.start,
                            match.end,
                            match.surface_form,
                        ),
                        run_id=run_id,
                        paragraph_id=paragraph_id,
                        surface_form=match.surface_form,
                        normalized_form=match.normalized_form,
                        start_offset=match.start,
                        end_offset=match.end,
                        detection_confidence=match.score,
                        ocr_suspect=match.ocr_suspect,
                    )
                    resolution = EntityResolutionRecord(
                        mention_id=mention.mention_id,
                        entity_id=match.entity_id,
                        relation_type=match.relation_type,
                        match_score=match.score,
                    )
                    pending_batch.append((mention, resolution))
                    all_report_rows.append({
                        "paragraph_id": paragraph_id,
                        "surface_form": match.surface_form,
                        "collapsed_form": match.collapsed_form,
                        "entity_id": match.entity_id,
                        "entity_name": match.entity_name,
                        "score": match.score,
                        "relation_type": match.relation_type,
                        "ocr_suspect": match.ocr_suspect,
                    })

                    if len(pending_batch) >= batch_size:
                        _flush(write_session)

            _flush(write_session)

        _log.info(
            "Enrichment pass complete: %d paragraphs scanned, %d mentions resolved%s",
            para_count,
            len(all_report_rows),
            " (dry-run, nothing written)" if dry_run else "",
        )

        if output_report:
            _write_csv_report(output_report, all_report_rows)
            _log.info("Report written to %s", output_report)

        return all_report_rows


def _write_csv_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "paragraph_id", "surface_form", "collapsed_form",
        "entity_id", "entity_name", "score", "relation_type", "ocr_suspect",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
