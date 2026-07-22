"""Rule-mined entity discovery for open-domain corpora.

A curated gazetteer is the wrong tool for an open corpus: the newspaper graph
had 75 seed entities against 550k pages, so 96% of mentions resolved to nothing
and the seeds were region-mismatched besides. Enumerating names by hand does not
scale and does not transfer — every new paper, region and decade would need it
redone.

This module takes the other route: *contextual* rules that recognise an entity
by the frame it sits in rather than by its spelling, so the same rules mine a
gazetteer out of any corpus in the genre. See
``resources_newspaper/entity_discovery_rules.yaml`` for the rules themselves and
the measurements behind them.

Discovery runs as an offline pass over raw text and emits a seed-entity CSV. It
is deliberately NOT part of per-document extraction: the precision thresholds
are corpus-level (how many documents a form appears in), and keeping it separate
means the mined vocabulary can be reviewed before it ever reaches the graph.
"""
from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

_log = logging.getLogger(__name__)

# Below this many characters a "name" is noise regardless of the frame it sat in.
_MIN_NAME_CHARS = 4


def _canonical_case(name: str) -> str:
    """Title-case an all-caps capture.

    Datelines are typeset in caps ("PORTLAND, Or."), so without this the same
    place is minted as "PORTLAND" or "Portland" depending on which rule happened
    to see it first — a non-deterministic entity name for identical input.
    """
    return name.title() if name.isupper() else name


@dataclass(slots=True, frozen=True)
class DiscoveryRule:
    name: str
    target_type: str
    pattern: re.Pattern[str]
    name_template: str | None = None
    min_occurrences: int = 1
    min_documents: int = 1

    def render(self, captured: str) -> str:
        name = _canonical_case(captured)
        return self.name_template.format(name) if self.name_template else name


@dataclass(slots=True)
class MinedEntity:
    name: str
    entity_type: str
    occurrences: int = 0
    document_count: int = 0
    rules: set[str] = field(default_factory=set)
    # Index of the document counted most recently. Tracking document *count*
    # this way instead of keeping a set of indices matters at corpus scale:
    # a set per candidate over 550k pages is hundreds of megabytes of nothing.
    _last_doc: int = -1

    def observe(self, doc_index: int) -> None:
        self.occurrences += 1
        if doc_index != self._last_doc:
            self._last_doc = doc_index
            self.document_count += 1


def load_discovery_rules(path: Path) -> tuple[list[DiscoveryRule], frozenset[str]]:
    """Load rules + stopwords from a discovery YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules: list[DiscoveryRule] = []
    for entry in data.get("rules") or ():
        rules.append(
            DiscoveryRule(
                name=entry["name"],
                target_type=entry["target_type"],
                # Original-case matching: capitalisation is the signal here.
                pattern=re.compile(entry["pattern"]),
                name_template=entry.get("name_template"),
                min_occurrences=int(entry.get("min_occurrences", 1)),
                min_documents=int(entry.get("min_documents", 1)),
            )
        )
    stopwords = frozenset(str(w).lower() for w in (data.get("stopwords") or ()))
    return rules, stopwords


_CALENDAR_WORDS: frozenset[str] = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    # Holidays are always capitalised, so the capitalisation profile reads them
    # as name-like: "at Christmas" mined Christmas as a Place (74 edges).
    "christmas", "easter", "thanksgiving", "epiphany", "pentecost",
})


def _singular_form(token: str) -> str:
    """Mirror of the resolver's inflection rule, for stopword checking."""
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _acceptable(name: str, stopwords: frozenset[str]) -> bool:
    cleaned = name.strip()
    if len(cleaned) < _MIN_NAME_CHARS:
        return False
    lowered = cleaned.lower()
    if lowered in stopwords or lowered in _CALENDAR_WORDS:
        return False
    # ...and the plural of one. Checking only the literal form let "Mondays",
    # "Fridays" and "Americans" become entities, which then collected every
    # mention of the singular the stopword list was meant to exclude.
    singular = _singular_form(lowered)
    if singular in stopwords or singular in _CALENDAR_WORDS:
        return False
    tokens = lowered.split()
    if not tokens:
        return False
    # A multi-word form is rejected only if *every* word is a common word, so
    # "Hood River" survives while "Half Price" and "Live Stock" do not. This is
    # why the stopword list carries commercial vocabulary: page-level OCR is
    # full of advertisement copy, and "at Half Price" sits in the same
    # prepositional frame as "at Portland".
    if all(token in stopwords for token in tokens):
        return False
    # Dates read "July 1st" / "Monday evening" and the numeral OCRs into
    # something wordlike ("July Ist"), which then looks like a two-word name.
    # No place name begins with a month or a weekday.
    if tokens[0] in _CALENDAR_WORDS:
        return False
    return any(ch.isalpha() for ch in cleaned)


_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")

# A candidate token must be Title-case at least this often, among the times it
# appears as either Title-case or lowercase, to count as name-like.
_MIN_CAPITALIZATION_RATIO = 0.65

# A token must lead this many multi-word person captures to read as a forename.
#
# The honorific frame captures one or two words, so "Mr. John Smith" yields a
# full name but "Mr. John," yields the bare forename. Those bare forenames then
# become entities and act as magnets — *John*, *George* and *Sarah* each
# collected resolutions from unrelated people. The capitalisation profile cannot
# help, since forenames are always capitalised; but a forename LEADS other names
# ("John Smith", "John Gavin") while a surname rarely does. Counted only over
# Person rules: applied to places it would delete "Portland" the moment
# "Portland Tuesday" was captured twice.
_GIVEN_NAME_MIN = 2


def capitalization_profile(documents: Iterable[str]) -> dict[str, float]:
    """Per-word share of Title-case usage across the corpus.

    No stopword list can enumerate the common words that end up in entity
    frames — "of Them", "to Sell", "of Ladies" mined *Them*, *Sell* and *Ladies*
    as places (697, 108 and 135 edges). But the corpus already says which words
    are names: "Portland" is virtually always capitalised while "them" is
    virtually never. That signal is free, self-calibrating, and needs no list,
    so it transfers to any corpus.

    ALL-CAPS tokens are ignored rather than counted as evidence: headlines and
    advertisement copy are set in caps, and counting them would promote exactly
    the ad vocabulary this is meant to exclude.
    """
    title: dict[str, int] = defaultdict(int)
    lower: dict[str, int] = defaultdict(int)
    for text in documents:
        if not text:
            continue
        for match in _WORD_RE.finditer(text):
            token = match.group(0)
            if token.islower():
                lower[token] += 1
            elif token[0].isupper() and token[1:].islower():
                title[token.lower()] += 1
    profile: dict[str, float] = {}
    for word in set(title) | set(lower):
        seen = title[word] + lower[word]
        if seen:
            profile[word] = title[word] / seen
    return profile


def _name_like(name: str, profile: dict[str, float], stopwords: frozenset[str]) -> bool:
    """Whether every informative token of *name* reads as a name in this corpus."""
    for token in name.lower().split():
        if token in stopwords:
            continue
        ratio = profile.get(token)
        # Unseen in either case (rare: OCR fragments) — no evidence, so no pass.
        if ratio is None or ratio < _MIN_CAPITALIZATION_RATIO:
            return False
    return True


def low_capitalization_terms(
    profile: dict[str, float], min_sightings: dict[str, int], min_seen: int = 25
) -> list[str]:
    """Words the corpus shows are not names, for the resolver to refuse.

    The same signal that keeps common words out of the gazetteer also says which
    *mentions* should never acquire an entity. Without it, "white" resolved to
    the surname Whited, "season" to Seaton, "heater" to Hester and "furniture"
    to Furninure — a class no hand-written stopword list can enumerate, but one
    the corpus identifies for free.

    Requires ``min_seen`` sightings so a rare word with no real evidence is not
    condemned on one lowercase appearance.
    """
    return sorted(
        word
        for word, ratio in profile.items()
        if ratio < _MIN_CAPITALIZATION_RATIO and min_sightings.get(word, 0) >= min_seen
    )


def mine_entities(
    documents: Iterable[str],
    rules: list[DiscoveryRule],
    stopwords: frozenset[str],
    profile_out: dict[str, float] | None = None,
    sightings_out: dict[str, int] | None = None,
) -> list[MinedEntity]:
    """Mine candidate entities from raw document texts.

    A candidate is kept when it clears the thresholds of at least one rule that
    matched it. Thresholds are per-rule so a high-precision frame (a dateline)
    can admit a name on a single sighting while a noisy one (a prepositional
    frame) demands corroboration across documents.

    Single streaming pass: *documents* is consumed once and never materialised,
    because the real corpus is ~550k pages / 11GB. The capitalisation profile
    and the forename tally are corpus-level statistics, but both are pure
    filters, so they are accumulated during the pass and applied at the end
    rather than forcing a second read.
    """
    title_counts: dict[str, int] = defaultdict(int)
    lower_counts: dict[str, int] = defaultdict(int)
    leading: dict[str, int] = defaultdict(int)
    person_rules = {r.name for r in rules if r.target_type == "Person"}

    # (normalized name, type) -> MinedEntity, plus per-rule tallies for gating.
    found: dict[tuple[str, str], MinedEntity] = {}
    per_rule: dict[tuple[str, str, str], MinedEntity] = {}

    for doc_index, text in enumerate(documents):
        if not text:
            continue
        for match in _WORD_RE.finditer(text):
            token = match.group(0)
            if token.islower():
                lower_counts[token] += 1
            elif token[0].isupper() and token[1:].islower():
                title_counts[token.lower()] += 1

        for rule in rules:
            for match in rule.pattern.finditer(text):
                captured = (match.group(1) or "").strip()
                if not _acceptable(captured, stopwords):
                    continue
                name = rule.render(captured)
                if not _acceptable(name, stopwords):
                    continue
                key = (name.lower(), rule.target_type)

                if rule.name in person_rules:
                    tokens = key[0].split()
                    if len(tokens) >= 2:
                        leading[tokens[0]] += 1

                entry = found.get(key)
                if entry is None:
                    entry = found[key] = MinedEntity(name=name, entity_type=rule.target_type)
                entry.observe(doc_index)
                entry.rules.add(rule.name)

                rule_key = (name.lower(), rule.target_type, rule.name)
                rule_entry = per_rule.get(rule_key)
                if rule_entry is None:
                    rule_entry = per_rule[rule_key] = MinedEntity(
                        name=name, entity_type=rule.target_type
                    )
                rule_entry.observe(doc_index)

    profile: dict[str, float] = {}
    sightings: dict[str, int] = {}
    for word in set(title_counts) | set(lower_counts):
        seen = title_counts[word] + lower_counts[word]
        if seen:
            profile[word] = title_counts[word] / seen
            sightings[word] = seen
    given_names = frozenset(t for t, n in leading.items() if n >= _GIVEN_NAME_MIN)
    # Exposed for callers that also want the corpus statistics (see
    # low_capitalization_terms); computing them needs a full pass, and the
    # corpus is too large to read twice just to recover them.
    if profile_out is not None:
        profile_out.update(profile)
    if sightings_out is not None:
        sightings_out.update(sightings)

    by_name = {rule.name: rule for rule in rules}
    kept: list[MinedEntity] = []
    for key, entity in found.items():
        # A bare forename is not an entity, however well-evidenced.
        if entity.entity_type == "Person" and key[0] in given_names:
            continue
        # Reads as a name in this corpus? (Applied here rather than during the
        # pass because the profile is only complete once the pass has finished.)
        if not _name_like(entity.name, profile, stopwords):
            continue
        # Keep if ANY rule that saw this form clears its own thresholds.
        for rule_name in entity.rules:
            rule = by_name[rule_name]
            tally = per_rule[(key[0], key[1], rule_name)]
            if (
                tally.occurrences >= rule.min_occurrences
                and tally.document_count >= rule.min_documents
            ):
                kept.append(entity)
                break

    kept.sort(key=lambda e: (-e.occurrences, e.name))
    return kept



# A variant is folded into a canonical form only if the canonical is this many
# times better evidenced — otherwise two genuinely distinct names of similar
# frequency ("Benton"/"Renton") could swallow one another.
_VARIANT_EVIDENCE_RATIO = 3.0


def _same_shape(left: str, right: str) -> bool:
    """Whether two names differ only as spellings of the same name.

    An OCR variant has the same word count and nearly the same length —
    "Prineviile"/"Prineville", "Dalle"/"Dalles". A *containment* relation is a
    different thing entirely: "William County" and "William" are not two
    spellings of one name, and merging them destroyed the specific entity while
    crediting the general one with the county rule's evidence. That is how
    "The Dalles" disappeared into "Dalle" and how forenames acquired 3,000
    sightings as Places.
    """
    left_tokens = left.split()
    right_tokens = right.split()
    if len(left_tokens) != len(right_tokens):
        return False
    return abs(len(left) - len(right)) <= 2


def merge_ocr_variants(entities: list[MinedEntity]) -> list[MinedEntity]:
    """Fold OCR-damaged spellings into the well-evidenced form they mangle.

    Mining a 1.6-billion-character corpus produces both "Prineville" and
    "Prineviile", "Silverton" and "Sllvcrton", "Colville" and "Coiville". Left
    alone each becomes its own entity, so one town's mentions split across
    several nodes — the fragmentation the entity layer exists to prevent.

    Uses the resolver's own matching rule, so what merges here is exactly what
    the resolver would have treated as the same token anyway.
    """
    from gemynd.core.resolver import token_anchor_ok

    ordered = sorted(entities, key=lambda e: (-e.occurrences, e.name))
    # Bucket by type and leading character so this stays near-linear; comparing
    # 72k entities pairwise would be 5 billion checks.
    buckets: dict[tuple[str, str], list[MinedEntity]] = defaultdict(list)
    kept: list[MinedEntity] = []

    for entity in ordered:
        form = entity.name.lower()
        if not form:
            continue
        key = (entity.entity_type, form[0])
        canonical = None
        for other in buckets[key]:
            if other.occurrences < entity.occurrences * _VARIANT_EVIDENCE_RATIO:
                continue
            if not _same_shape(form, other.name.lower()):
                continue
            if token_anchor_ok(form, other.name.lower()):
                canonical = other
                break
        if canonical is not None:
            canonical.occurrences += entity.occurrences
            canonical.document_count += entity.document_count
            canonical.rules |= entity.rules
            continue
        buckets[key].append(entity)
        kept.append(entity)

    kept.sort(key=lambda e: (-e.occurrences, e.name))
    return kept


def resolve_type_conflicts(entities: list[MinedEntity]) -> list[MinedEntity]:
    """Keep one type per surface form — the best-evidenced one.

    "Lincoln" can be captured as both a Place and a Person. Emitting both makes
    the resolver pick between two candidates that are equally good on string
    similarity, which is exactly the ambiguity that produces junk edges.
    """
    best: dict[str, MinedEntity] = {}
    for entity in entities:
        key = entity.name.lower()
        current = best.get(key)
        if current is None or (entity.occurrences, entity.document_count) > (
            current.occurrences,
            current.document_count,
        ):
            best[key] = entity
    return sorted(best.values(), key=lambda e: (-e.occurrences, e.name))


def write_seed_csv(entities: list[MinedEntity], path: Path) -> int:
    """Write mined entities in the seed_entities.csv long format.

    Provenance (which rules fired, how much evidence) is written as entity
    properties so a reviewer can tell a dateline-confirmed place from one that
    scraped past the prepositional threshold.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entity_type", "name", "prop_key", "prop_value"])
        for entity in entities:
            writer.writerow([entity.entity_type, entity.name, "discovered_by", ",".join(sorted(entity.rules))])
            writer.writerow([entity.entity_type, entity.name, "occurrences", str(entity.occurrences)])
            writer.writerow([entity.entity_type, entity.name, "document_count", str(entity.document_count)])
    _log.info("wrote %d discovered entities to %s", len(entities), path)
    return len(entities)
