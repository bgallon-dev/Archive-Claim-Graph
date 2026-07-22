from __future__ import annotations

import difflib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

from gemynd.core.ids import make_entity_id
from gemynd.core.models import EntityRecord, EntityResolutionRecord, MentionRecord
from gemynd.shared.resource_loader import (
    TURNBULL_RESOURCES_DIR,
    load_claim_relation_compatibility,
    load_discovered_entity_rows,
    load_seed_entity_rows,
)

if TYPE_CHECKING:
    from gemynd.ingest.extractors.mention_extractor import ResolutionContext
    from gemynd.core.domain_config import CompositeDomainConfig, DomainConfig


RESOLUTION_CONTRACT: dict[str, str] = {
    "REFERS_TO": (
        "match_score >= 0.85 and uniqueness_gap >= 0.05. "
        "Algorithmically unambiguous match. "
        "Treat as confirmed unless an EntityResolutionConfirmationRecord with relation_type='REFUTED_BY' overrides it."
    ),
    "POSSIBLY_REFERS_TO": (
        "match_score 0.50-0.84, or >= 0.85 without uniqueness gap. "
        "Plausible but ambiguous. Requires human review. "
        "Can be promoted to CONFIRMED_AS or explicitly rejected via EntityResolutionConfirmationRecord."
    ),
    "CONFIRMED_AS": (
        "Human-confirmed resolution. Overrides algorithmic REFERS_TO or POSSIBLY_REFERS_TO. "
        "Set via EntityResolutionConfirmationRecord with confirmed_by and confirmed_at."
    ),
    "REFUTED_BY": (
        "Human-rejected resolution. Overrides algorithmic REFERS_TO for this mention-entity pair. "
        "Set via EntityResolutionConfirmationRecord."
    ),
}


@dataclass(slots=True)
class ResolutionPolicy:
    refers_to_threshold: float = 0.85
    maybe_threshold: float = 0.65
    uniqueness_gap: float = 0.05


class EntityResolver(Protocol):
    def resolve(
        self,
        mentions: list[MentionRecord],
        contexts: dict[str, ResolutionContext] | None = None,
        document_entity_counts: Counter[str] | None = None,
    ) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        ...


def default_seed_entities(resources_dir: Path | None = None) -> list[EntityRecord]:
    """Build EntityRecord list from seed_entities.csv.

    A domain MUST be named: passing ``None`` raises rather than silently
    loading the bundled Turnbull seeds. Seeds are the single most
    domain-defining artifact, so a silent default here is the most
    fundamental leak. Callers that genuinely want Turnbull pass
    :data:`gemynd.shared.resource_loader.TURNBULL_RESOURCES_DIR` explicitly.

    The CSV uses a long format: one row per (entity, property) pair.
    Rows sharing the same (entity_type, name) are merged into one EntityRecord.
    """
    if resources_dir is None:
        raise ValueError(
            "default_seed_entities requires an explicit resources_dir; the silent "
            "Turnbull default has been removed. Pass TURNBULL_RESOURCES_DIR for the "
            "Turnbull corpus, or the domain's resources directory."
        )
    rows = load_seed_entity_rows(resources_dir)

    # Optional rule-mined vocabulary, unioned in after the curated seeds so the
    # curated ones win any collision. See ingest/extractors/entity_discovery.py:
    # a hand-curated gazetteer cannot cover an open corpus (75 seeds against
    # 550k newspaper pages left 96% of mentions resolving to nothing), so the
    # discovery rules mine one. Absent file = no-op, which is every other domain.
    rows = rows + load_discovered_entity_rows(resources_dir)

    # Preserve insertion order; group properties by (entity_type, name).
    props_by_key: dict[tuple[str, str], dict[str, object]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row["entity_type"], row["name"])
        if key not in props_by_key:
            props_by_key[key] = {}
            order.append(key)
        if row["prop_key"] and row["prop_value"]:
            props_by_key[key][row["prop_key"]] = row["prop_value"]

    entities: list[EntityRecord] = []
    seen_forms: set[str] = set()
    for entity_type, name in order:
        normalized = normalize_name(name)
        # One canonical entity per surface form. Two entities sharing a form
        # are indistinguishable to a string resolver, so they would split the
        # form's mentions arbitrarily between them.
        if normalized in seen_forms:
            continue
        seen_forms.add(normalized)
        entities.append(
            EntityRecord(
                entity_id=make_entity_id(entity_type, normalized),
                entity_type=entity_type,
                name=name,
                normalized_form=normalized,
                properties=props_by_key[(entity_type, name)],
            )
        )
    return entities


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s\-]", "", name.strip().lower())
    return re.sub(r"\s+", " ", normalized)



class DictionaryFuzzyResolver:
    # Boost magnitudes indexed by position in the preferred_entity_types list.
    _POSITION_BOOSTS: tuple[float, ...] = (0.05, 0.03, 0.01)
    # Penalty when the entity type is absent from a non-empty preferred list.
    _ABSENT_PENALTY: float = -0.02

    def __init__(
        self,
        seed_entities: Iterable[EntityRecord] | None = None,
        policy: ResolutionPolicy | None = None,
        supplementary_candidates: list[EntityRecord] | None = None,
        resources_dir: Path | None = None,
        config: "DomainConfig | CompositeDomainConfig | None" = None,
    ) -> None:
        self._policy = policy or ResolutionPolicy()
        if config is not None and seed_entities is None:
            self._seed_entities = list(config.seed_entities)
        elif seed_entities is not None:
            self._seed_entities = list(seed_entities)
        else:
            # No vocabulary supplied. Fall back to the Turnbull corpus *visibly*
            # (the production retrieval layer threads a DomainConfig; this branch
            # serves single-corpus/test construction). Never a silent None.
            self._seed_entities = default_seed_entities(TURNBULL_RESOURCES_DIR)
        if supplementary_candidates:
            # Seed vocabulary wins: skip graph candidates whose normalized_form
            # appears in the seed list, regardless of entity_type. Keying on
            # (type, form) let a graph-resident Organization shadow a seeded
            # Place sharing the same form (e.g. WASHINGTON STATE Org vs.
            # Washington State Place), fragmenting the canonical entity.
            seed_forms = {e.normalized_form for e in self._seed_entities}
            extras = [
                e for e in supplementary_candidates
                if e.normalized_form not in seed_forms
            ]
            self._seed_entities = self._seed_entities + extras

        # Load preferred_entity_types from claim_relation_compatibility.yaml.
        # This drives the YAML-based claim-type-conditioned score adjustments.
        if config is not None:
            self._preferred_entity_types: dict[str, list[str]] = config.preferred_entity_types
        else:
            compat_data = load_claim_relation_compatibility(resources_dir)
            self._preferred_entity_types = compat_data.get("preferred_entity_types", {})

        # Opt-in type filter (see DomainConfig.resolution_type_filter). Without
        # it nothing stops a mention typed Person resolving to a Place: "F. J.
        # Cheney" resolved to the town of Cheney, and "Condon Globe" to Condon.
        self._type_filter = bool(config is not None and config.resolution_type_filter)
        self._type_rules = list(config.entity_type_rules) if config is not None else []
        # Words the domain's discovery rules declare are not names. Excluding
        # them when *minting* entities was only half the job — they still occur
        # as mentions, and an unblocked "american"/"monday"/"price" attaches to
        # whatever resembles it (Americans, Mondays, Prince).
        self._non_entity_terms: frozenset[str] = (
            config.non_entity_terms if config is not None else frozenset()
        )
        self._block_index = _build_block_index(self._seed_entities)

    def resolve(
        self,
        mentions: list[MentionRecord],
        contexts: dict[str, ResolutionContext] | None = None,
        document_entity_counts: Counter[str] | None = None,
    ) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        """Resolve mentions to entities using a two-pass algorithm when contexts are provided.

        When *contexts* is None, the method runs a single pass identical to the
        original implementation (backward-compatible).

        **Two-pass algorithm** (active when contexts is not None):

        Pass 1 — high-confidence mentions only (score >= refers_to_threshold with
        uniqueness gap).  Builds a per-paragraph co-mention type counter used in
        Pass 2.  Populates context.resolved_entity_types as an output side-effect.

        Pass 2 — remaining mentions receive a +0.03 co-mention consistency boost
        for candidates whose entity_type matches the most-frequent type resolved
        in the same paragraph during Pass 1.

        The *document_entity_counts* Counter is threaded into _candidate_scores()
        (A3 frequency prior) in both passes.
        """
        if contexts is None:
            # Single-pass fallback — identical to original implementation.
            return self._resolve_single_pass(mentions, document_entity_counts)

        from collections import defaultdict

        entity_map: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []
        resolved_mention_ids: set[str] = set()
        # co_mention_type_counts[paragraph_id][entity_type] = count of REFERS_TO
        co_mention_type_counts: dict[str, Counter[str]] = defaultdict(Counter)

        # ── Pass 1: high-confidence resolutions ───────────────────────────────
        for mention in mentions:
            context = contexts.get(mention.paragraph_id)
            candidates = self._candidate_scores(mention, context, document_entity_counts)
            if not candidates:
                continue
            top_entity, top_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            unique = top_score - second_score >= self._policy.uniqueness_gap

            if top_score >= self._policy.refers_to_threshold and unique:
                entity_map[top_entity.entity_id] = top_entity
                resolutions.append(EntityResolutionRecord(
                    mention_id=mention.mention_id,
                    entity_id=top_entity.entity_id,
                    relation_type="REFERS_TO",
                    match_score=round(top_score, 4),
                ))
                resolved_mention_ids.add(mention.mention_id)
                co_mention_type_counts[mention.paragraph_id][top_entity.entity_type] += 1

        # Populate resolved_entity_types on each context as an output.
        for pid, type_counter in co_mention_type_counts.items():
            ctx = contexts.get(pid)
            if ctx is not None:
                ctx.resolved_entity_types = sorted(type_counter.elements())

        # ── Pass 2: remaining mentions with co-mention boost ──────────────────
        for mention in mentions:
            if mention.mention_id in resolved_mention_ids:
                continue
            context = contexts.get(mention.paragraph_id)
            candidates = self._candidate_scores(mention, context, document_entity_counts)
            if not candidates:
                continue

            # Apply co-mention consistency boost (+0.03) for the most-frequent
            # entity type in this paragraph's Pass-1 resolutions.
            para_counter = co_mention_type_counts.get(mention.paragraph_id)
            if para_counter:
                most_frequent_type = para_counter.most_common(1)[0][0]
                candidates = [
                    (entity, min(score + 0.03, 1.0) if entity.entity_type == most_frequent_type else score)
                    for entity, score in candidates
                ]
                candidates.sort(key=lambda row: row[1], reverse=True)

            top_entity, top_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            unique = top_score - second_score >= self._policy.uniqueness_gap

            if top_score >= self._policy.refers_to_threshold and unique:
                relation = "REFERS_TO"
            elif top_score >= self._policy.maybe_threshold:
                relation = "POSSIBLY_REFERS_TO"
            else:
                continue

            entity_map[top_entity.entity_id] = top_entity
            resolutions.append(EntityResolutionRecord(
                mention_id=mention.mention_id,
                entity_id=top_entity.entity_id,
                relation_type=relation,
                match_score=round(top_score, 4),
            ))

        return list(entity_map.values()), resolutions

    def _resolve_single_pass(
        self,
        mentions: list[MentionRecord],
        document_entity_counts: Counter[str] | None = None,
    ) -> tuple[list[EntityRecord], list[EntityResolutionRecord]]:
        """Original single-pass resolution — used when contexts=None."""
        entity_map: dict[str, EntityRecord] = {}
        resolutions: list[EntityResolutionRecord] = []

        for mention in mentions:
            candidates = self._candidate_scores(mention, None, document_entity_counts)
            if not candidates:
                continue
            top_entity, top_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            unique = top_score - second_score >= self._policy.uniqueness_gap

            if top_score >= self._policy.refers_to_threshold and unique:
                relation = "REFERS_TO"
            elif top_score >= self._policy.maybe_threshold:
                relation = "POSSIBLY_REFERS_TO"
            else:
                continue

            entity_map[top_entity.entity_id] = top_entity
            resolutions.append(EntityResolutionRecord(
                mention_id=mention.mention_id,
                entity_id=top_entity.entity_id,
                relation_type=relation,
                match_score=round(top_score, 4),
            ))

        return list(entity_map.values()), resolutions

    def _claim_type_adjustment(self, entity_type: str, claim_types: list[str]) -> float:
        """Return the net score adjustment driven by preferred_entity_types.

        For each claim type in *claim_types*, compute the boost/penalty for
        *entity_type* based on its position in preferred_entity_types[claim_type].
        Return the maximum positive adjustment found, or the minimum negative
        adjustment if no positive adjustments exist.  Return 0.0 when
        preferred_entity_types has no entry for any of the claim types.
        """
        if not claim_types:
            return 0.0
        adjustments: list[float] = []
        for ct in claim_types:
            preferred = self._preferred_entity_types.get(ct)
            if preferred is None:
                # No preference data for this claim type — neutral.
                continue
            if not preferred:
                # Explicitly empty list (e.g. unclassified_assertion) — neutral.
                continue
            try:
                pos = preferred.index(entity_type)
                adj = self._POSITION_BOOSTS[pos] if pos < len(self._POSITION_BOOSTS) else 0.0
            except ValueError:
                adj = self._ABSENT_PENALTY
            adjustments.append(adj)

        if not adjustments:
            return 0.0
        if any(a > 0 for a in adjustments):
            return max(adjustments)
        return min(adjustments)

    def _blocked_candidates(self, normalized: str) -> list[EntityRecord]:
        """Entities that could possibly match *normalized* (see _build_block_index)."""
        exact_index, fuzzy_index = self._block_index
        seen: dict[str, EntityRecord] = {}
        for token in set(normalized.split()):
            if not token:
                continue
            for entity in exact_index.get(token, ()):
                seen[entity.entity_id] = entity
            singular = _singular(token)
            if singular != token:
                for entity in exact_index.get(singular, ()):
                    seen[entity.entity_id] = entity
            if len(token) < _MIN_FUZZY_TOKEN_LEN:
                continue  # zero edit budget — an exact match is the only match
            first = token[0]
            for length in range(
                max(len(token) - _MAX_TOKEN_LEN_DELTA, _MIN_FUZZY_TOKEN_LEN),
                len(token) + _MAX_TOKEN_LEN_DELTA + 1,
            ):
                for entity in fuzzy_index.get((first, length), ()):
                    seen[entity.entity_id] = entity
        return list(seen.values())

    def _candidate_scores(
        self,
        mention: MentionRecord,
        context: ResolutionContext | None = None,
        document_entity_counts: Counter[str] | None = None,
    ) -> list[tuple[EntityRecord, float]]:
        """Score all seed entities against *mention* and return sorted candidates.

        Score calculation order:
        1. Base string similarity (similarity_score).
        2. [A1] YAML-driven claim-type adjustment (boost or small penalty).
           Floor at maybe_threshold to avoid hard-filtering borderline candidates.
        3. [A3] Document-level entity frequency prior (log-scaled boost, capped at 0.06).
        Candidates below maybe_threshold after all adjustments are excluded.
        """
        if mention.normalized_form in self._non_entity_terms:
            return []

        # Type inferred from the mention's own surface form. Seeds are excluded
        # from the inference so it reflects the rules rather than the answer we
        # are about to look up. None means "no opinion" and filters nothing.
        hint: str | None = None
        if self._type_filter and self._type_rules:
            from gemynd.core.entity_type_inference import infer_entity_type

            hint = infer_entity_type(mention.surface_form, rules=self._type_rules)

        scored: list[tuple[EntityRecord, float]] = []
        for entity in self._blocked_candidates(mention.normalized_form):
            if hint is not None and entity.entity_type != hint:
                continue
            score = similarity_score(mention.normalized_form, entity.normalized_form)
            if score < self._policy.maybe_threshold:
                continue
            # A character-similar candidate with no shared word is a coincidence,
            # not a reference. See token_anchor_ok — this gate is what takes the
            # spurious resolution rate down from 44.3%.
            if not token_anchor_ok(mention.normalized_form, entity.normalized_form):
                continue

            # A1: YAML-driven claim-type-conditioned adjustment.
            if context is not None and context.claim_types:
                net = self._claim_type_adjustment(entity.entity_type, context.claim_types)
                score = min(score + net, 1.0)
                score = max(score, self._policy.maybe_threshold)

            # A3: Document-level entity frequency prior.
            if document_entity_counts and entity.entity_id in document_entity_counts:
                n = document_entity_counts[entity.entity_id]
                freq_boost = min(0.02 * math.log1p(n), 0.06)
                score = min(score + freq_boost, 1.0)

            scored.append((entity, score))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored


# A contained fragment must be at least this fraction of the longer string for
# the containment boost to apply (see similarity_score).
_CONTAINMENT_MIN_RATIO: float = 0.5

# Tokens that never identify an entity on their own. A shared token drawn only
# from this set is not evidence: "wasco county"/"adams county" and "prosser
# falls"/"spokane falls" share exactly one token and denote different places.
_GENERIC_TOKENS: frozenset[str] = frozenset({
    "county", "city", "town", "township", "precinct", "state", "territory",
    "river", "creek", "lake", "valley", "falls", "springs", "mountain",
    "canyon", "ridge", "island", "harbor", "prairie", "butte", "basin",
    "company", "co", "co.", "corp", "corporation", "inc", "ltd", "railway",
    "railroad", "rr", "association", "assn", "society", "club", "lodge",
    "department", "board", "commission", "bureau", "agency", "committee",
    "council", "league", "union", "court", "office", "district",
    "street", "avenue", "road", "of", "the", "and",
})

# Qualifiers, which are generic ONLY when standing alone. A bare "north"
# resolved to the mined "North Yak" 60 times and "eastern" to "Eastern Oregon"
# 18 times — but the same words carry real names ("Great Northern Railway",
# "North Yakima"), so they disqualify a match only when they *are* the mention.
_QUALIFIER_TOKENS: frozenset[str] = frozenset({
    "north", "south", "east", "west", "northern", "southern", "eastern",
    "western", "upper", "lower", "big", "little", "great", "grand", "old", "new",
})

def _edit_budget(length: int) -> int:
    """How many character edits two tokens of this length may differ by.

    A similarity *ratio* cannot do this job: "tumbull"/"turnbull" (the rn->m
    OCR confusion) and "oregonian"/"oregon" (a different word) both score 0.80
    under difflib. An absolute edit budget separates them, because OCR damage is
    a fixed number of characters while a different word diverges proportionally.

    Short tokens get no budget at all — at four characters a single edit turns
    "work" into "york", which is how "work" resolved to New York 18 times.
    """
    if length < 6:
        return 0
    if length < 8:
        return 1
    return 2


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _token_similar(left: str, right: str) -> bool:
    """Same word, allowing for OCR damage."""
    if left == right:
        return True
    # Inflection is not damage, and is not covered by the edit budget: "coots"
    # vs "coot" is one edit on a five-character token, where the budget is zero.
    # The stem must still be substantial — stripping an "s" off a three-letter
    # token matches almost anything, which collapsed the month "May" onto the
    # surname "Mays" 89 times.
    stem = _singular(left)
    if len(stem) >= 4 and stem == _singular(right):
        return True
    budget = _edit_budget(max(len(left), len(right)))
    if budget == 0 or abs(len(left) - len(right)) > budget:
        return False
    # Distinct words most often diverge at the front ("southern"/"northern" is
    # only two edits apart, the same distance as the tumbull/turnbull OCR
    # confusion). A damaged leading character is comparatively rare, so
    # requiring it to survive costs little and separates the two cases.
    if left[0] != right[0]:
        return False
    # A strict prefix relation is derivation, not damage: "harris"/"harrison"
    # and "stevens"/"stevenson" are different names that sit inside the same
    # 2-edit budget as the tumbull/turnbull OCR confusion. Truncation damage
    # loses a character or two, so allow only a difference of one.
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if longer.startswith(shorter) and len(longer) - len(shorter) > 1:
        return False

    # Bounded Levenshtein: bail as soon as every cell in a row exceeds budget.
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, start=1):
        current = [i]
        for j, rch in enumerate(right, start=1):
            current.append(
                previous[j - 1] if lch == rch
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        if min(current) > budget:
            return False
        previous = current
    return previous[-1] <= budget


# Widest length difference _token_similar can tolerate (see _edit_budget).
_MAX_TOKEN_LEN_DELTA = 2

# Shortest token that can participate in a fuzzy match at all. _edit_budget is
# 0 below 6 characters, so a 5-character token can only ever match a 6 (budget
# 1 comes from the *longer* side); anything shorter must match exactly. Keeping
# short tokens out of the fuzzy index is free recall-wise and removes the bulk
# of the candidates, since short tokens are the common ones.
_MIN_FUZZY_TOKEN_LEN = 5


def _build_block_index(
    entities: list[EntityRecord],
) -> tuple[dict[str, list[EntityRecord]], dict[tuple[str, int], list[EntityRecord]]]:
    """Index entities so a mention only compares against plausible candidates.

    Comparing every mention against every entity is fine for a curated 75-entry
    gazetteer and hopeless for a mined one: 45k mentions against 48k entities
    took 7.8s per document, i.e. 26 minutes per 200-document checkpoint.

    Two indexes, because tokens match two different ways:

    * ``exact`` — token, plus its singular form so "coots"/"coot" collide.
    * ``fuzzy`` — (first character, length), holding only tokens long enough to
      have a non-zero edit budget.

    Exact, not approximate. ``token_anchor_ok`` only succeeds when some token of
    the shorter name matches one of the longer's, and ``_token_similar`` only
    matches tokens that are equal, share a singular, or share a first character
    while differing in length by at most the budget. Every one of those routes
    has an index, so nothing resolvable is skipped.
    """
    exact: dict[str, list[EntityRecord]] = defaultdict(list)
    fuzzy: dict[tuple[str, int], list[EntityRecord]] = defaultdict(list)
    for entity in entities:
        for token in set(entity.normalized_form.split()):
            if not token:
                continue
            exact[token].append(entity)
            singular = _singular(token)
            if singular != token:
                exact[singular].append(entity)
            if len(token) >= _MIN_FUZZY_TOKEN_LEN:
                fuzzy[(token[0], len(token))].append(entity)
    return exact, fuzzy


def _token_related(left: str, right: str) -> bool:
    """Looser than :func:`_token_similar`: also allows a shared stem.

    Used only for the trailing token, where derivation is normal and harmless
    ("live trap" / "live trapping"). It is deliberately NOT used for the leading
    token, because that is where a shared stem means a different entity —
    "oregonian" would otherwise match "oregon".
    """
    if _token_similar(left, right):
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return len(shorter) >= 4 and longer.startswith(shorter)


def token_anchor_ok(left: str, right: str) -> bool:
    """Whether an inexact match between two normalized names has token evidence.

    ``similarity_score`` is a character metric, and on short strings character
    overlap is close to meaningless: ``difflib`` scores "house"/"palouse" at
    0.83 and "spain"/"spokane" at 0.67 despite no word in common. Measured on
    checkpoint 1, that produced a **44.3% spurious resolution rate** — 95% of
    every POSSIBLY_REFERS_TO edge was wrong ("range"->"Sprague",
    "phone"->"Spokane", "work"->"New York"), and mentions of 4-8 characters were
    wrong essentially 100% of the time.

    A real reference shares a *word*, not a run of characters. So require:

    1. at least one token of the shorter name to match a token of the longer
       one (fuzzily, so OCR variants survive), and
    2. at least one such matched token to be non-generic, and
    3. the shorter name's *leading* token to be among the matches — this is what
       separates "southern pacific"/"northern pacific railway" (shares only
       "pacific") from "northern pacific"/"northern pacific railway".

    Exact matches bypass the gate entirely.
    """
    if left == right:
        return True
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return False

    shorter, longer = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    matched = [t for t in shorter if any(_token_similar(t, o) for o in longer)]
    if not matched:
        return False
    uninformative = (
        _GENERIC_TOKENS if len(shorter) > 1 else _GENERIC_TOKENS | _QUALIFIER_TOKENS
    )
    if all(token in uninformative for token in matched):
        return False
    if shorter[0] not in matched:
        return False
    # A multi-token name that matches only on its leading token is a different
    # name that happens to share a first word. This is the given-name collision:
    # "william watson"/"william grant"/"william pritchard" all matched "William
    # Howard Taft", and "washington street" matched "Washington State".
    if len(shorter) >= 2 and not any(
        _token_related(shorter[-1], other) for other in longer
    ):
        return False
    return True


def similarity_score(left: str, right: str) -> float:
    if left == right:
        return 1.0
    ratio = difflib.SequenceMatcher(a=left, b=right).ratio()
    # Substring containment is only a meaningful signal when the shorter side is
    # a SUBSTANTIAL part of the longer one. A bare ">= 4 chars" guard is far too
    # weak: it boosts to exactly the 0.88 resolution threshold, so "fall" resolved
    # to "spokane falls", "west" to "western federation of miners", "chicago" to
    # "chicago milwaukee and st. paul railway" and "world" to "industrial workers
    # of the world". That single rule produced 45.3% spurious REFERS_TO edges
    # (242,464 of 535,624) on the newspaper graph. Require the shorter side to be
    # at least half the longer one as well.
    if left in right or right in left:
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if len(shorter) >= 4 and len(shorter) >= _CONTAINMENT_MIN_RATIO * len(longer):
            ratio = max(ratio, 0.88)
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
        ratio = max(ratio, overlap)
    return round(ratio, 4)
