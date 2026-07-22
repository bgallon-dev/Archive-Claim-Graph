from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from gemynd.core.claim_contract import (
    ALLOWED_CLAIM_TYPES,
    CLAIM_LOCATION_RELATION,
    UNCLASSIFIED_TYPE,
    claim_relation_priority,
    validate_claim_link_relation,
)
from gemynd.core.claim_validator import is_valid_claim_sentence
from gemynd.core.entity_type_inference import infer_entity_type
from gemynd.core.resolver import default_seed_entities
from gemynd.shared.resource_loader import (
    TURNBULL_RESOURCES_DIR,
    load_claim_type_patterns,
    load_claim_role_policy,
    load_entity_type_rules,
)
from gemynd.shared.text_utils import normalize_for_match
from typing import TYPE_CHECKING

from ._draft_audit import tag_hybrid_trace
from ._resources import (
    default_claim_role_policy,
    default_claim_type_patterns,
    seed_terms_for,
)

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig


def _collect_seed_terms(seed_types: tuple[str, ...], seed_entities: list) -> list[str]:
    """Names of seed entities whose type is in *seed_types*, longest-first."""
    labels = set(seed_types)
    return sorted(
        {entity.name for entity in seed_entities if entity.entity_type in labels},
        key=len,
        reverse=True,
    )


def _compile_occurrence_re(verbs: tuple[str, ...]) -> "re.Pattern[str] | None":
    """Occurrence-context regex from a domain's occurrence verbs (or None)."""
    if not verbs:
        return None
    return re.compile(
        r"\b(" + "|".join(verbs) + r")"
        r"\s+(?:into|in|at|near|on|within|around|from)\s*$",
        re.IGNORECASE,
    )


def _compile_subject_re(verbs: tuple[str, ...]) -> "re.Pattern[str] | None":
    """Subject-trigger regex from a domain's subject verbs (or None)."""
    if not verbs:
        return None
    return re.compile(r"\b(" + "|".join(verbs) + r")\b", re.IGNORECASE)


@dataclass(slots=True)
class ClaimLinkDraft:
    surface_form: str
    normalized_form: str
    relation_type: str
    start_offset: int | None = None
    end_offset: int | None = None
    entity_type_hint: str | None = None


@dataclass(slots=True)
class ClaimDraft:
    claim_type: str
    source_sentence: str
    normalized_sentence: str
    epistemic_status: str
    extraction_confidence: float
    evidence_start: int | None = None
    evidence_end: int | None = None
    claim_date: str | None = None
    notes: str = ""
    claim_links: list[ClaimLinkDraft] = field(default_factory=list)
    extraction_source: str = "rules"
    decision_trace: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    fallback_used: bool = False

    @property
    def certainty(self) -> str:
        return self.epistemic_status

    @certainty.setter
    def certainty(self, value: str) -> None:
        self.epistemic_status = value


class ClaimExtractor(Protocol):
    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        ...


_ABBREVS = {
    "mr", "mrs", "ms", "dr", "gov", "col", "lt", "sgt", "st", "jr", "sr",
    "mgr", "dept", "approx", "oct", "nov", "dec", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sept", "sep", "no", "vs", "etc", "al",
    "ibid", "op", "cf", "ca", "vol", "pp", "fig",
}
_ABBREV_LINE_END_RE = re.compile(r"\b([A-Za-z]{1,6})\.$", re.IGNORECASE)
_CONTINUATION_CHARS = frozenset(",;")


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    text = text.strip()
    if not text:
        return []

    # Step 1: join OCR line-break continuations
    lines = text.split("\n")
    buf = ""
    merged: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        if buf:
            m = _ABBREV_LINE_END_RE.search(buf)
            is_abbrev = bool(m and m.group(1).lower() in _ABBREVS)
            is_cont = buf[-1] in _CONTINUATION_CHARS or buf[-1].islower()
            if is_abbrev or is_cont:
                buf = buf + " " + line
                continue
            merged.append(buf)
        buf = line
    if buf:
        merged.append(buf)

    rejoined = " ".join(merged)

    # Step 2: split at sentence boundaries, skipping abbreviation periods
    spans: list[tuple[str, int, int]] = []
    boundary_re = re.compile(r"([.!?])\s+(?=[A-Z])")
    prev = 0
    for match in boundary_re.finditer(rejoined):
        preceding = rejoined[prev:match.start()]
        ab = _ABBREV_LINE_END_RE.search(preceding.rstrip())
        if ab and ab.group(1).lower() in _ABBREVS:
            continue
        sentence = (preceding + match.group(1)).strip()
        if sentence:
            spans.append((sentence, prev, match.start() + 1))
        prev = match.end()

    tail = rejoined[prev:].strip()
    if tail:
        spans.append((tail, prev, len(rejoined)))

    return spans if spans else [(rejoined, 0, len(rejoined))]


# Seed-term lists, claim-link roles, occurrence verbs, and subject-trigger verbs
# now come from the domain config (built in __init__). The former module-level
# _seed_terms / _default_*_terms / SUBJECT_TRIGGER_VERBS / _OCCURRENCE_VERB_PREP
# constants were Turnbull-specific and have been relocated to domain_schema.yaml.
_TOPIC_VERB = re.compile(
    r"\b(concerns?|regarding|in reference to|conditions? at|management of)\s*$",
    re.IGNORECASE,
)


def _find_term_matches(text: str, terms: list[str]) -> list[tuple[str, int, int]]:
    matches: list[tuple[str, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for term in terms:
        for match in re.finditer(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            span = (match.start(), match.end())
            if any(not (span[1] <= left or span[0] >= right) for left, right in occupied):
                continue
            occupied.append(span)
            matches.append((match.group(0), match.start(), match.end()))
    matches.sort(key=lambda row: row[1])
    return matches


def _dedupe_claim_links(claim_links: list[ClaimLinkDraft]) -> list[ClaimLinkDraft]:
    deduped: dict[tuple[int | None, int | None, str], ClaimLinkDraft] = {}
    for draft in claim_links:
        relation_type = validate_claim_link_relation(draft.relation_type)
        if not relation_type:
            continue
        draft.relation_type = relation_type
        key = (draft.start_offset, draft.end_offset, draft.normalized_form)
        existing = deduped.get(key)
        if not existing or claim_relation_priority(draft.relation_type) < claim_relation_priority(existing.relation_type):
            deduped[key] = draft
    return sorted(deduped.values(), key=lambda row: (row.start_offset or -1, row.end_offset or -1, row.normalized_form))


def _resolve_rules_fallback(flag: bool | None) -> bool:
    """Whether the rule extractor proposes unclassified candidates for un-typed
    (but valid) sentences. Env ``CLAIM_RULES_FALLBACK`` (default off) enables the
    recall-first mode used when an LLM validates the rule proposals downstream.
    """
    if flag is not None:
        return flag
    return (os.environ.get("CLAIM_RULES_FALLBACK") or "0").strip().lower() in {"1", "true", "yes", "on"}


class RuleBasedClaimExtractor:
    _TYPE_SCORE_THRESHOLD: float = 0.85
    # Confidence assigned to unclassified fallback candidates — deliberately low:
    # these are recall-first proposals for an LLM validator to confirm/reject.
    _UNCLASSIFIED_CONF: float = 0.30
    _TYPE_MARGIN: float = 0.35
    # Confidence calibration constants
    _SCORE_CEILING: float = 2.5   # raw pattern scores at or above this map to score_norm=1.0
    _CONF_LO: float = 0.60        # confidence at threshold score with zero margin
    _CONF_HI: float = 0.92        # confidence at ceiling score with full margin
    _SCORE_WEIGHT: float = 0.65   # fraction of range driven by score signal
    _MARGIN_WEIGHT: float = 0.35  # fraction of range driven by margin signal
    _EPISTEMIC_DISCOUNT: float = 0.08  # subtracted for uncertain epistemic status
    _uncertain_tokens = re.compile(r"\b(about|approx|approximately|around|estimated|reported|possibly|likely)\b", re.IGNORECASE)

    def __init__(
        self,
        resources_dir: Path | None = None,
        config: "DomainConfig | None" = None,
        unclassified_fallback: bool | None = None,
    ) -> None:
        # The extractor always operates on a DomainConfig. When none is supplied,
        # load the named resources dir (or the Turnbull corpus explicitly) — never
        # a silent default. This collapses the former config/resources/bare
        # branches and removes the module-level Turnbull term caches.
        if config is None:
            from gemynd.core.domain_config import load_domain_config
            config = load_domain_config(
                resources_dir if resources_dir is not None else TURNBULL_RESOURCES_DIR
            )

        self._unclassified_fallback = _resolve_rules_fallback(unclassified_fallback)
        # Validity gate for fallback proposals — same check phases.py applies to
        # every draft, so an un-typed sentence only becomes a candidate if it
        # reads as a genuine assertive sentence (drops ads / headings / OCR junk).
        self._validator_verb_re = config.validator_verb_re
        self._validator_heading_re = config.validator_heading_re
        self._ROLE_POLICY = config.claim_role_policy
        self._type_scored_patterns = config.claim_type_patterns
        self._seed_entities_all = list(config.seed_entities)
        self._entity_type_rules = list(config.entity_type_rules)
        self._claim_location_relation = config.claim_location_relation
        self._claim_link_roles = config.claim_link_roles
        # Per-role term lists, keyed by the role's entity_type, gathered once.
        self._role_terms = {
            role.entity_type: _collect_seed_terms(role.seed_types, config.seed_entities)
            for role in self._claim_link_roles
        }
        self._occurrence_re = _compile_occurrence_re(config.claim_link_occurrence_verbs)
        self._subject_re = _compile_subject_re(config.claim_link_subject_verbs)

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        claims: list[ClaimDraft] = []
        for sentence, start, end in _split_sentences(paragraph_text):
            claim_type, best_score, margin, secondary_label, matched_patterns, fallback_used = (
                self._detect_type_scored(sentence)
            )
            if not claim_type:
                fb = self._unclassified_draft(sentence, start, end)
                if fb is not None:
                    claims.append(fb)
                continue
            normalized = normalize_for_match(sentence)
            epistemic_status = "uncertain" if self._uncertain_tokens.search(sentence) else "certain"
            score_norm = min(
                (best_score - self._TYPE_SCORE_THRESHOLD) / (self._SCORE_CEILING - self._TYPE_SCORE_THRESHOLD),
                1.0,
            )
            margin_norm = min(margin / self._TYPE_MARGIN, 1.0)
            combined = self._SCORE_WEIGHT * score_norm + self._MARGIN_WEIGHT * margin_norm
            confidence_raw = self._CONF_LO + (self._CONF_HI - self._CONF_LO) * combined
            if epistemic_status == "uncertain":
                confidence_raw -= self._EPISTEMIC_DISCOUNT
            extraction_confidence = round(max(0.0, min(1.0, confidence_raw)), 4)
            claim_links = self._extract_claim_links(sentence, claim_type, start)

            trace: list[str] = [
                f"type_pattern:{claim_type}",
                f"raw_score:{best_score:.3f}",
                f"margin:{margin:.3f}",
                f"conf:{extraction_confidence:.4f}",
            ]
            if fallback_used and secondary_label:
                trace.append(f"narrow_margin:{secondary_label}")
            trace.append(f"epistemic:{epistemic_status}")
            for link in claim_links:
                trace.append(f"link:{link.entity_type_hint}={link.surface_form}->{link.relation_type}")

            claims.append(
                ClaimDraft(
                    claim_type=claim_type,
                    source_sentence=sentence,
                    normalized_sentence=normalized,
                    epistemic_status=epistemic_status,
                    extraction_confidence=extraction_confidence,
                    evidence_start=start,
                    evidence_end=end,
                    notes=secondary_label or "",
                    claim_links=claim_links,
                    extraction_source="rules",
                    decision_trace=trace,
                    matched_patterns=matched_patterns,
                    fallback_used=fallback_used,
                )
            )
        return claims

    def _unclassified_draft(self, sentence: str, start: int, end: int) -> "ClaimDraft | None":
        """Recall-first fallback: propose an ``unclassified_assertion`` candidate
        for a sentence that no claim-type pattern matched but that still reads as a
        genuine assertive sentence. Off unless ``unclassified_fallback`` is set.

        The proposal is deliberately low-confidence — it is meant for an LLM
        validator to confirm or reject downstream, not to stand on its own. The
        validity gate mirrors the ``is_valid_claim_sentence`` post-filter in
        ``phases.py`` (same verb/heading regexes) so a proposal is never something
        that filter would immediately drop.
        """
        if not self._unclassified_fallback:
            return None
        valid, _reason = is_valid_claim_sentence(
            sentence,
            verb_re=self._validator_verb_re,
            heading_re=self._validator_heading_re,
        )
        if not valid:
            return None
        normalized = normalize_for_match(sentence)
        epistemic_status = "uncertain" if self._uncertain_tokens.search(sentence) else "certain"
        claim_links = self._extract_claim_links(sentence, UNCLASSIFIED_TYPE, start)
        trace = [
            f"type_pattern:{UNCLASSIFIED_TYPE}",
            "unclassified_fallback",
            "needs_llm_validation",
            f"conf:{self._UNCLASSIFIED_CONF:.4f}",
            f"epistemic:{epistemic_status}",
        ]
        for link in claim_links:
            trace.append(f"link:{link.entity_type_hint}={link.surface_form}->{link.relation_type}")
        return ClaimDraft(
            claim_type=UNCLASSIFIED_TYPE,
            source_sentence=sentence,
            normalized_sentence=normalized,
            epistemic_status=epistemic_status,
            extraction_confidence=self._UNCLASSIFIED_CONF,
            evidence_start=start,
            evidence_end=end,
            notes="",
            claim_links=claim_links,
            extraction_source="rules_unclassified",
            decision_trace=trace,
            matched_patterns=[],
            fallback_used=True,
        )

    def _detect_type_scored(
        self, sentence: str
    ) -> tuple[str | None, float, float, str | None, list[str], bool]:
        """Return (claim_type, best_score, margin, secondary_label, matched_patterns, fallback_used).

        best_score and margin are raw pattern accumulation values; confidence
        calibration is performed by the caller.
        """
        scores: dict[str, float] = {}
        matched_patterns: list[str] = []
        for claim_type, pattern, base_weight in self._type_scored_patterns:
            hits = pattern.findall(sentence)
            if hits:
                scores[claim_type] = scores.get(claim_type, 0.0) + base_weight * len(hits)
                for token in hits:
                    t = token if isinstance(token, str) else token[0]
                    matched_patterns.append(f"{claim_type}:{t.lower()}")

        if not scores:
            return None, 0.0, 0.0, None, [], False

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        second_type = ranked[1][0] if len(ranked) > 1 else None

        if best_score < self._TYPE_SCORE_THRESHOLD:
            return None, 0.0, 0.0, None, [], False

        margin = best_score - second_score
        secondary_label: str | None = None
        fallback_used = False
        if margin < self._TYPE_MARGIN:
            secondary_label = f"secondary:{second_type}" if second_type else None
            fallback_used = True

        return best_type, best_score, margin, secondary_label, matched_patterns, fallback_used

    def _extract_claim_links(self, sentence: str, claim_type: str, sentence_start: int) -> list[ClaimLinkDraft]:
        claim_links: list[ClaimLinkDraft] = []

        # Roles are iterated in config order, which is significant: cross-role
        # span collisions are resolved downstream by relation priority with an
        # offset-stable sort, so order must match the domain's declared order.
        for role in self._claim_link_roles:
            terms = self._role_terms.get(role.entity_type, [])
            for surface, start, end in _find_term_matches(sentence, terms):
                if role.is_location_role:
                    relation_type = (
                        self._claim_location_relation
                        if self._is_occurrence_location(sentence, start)
                        else role.default_relation
                    )
                    entity_type_hint = role.entity_type
                    lowered = surface.lower()
                    for substring, label in role.hint_substring_overrides:
                        if substring in lowered:
                            entity_type_hint = label
                            break
                else:
                    relation_type = self._ROLE_POLICY.get(
                        (claim_type, role.entity_type), role.default_relation
                    )
                    if relation_type is None:
                        if claim_type not in role.fallback_claim_types:
                            continue
                        relation_type = role.fallback_relation
                    entity_type_hint = role.entity_type
                if relation_type is None:
                    continue
                claim_links.append(
                    self._make_link(
                        surface, relation_type, sentence_start + start, sentence_start + end, entity_type_hint
                    )
                )

        claim_links.extend(self._extract_subject_links(sentence, sentence_start))
        return _dedupe_claim_links(claim_links)

    @staticmethod
    def _make_link(
        surface_form: str,
        relation_type: str,
        start_offset: int,
        end_offset: int,
        entity_type_hint: str | None,
    ) -> ClaimLinkDraft:
        return ClaimLinkDraft(
            surface_form=surface_form,
            normalized_form=normalize_for_match(surface_form),
            relation_type=relation_type,
            start_offset=start_offset,
            end_offset=end_offset,
            entity_type_hint=entity_type_hint,
        )

    def _is_occurrence_location(self, sentence: str, start_offset: int) -> bool:
        if self._occurrence_re is None:
            return False
        prefix = sentence[:start_offset].lower().rstrip()
        return bool(self._occurrence_re.search(prefix))

    def _extract_subject_links(self, sentence: str, sentence_start: int) -> list[ClaimLinkDraft]:
        if self._subject_re is None or not self._subject_re.search(sentence):
            return []
        subject_match = re.match(r"\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+){0,3})", sentence)
        if not subject_match:
            return []
        surface = subject_match.group(1).strip()
        entity_type_hint = infer_entity_type(
            surface,
            seed_entities=self._seed_entities_all,
            rules=self._entity_type_rules,
        )
        return [
            self._make_link(
                surface,
                "SUBJECT_OF_CLAIM",
                sentence_start + subject_match.start(1),
                sentence_start + subject_match.end(1),
                entity_type_hint,
            )
        ]


class ClaimLLMAdapter(Protocol):
    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        ...


class NullLLMAdapter:
    def extract_claims(self, paragraph_text: str) -> list[dict[str, object]]:
        return []


class LLMClaimExtractor:
    def __init__(self, adapter: ClaimLLMAdapter) -> None:
        self._adapter = adapter

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        drafts: list[ClaimDraft] = []
        for row in self._adapter.extract_claims(paragraph_text):
            raw = str(row.get("source_sentence", "") or row.get("raw_sentence", "")).strip()
            if not raw:
                continue
            normalized = str(row.get("normalized_sentence", "")).strip() or normalize_for_match(raw)
            drafts.append(
                ClaimDraft(
                    claim_type=str(row.get("claim_type", "management_action")),
                    source_sentence=raw,
                    normalized_sentence=normalized,
                    epistemic_status=str(row.get("epistemic_status", row.get("certainty", "uncertain"))),
                    extraction_confidence=max(0.0, min(1.0, float(row.get("extraction_confidence", 0.6)))),
                    evidence_start=int(row["evidence_start"]) if row.get("evidence_start") is not None else None,
                    evidence_end=int(row["evidence_end"]) if row.get("evidence_end") is not None else None,
                    claim_date=str(row["claim_date"]) if row.get("claim_date") else None,
                    notes=str(row.get("notes", "")),
                    claim_links=self._parse_claim_links(row.get("claim_links", [])),
                    extraction_source="llm",
                    decision_trace=["source:llm"],
                    matched_patterns=[],
                    fallback_used=False,
                )
            )
        return drafts

    @staticmethod
    def _parse_claim_links(payload: object) -> list[ClaimLinkDraft]:
        if not isinstance(payload, list):
            return []
        claim_links: list[ClaimLinkDraft] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            relation_type = validate_claim_link_relation(str(row.get("relation_type", "")))
            raw = str(row.get("surface_form", "") or row.get("text", "")).strip()
            # Always canonically normalize the surface form. The LLM's own
            # "normalized_form" is inconsistent (e.g. "Spokane, Washington" vs the
            # entity's canonical "spokane"), which broke every claim-link
            # resolution path. Derive it the same way mentions/entities do.
            normalized = normalize_for_match(raw)
            if not relation_type or not raw:
                continue
            claim_links.append(
                ClaimLinkDraft(
                    surface_form=raw,
                    normalized_form=normalized,
                    relation_type=relation_type,
                    start_offset=int(row["start_offset"]) if row.get("start_offset") is not None else None,
                    end_offset=int(row["end_offset"]) if row.get("end_offset") is not None else None,
                    entity_type_hint=str(row["entity_type_hint"]) if row.get("entity_type_hint") else None,
                )
            )
        return _dedupe_claim_links(claim_links)


@dataclass
class HybridTelemetry:
    rule_only: list[str]
    llm_only: list[str]
    overlapping: list[str]
    label_changed: list[str]
    links_inherited: list[str]
    confidence_deltas: list[tuple[str, float, float]]

    @property
    def rule_only_count(self) -> int:
        return len(self.rule_only)

    @property
    def llm_only_count(self) -> int:
        return len(self.llm_only)

    @property
    def overlap_count(self) -> int:
        return len(self.overlapping)


# LLM-call gating: the LLM is the most expensive stage (a network call per
# paragraph on a local model). These modes let the LLM act as a *fallback* to the
# rule extractor instead of always running — trading some recall (incremental
# claims on already-covered paragraphs) for far fewer calls on bulk local ingest.
_LLM_GATE_MODES = ("always", "on_miss", "prefilter", "on_miss_prefilter")


def _resolve_llm_gate(gate: str | None) -> str:
    resolved = (gate or os.environ.get("CLAIM_LLM_GATE") or "always").strip().lower()
    if resolved not in _LLM_GATE_MODES:
        return "always"
    return resolved


# LLM role: `extractor` (default) runs the LLM to find claims independently and
# merges with rules. `validator` inverts that — the rules propose candidates
# (with the unclassified fallback on) and the LLM only confirms/re-types/rejects
# them in cheap batched calls. Validation is far cheaper than extraction and lets
# the rules run at high recall while the LLM supplies precision.
_LLM_ROLES = ("extractor", "validator")


def _resolve_llm_role(role: str | None) -> str:
    resolved = (role or os.environ.get("CLAIM_LLM_ROLE") or "extractor").strip().lower()
    if resolved not in _LLM_ROLES:
        return "extractor"
    return resolved


def apply_verdicts(
    candidates: list["ClaimDraft"],
    verdicts: list[dict[str, Any]],
    min_confirmed_conf: float = 0.5,
) -> list["ClaimDraft"]:
    """Map validator verdicts onto candidate drafts.

    Rejected candidates (``is_claim`` false) are dropped; confirmed ones are
    re-typed to the validator's ``claim_type`` (when given), have their
    confidence raised to the verdict's (floored at *min_confirmed_conf* so a
    confirmed claim never keeps the 0.30 proposal marker), and are re-tagged
    ``rules_validated``. Verdicts align to *candidates* by position.
    """
    survivors: list[ClaimDraft] = []
    for draft, verdict in zip(candidates, verdicts):
        if not verdict.get("is_claim", True):
            continue
        new_type = verdict.get("claim_type") or draft.claim_type
        conf = verdict.get("confidence")
        new_conf = max(float(conf), min_confirmed_conf) if isinstance(conf, (int, float)) else draft.extraction_confidence
        draft.claim_type = new_type
        draft.extraction_confidence = round(min(1.0, new_conf), 4)
        draft.extraction_source = "rules_validated"
        draft.decision_trace = list(draft.decision_trace) + [
            f"llm_validated:confirmed:{new_type}:conf={draft.extraction_confidence:.2f}"
        ]
        survivors.append(draft)
    return survivors


def _paragraph_is_low_value(text: str) -> bool:
    """Heuristic: is this paragraph unlikely to hold any extractable claim?

    Targets OCR garbage / mastheads / tables / headline fragments so a call is
    not spent on them. Conservative — only skips clear non-prose (claims live in
    sentences, which this keeps).
    """
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(1, len(text)) < 0.6:  # mostly digits / punctuation
        return True
    words = text.split()
    if len(words) < 6:  # too short to be an assertive sentence
        return True
    real_words = sum(1 for w in words if len(w) >= 3 and w.islower())
    if real_words / max(1, len(words)) < 0.25:  # ALLCAPS / garbled OCR
        return True
    return False


class HybridClaimExtractor:
    def __init__(
        self,
        rules_extractor: ClaimExtractor | None = None,
        llm_extractor: ClaimExtractor | None = None,
        resources_dir: Path | None = None,
        config: "DomainConfig | None" = None,
        token_logger: Any | None = None,
        gate: str | None = None,
        role: str | None = None,
        validator: Any | None = None,
    ) -> None:
        self._role = _resolve_llm_role(role)
        # In validator mode the rules MUST propose unclassified candidates for the
        # LLM to judge, so force the fallback on (unless a rules extractor was
        # injected, e.g. in tests).
        self._rules = rules_extractor or RuleBasedClaimExtractor(
            resources_dir=resources_dir,
            config=config,
            unclassified_fallback=True if self._role == "validator" else None,
        )
        self._validator = validator
        if self._role == "validator":
            self._llm = None
            if self._validator is None:
                from gemynd.ingest.extractors.claim_validator_adapter import try_create_validator
                self._validator = try_create_validator(config=config)
        elif llm_extractor is not None:
            self._llm = llm_extractor
        else:
            from gemynd.ingest.extractors.llm_adapter_factory import try_create_llm_adapter
            self._llm = LLMClaimExtractor(try_create_llm_adapter(config=config, token_logger=token_logger))
        # `always` (default) preserves legacy behavior; `on_miss` makes the LLM a
        # true fallback (skip it when rules already found a claim); `prefilter`
        # skips low-value paragraphs; `on_miss_prefilter` does both.
        self._gate = _resolve_llm_gate(gate)
        self.last_telemetry: HybridTelemetry | None = None

    def _validate_extract(self, paragraph_text: str) -> list[ClaimDraft]:
        """Validator role: rules propose, the LLM confirms/re-types/rejects.

        Pattern-typed rule claims pass through untouched (already typed with real
        confidence). Only the unclassified fallback candidates are sent to the
        validator; if no validator is configured the candidates survive as-is
        (fail-open — never silently drop rule proposals).
        """
        drafts = self._rules.extract(paragraph_text)
        typed = [d for d in drafts if d.extraction_source != "rules_unclassified"]
        candidates = [d for d in drafts if d.extraction_source == "rules_unclassified"]
        if not candidates:
            return typed
        if self._validator is None:
            return typed + candidates
        verdicts = self._validator.validate([c.source_sentence for c in candidates])
        return typed + apply_verdicts(candidates, verdicts)

    def _should_call_llm(self, paragraph_text: str, rule_hit: bool) -> bool:
        gate = self._gate
        if gate == "always":
            return True
        if gate in ("on_miss", "on_miss_prefilter") and rule_hit:
            return False
        if gate in ("prefilter", "on_miss_prefilter") and _paragraph_is_low_value(paragraph_text):
            return False
        return True

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        if self._role == "validator":
            return self._validate_extract(paragraph_text)
        telemetry = HybridTelemetry(
            rule_only=[], llm_only=[], overlapping=[],
            label_changed=[], links_inherited=[], confidence_deltas=[],
        )
        merged: dict[str, ClaimDraft] = {}
        llm_seen: set[str] = set()

        for draft in self._rules.extract(paragraph_text):
            merged[draft.normalized_sentence] = draft

        llm_drafts = (
            self._llm.extract(paragraph_text)
            if self._should_call_llm(paragraph_text, bool(merged))
            else []
        )
        for llm_draft in llm_drafts:
            key = llm_draft.normalized_sentence
            llm_seen.add(key)
            rule_draft = merged.get(key)

            if rule_draft is None:
                telemetry.llm_only.append(key)
                merged[key] = llm_draft
                continue

            # Overlap — both rule and LLM saw this sentence
            rule_conf = rule_draft.extraction_confidence
            llm_conf = llm_draft.extraction_confidence
            telemetry.overlapping.append(key)
            telemetry.confidence_deltas.append((key, rule_conf, llm_conf))

            if llm_conf > rule_conf:
                # LLM wins
                inherited = not llm_draft.claim_links and bool(rule_draft.claim_links)
                if inherited:
                    llm_draft.claim_links = list(rule_draft.claim_links)
                    telemetry.links_inherited.append(key)

                label_diff = llm_draft.claim_type != rule_draft.claim_type
                if label_diff:
                    telemetry.label_changed.append(key)

                llm_draft.extraction_source = "hybrid"
                tag_hybrid_trace(
                    llm_draft,
                    f"hybrid:llm_won(conf={llm_conf:.3f}>rule={rule_conf:.3f})",
                )
                if inherited:
                    llm_draft.decision_trace.append("hybrid:links_inherited")
                if label_diff:
                    llm_draft.decision_trace.append(
                        f"hybrid:label_changed:{rule_draft.claim_type}->{llm_draft.claim_type}"
                    )
                merged[key] = llm_draft
            else:
                # Rule wins — annotate the existing rule draft
                rule_draft.extraction_source = "hybrid"
                tag_hybrid_trace(
                    rule_draft,
                    f"hybrid:rule_won(conf={rule_conf:.3f}>={llm_conf:.3f})",
                )

        # Sentences only seen by rules (not in LLM output at all)
        for key, draft in merged.items():
            if key not in llm_seen and draft.extraction_source == "rules":
                telemetry.rule_only.append(key)

        self.last_telemetry = telemetry
        return list(merged.values())
