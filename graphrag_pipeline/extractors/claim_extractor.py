from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..claim_contract import (
    ALLOWED_CLAIM_TYPES,
    CLAIM_LOCATION_RELATION,
    UNCLASSIFIED_TYPE,
    claim_relation_priority,
    validate_claim_link_relation,
)
from ..resolver import default_seed_entities
from ..resource_loader import load_claim_type_patterns, load_claim_role_policy

_LOADED_TYPE_PATTERNS = load_claim_type_patterns()
_LOADED_ROLE_POLICY = load_claim_role_policy()


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


def _normalize_sentence(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"\s+", " ", lowered)


def _normalize_link_form(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    text = text.strip()
    if not text:
        return []
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^.!?]+[.!?]?", text):
        sentence = match.group(0).strip()
        if sentence:
            spans.append((sentence, match.start(), match.end()))
    return spans

# Types the extractor can produce directly (excludes the coercion fallback).
VALID_CLAIM_TYPES: frozenset[str] = ALLOWED_CLAIM_TYPES - {UNCLASSIFIED_TYPE}
_SEED_ENTITIES = default_seed_entities()


def _seed_terms(*entity_types: str) -> list[str]:
    labels = set(entity_types)
    return sorted(
        {entity.name for entity in _SEED_ENTITIES if entity.entity_type in labels},
        key=len,
        reverse=True,
    )


SPECIES_TERMS = _seed_terms("Species")
HABITAT_TERMS = _seed_terms("Habitat")
METHOD_TERMS = _seed_terms("SurveyMethod")
LOCATION_TERMS = _seed_terms("Place", "Refuge")
ACTIVITY_TERMS = _seed_terms("Activity")
SUBJECT_TRIGGER_VERBS = re.compile(
    r"\b(recommended|reported|noted|managed|authorized|requested|directed|approved|announced|said|wrote)\b",
    re.IGNORECASE,
)
_OCCURRENCE_VERB_PREP = re.compile(
    r"\b(observed|found|seen|collected|taken|trapped|nested|wintered"
    r"|came|arrived|occurred|burned|happened|departed)"
    r"\s+(?:into|in|at|near|on|within|around|from)\s*$",
    re.IGNORECASE,
)
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


class RuleBasedClaimExtractor:
    _ROLE_POLICY: dict[tuple[str, str], str] = _LOADED_ROLE_POLICY
    _type_scored_patterns: list[tuple[str, re.Pattern[str], float]] = _LOADED_TYPE_PATTERNS
    _TYPE_SCORE_THRESHOLD: float = 0.8
    _TYPE_MARGIN: float = 0.3
    _TYPE_CONFIDENCE_PENALTY: float = 0.08
    _uncertain_tokens = re.compile(r"\b(about|approx|approximately|around|estimated|reported|possibly|likely)\b", re.IGNORECASE)

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        claims: list[ClaimDraft] = []
        for sentence, start, end in _split_sentences(paragraph_text):
            claim_type, confidence_adj, secondary_label, matched_patterns, fallback_used = (
                self._detect_type_scored(sentence)
            )
            if not claim_type:
                continue
            normalized = _normalize_sentence(sentence)
            epistemic_status = "uncertain" if self._uncertain_tokens.search(sentence) else "certain"
            base_confidence = 0.78 if epistemic_status == "certain" else 0.68
            extraction_confidence = round(base_confidence + confidence_adj, 4)
            claim_links = self._extract_claim_links(sentence, claim_type, start)

            trace: list[str] = [f"type_pattern:{claim_type}", f"type_score:{base_confidence + confidence_adj:.2f}"]
            if fallback_used and secondary_label:
                trace.append(f"confidence_penalty:{secondary_label}")
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

    def _detect_type_scored(
        self, sentence: str
    ) -> tuple[str | None, float, str | None, list[str], bool]:
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
            return None, 0.0, None, [], False

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        second_type = ranked[1][0] if len(ranked) > 1 else None

        if best_score < self._TYPE_SCORE_THRESHOLD:
            return None, 0.0, None, [], False

        confidence_adjustment = 0.0
        secondary_label: str | None = None
        fallback_used = False
        if best_score - second_score < self._TYPE_MARGIN:
            confidence_adjustment = -self._TYPE_CONFIDENCE_PENALTY
            secondary_label = f"secondary:{second_type}" if second_type else None
            fallback_used = True

        return best_type, confidence_adjustment, secondary_label, matched_patterns, fallback_used

    def _extract_claim_links(self, sentence: str, claim_type: str, sentence_start: int) -> list[ClaimLinkDraft]:
        claim_links: list[ClaimLinkDraft] = []

        for surface, start, end in _find_term_matches(sentence, SPECIES_TERMS):
            relation_type = self._ROLE_POLICY.get((claim_type, "Species"), "SPECIES_FOCUS")
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "Species"))

        for surface, start, end in _find_term_matches(sentence, HABITAT_TERMS):
            relation_type = self._ROLE_POLICY.get((claim_type, "Habitat"), "HABITAT_FOCUS")
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "Habitat"))

        for surface, start, end in _find_term_matches(sentence, METHOD_TERMS):
            relation_type = self._ROLE_POLICY.get((claim_type, "SurveyMethod"), "METHOD_FOCUS")
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "SurveyMethod"))

        _activity_fallback_types = {"predator_control", "management_action", "economic_use", "development_activity", "public_contact"}
        for surface, start, end in _find_term_matches(sentence, ACTIVITY_TERMS):
            relation_type = self._ROLE_POLICY.get((claim_type, "Activity"))
            if relation_type is None:
                if claim_type not in _activity_fallback_types:
                    continue
                relation_type = "MANAGEMENT_TARGET"
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "Activity"))

        for surface, start, end in _find_term_matches(sentence, LOCATION_TERMS):
            relation_type = CLAIM_LOCATION_RELATION if self._is_occurrence_location(sentence, start) else "LOCATION_FOCUS"
            entity_type_hint = "Refuge" if "refuge" in surface.lower() else "Place"
            claim_links.append(
                self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, entity_type_hint)
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
            normalized_form=_normalize_link_form(surface_form),
            relation_type=relation_type,
            start_offset=start_offset,
            end_offset=end_offset,
            entity_type_hint=entity_type_hint,
        )

    @staticmethod
    def _is_occurrence_location(sentence: str, start_offset: int) -> bool:
        prefix = sentence[:start_offset].lower().rstrip()
        return bool(_OCCURRENCE_VERB_PREP.search(prefix))

    def _extract_subject_links(self, sentence: str, sentence_start: int) -> list[ClaimLinkDraft]:
        if not SUBJECT_TRIGGER_VERBS.search(sentence):
            return []
        subject_match = re.match(r"\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+){0,3})", sentence)
        if not subject_match:
            return []
        surface = subject_match.group(1).strip()
        entity_type_hint = "Organization" if any(token in surface for token in ("Club", "Council", "Assn", "Association", "WPA")) else "Person"
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
            normalized = str(row.get("normalized_sentence", "")).strip() or _normalize_sentence(raw)
            drafts.append(
                ClaimDraft(
                    claim_type=str(row.get("claim_type", "management_action")),
                    source_sentence=raw,
                    normalized_sentence=normalized,
                    epistemic_status=str(row.get("epistemic_status", row.get("certainty", "uncertain"))),
                    extraction_confidence=float(row.get("extraction_confidence", 0.6)),
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
            normalized = str(row.get("normalized_form", "")).strip() or _normalize_link_form(raw)
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


class HybridClaimExtractor:
    def __init__(
        self,
        rules_extractor: ClaimExtractor | None = None,
        llm_extractor: ClaimExtractor | None = None,
    ) -> None:
        self._rules = rules_extractor or RuleBasedClaimExtractor()
        self._llm = llm_extractor or LLMClaimExtractor(NullLLMAdapter())
        self.last_telemetry: HybridTelemetry | None = None

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        telemetry = HybridTelemetry(
            rule_only=[], llm_only=[], overlapping=[],
            label_changed=[], links_inherited=[], confidence_deltas=[],
        )
        merged: dict[str, ClaimDraft] = {}
        llm_seen: set[str] = set()

        for draft in self._rules.extract(paragraph_text):
            merged[draft.normalized_sentence] = draft

        for llm_draft in self._llm.extract(paragraph_text):
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
                llm_draft.decision_trace = list(llm_draft.decision_trace)
                llm_draft.decision_trace.append(
                    f"hybrid:llm_won(conf={llm_conf:.3f}>rule={rule_conf:.3f})"
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
                rule_draft.decision_trace = list(rule_draft.decision_trace)
                rule_draft.decision_trace.append(
                    f"hybrid:rule_won(conf={rule_conf:.3f}>={llm_conf:.3f})"
                )

        # Sentences only seen by rules (not in LLM output at all)
        for key, draft in merged.items():
            if key not in llm_seen and draft.extraction_source == "rules":
                telemetry.rule_only.append(key)

        self.last_telemetry = telemetry
        return list(merged.values())
