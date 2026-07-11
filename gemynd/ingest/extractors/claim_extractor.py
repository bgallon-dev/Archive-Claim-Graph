from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gemynd.core.claim_contract import (
    ALLOWED_CLAIM_TYPES,
    CLAIM_LOCATION_RELATION,
    UNCLASSIFIED_TYPE,
    claim_relation_priority,
    validate_claim_link_relation,
)
from gemynd.core.entity_type_inference import infer_entity_type
from gemynd.core.resolver import default_seed_entities
from gemynd.shared.resource_loader import (
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


def _seed_terms(*entity_types: str, seed: list | None = None) -> list[str]:
    entities = seed if seed is not None else default_seed_entities()
    labels = set(entity_types)
    return sorted(
        {entity.name for entity in entities if entity.entity_type in labels},
        key=len,
        reverse=True,
    )
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
    _TYPE_SCORE_THRESHOLD: float = 0.85
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
    ) -> None:
        if config is not None:
            self._ROLE_POLICY = config.claim_role_policy
            self._type_scored_patterns = config.claim_type_patterns
            self._seed_entities_all = list(config.seed_entities)
            self._entity_type_rules = list(config.entity_type_rules)
            self._species_terms  = _seed_terms("Species",         seed=config.seed_entities)
            self._habitat_terms  = _seed_terms("Habitat",         seed=config.seed_entities)
            self._method_terms   = _seed_terms("SurveyMethod",    seed=config.seed_entities)
            self._location_terms = _seed_terms("Place", "Refuge", seed=config.seed_entities)
            self._activity_terms = _seed_terms("Activity",        seed=config.seed_entities)
        elif resources_dir is not None:
            self._ROLE_POLICY = load_claim_role_policy(resources_dir)
            self._type_scored_patterns = load_claim_type_patterns(resources_dir)
            _seed = default_seed_entities(resources_dir)
            self._seed_entities_all = list(_seed)
            self._entity_type_rules = load_entity_type_rules(resources_dir)
            self._species_terms  = _seed_terms("Species",               seed=_seed)
            self._habitat_terms  = _seed_terms("Habitat",               seed=_seed)
            self._method_terms   = _seed_terms("SurveyMethod",          seed=_seed)
            self._location_terms = _seed_terms("Place", "Refuge",       seed=_seed)
            self._activity_terms = _seed_terms("Activity",              seed=_seed)
        else:
            self._ROLE_POLICY          = default_claim_role_policy()
            self._type_scored_patterns = default_claim_type_patterns()
            self._seed_entities_all    = list(default_seed_entities())
            self._entity_type_rules    = load_entity_type_rules()
            self._species_terms        = list(seed_terms_for("Species"))
            self._habitat_terms        = list(seed_terms_for("Habitat"))
            self._method_terms         = list(seed_terms_for("SurveyMethod"))
            self._location_terms       = list(seed_terms_for("Place", "Refuge"))
            self._activity_terms       = list(seed_terms_for("Activity"))

    def extract(self, paragraph_text: str) -> list[ClaimDraft]:
        claims: list[ClaimDraft] = []
        for sentence, start, end in _split_sentences(paragraph_text):
            claim_type, best_score, margin, secondary_label, matched_patterns, fallback_used = (
                self._detect_type_scored(sentence)
            )
            if not claim_type:
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

        for surface, start, end in _find_term_matches(sentence, self._species_terms):
            relation_type = self._ROLE_POLICY.get((claim_type, "Species"), "SPECIES_FOCUS")
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "Species"))

        for surface, start, end in _find_term_matches(sentence, self._habitat_terms):
            relation_type = self._ROLE_POLICY.get((claim_type, "Habitat"), "HABITAT_FOCUS")
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "Habitat"))

        for surface, start, end in _find_term_matches(sentence, self._method_terms):
            relation_type = self._ROLE_POLICY.get((claim_type, "SurveyMethod"), "METHOD_FOCUS")
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "SurveyMethod"))

        _activity_fallback_types = {"predator_control", "management_action", "economic_use", "development_activity", "public_contact"}
        for surface, start, end in _find_term_matches(sentence, self._activity_terms):
            relation_type = self._ROLE_POLICY.get((claim_type, "Activity"))
            if relation_type is None:
                if claim_type not in _activity_fallback_types:
                    continue
                relation_type = "MANAGEMENT_TARGET"
            claim_links.append(self._make_link(surface, relation_type, sentence_start + start, sentence_start + end, "Activity"))

        for surface, start, end in _find_term_matches(sentence, self._location_terms):
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
            normalized_form=normalize_for_match(surface_form),
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
            normalized = str(row.get("normalized_form", "")).strip() or normalize_for_match(raw)
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
        resources_dir: Path | None = None,
        config: "DomainConfig | None" = None,
        token_logger: Any | None = None,
    ) -> None:
        self._rules = rules_extractor or RuleBasedClaimExtractor(
            resources_dir=resources_dir, config=config
        )
        if llm_extractor is not None:
            self._llm = llm_extractor
        else:
            from gemynd.ingest.extractors.openai_claim_adapter import create_claim_llm_adapter
            self._llm = LLMClaimExtractor(create_claim_llm_adapter(config=config, token_logger=token_logger))
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
