from __future__ import annotations

import difflib
import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TYPE_CHECKING

from gemynd.core.resolver import default_seed_entities
from gemynd.shared.resource_loader import TURNBULL_RESOURCES_DIR
from gemynd.shared.text_utils import normalize_for_match
from ..span_utils import resolve_overlapping_spans
from ._draft_audit import tag_hybrid_trace
# Explicit-dir calls into the shared process-wide resource cache: the caller
# names the corpus (never a silent no-arg default), but repeat loads still hit
# the same cached instance the rest of the extractor family uses.
from ._resources import default_ocr_corrections

if TYPE_CHECKING:
    from gemynd.core.domain_config import DomainConfig


# ``get_ocr_flags`` is also called (with no vocabulary args) by the spelling-
# review queue and the junk-mention review detector, which are not yet
# domain-threaded. Those callers fall back to the Turnbull reference vocabulary
# *explicitly* via the cached helpers below — a visible, documented coupling,
# not a silent no-arg loader default. (Threading domain config through the
# spelling/review layer is a separate follow-up.) The mention extractor itself
# no longer uses these: it always constructs from a DomainConfig (see __init__).
def _turnbull_ocr_corrections() -> frozenset[str]:
    return default_ocr_corrections(TURNBULL_RESOURCES_DIR)


@functools.cache
def _turnbull_lexicon_hints() -> dict[str, list[str]]:
    hints: dict[str, list[str]] = {}
    for entity in default_seed_entities(TURNBULL_RESOURCES_DIR):
        hints.setdefault(entity.name.lower(), []).append(entity.entity_type)
    return hints


@dataclass(slots=True)
class MentionDraft:
    surface_form: str
    normalized_form: str
    start_offset: int
    end_offset: int
    detection_confidence: float
    ocr_flags: list[str]
    entity_type_hints: list[str] = field(default_factory=list)
    detection_source: str = "proper_noun"
    decision_trace: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass(slots=True)
class CandidateSpan:
    """Raw positional output of the detection pass.

    Carries no OCR flags and no entity type hints — those are
    classification-stage concerns resolved in _classify().
    """
    surface_form: str
    start_offset: int
    end_offset: int
    detection_confidence: float    # base confidence; bonus applied in _classify()
    detection_source: str          # "seed_lexicon" | "acronym" | "initialized_name" | "proper_noun"
    matched_patterns: list[str]
    fallback_used: bool
    fuzzy_hit: str | None = None   # populated for proper_noun fuzzy matches


@dataclass(slots=True)
class ResolutionContext:
    """Per-paragraph context threaded into the entity resolver.

    Built after the per-paragraph extraction loop in pipeline.py and keyed
    by paragraph_id in the contexts dict passed to DictionaryFuzzyResolver.resolve().
    """
    paragraph_id: str
    claim_types: list[str]    # deduplicated and sorted claim_type values for this paragraph
    resolved_entity_types: list[str] = field(default_factory=list)
    # Populated by DictionaryFuzzyResolver.resolve() during Pass 1 as an output.
    # Each entry is the entity_type of an entity resolved REFERS_TO in this paragraph.


class MentionExtractor(Protocol):
    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        ...


def _fuzzy_match_seed(text: str, lexicon_hints: dict[str, list[str]], threshold: float = 0.78) -> str | None:
    best_name: str | None = None
    best_ratio = 0.0
    for name in lexicon_hints:
        if abs(len(name) - len(text)) > 3:
            continue
        ratio = difflib.SequenceMatcher(None, text, name).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, name
    return best_name if best_ratio >= threshold else None


def get_ocr_flags(
    surface: str,
    *,
    known_ocr_errors: frozenset[str] | None = None,
    lexicon_hints: dict[str, list[str]] | None = None,
) -> list[str]:
    normalized = normalize_for_match(surface)
    known_errors = known_ocr_errors if known_ocr_errors is not None else _turnbull_ocr_corrections()
    hints = lexicon_hints if lexicon_hints is not None else _turnbull_lexicon_hints()

    flags: list[str] = []
    if normalized in known_errors:
        flags.append("ocr_suspect_list")

    if re.search(r"(?<=[a-z])[0-9]|[0-9](?=[a-z])", normalized):
        flags.append("digit_in_token")
        for digit, letter in [("0", "o"), ("1", "l")]:
            candidate = normalized.replace(digit, letter)
            if candidate in hints:
                flags.append(f"near_seed_term:{candidate}")

    if "rn" in normalized:
        candidate = normalized.replace("rn", "m")
        if candidate in hints:
            flags.append(f"near_seed_term:{candidate}")
        elif "m" not in normalized:
            flags.append("rn_m_confusion")

    if "li" in normalized:
        candidate = normalized.replace("li", "h")
        if candidate in hints:
            flags.append(f"near_seed_term:{candidate}")

    if not any(flag.startswith("near_seed_term") for flag in flags):
        best = _fuzzy_match_seed(normalized, hints)
        if best:
            flags.append(f"near_seed_term:{best}")

    return flags


class RuleBasedMentionExtractor:
    _STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "in", "at", "on", "of", "for", "with", "and", "or",
        "but", "to", "from", "by", "about", "during", "near", "after", "before",
        "that", "this", "these", "those", "it", "its",
    })

    # Stage 2: acronyms and initialized names.
    _acronym = re.compile(
        r"\b[A-Z]{2,}\b"
        r"|[A-Z]\.(?:\s*[A-Z]\.)+(?:\s+[A-Z][a-z]+)*"
    )
    _initialized_name = re.compile(
        r"\b(?:Dr|Mr|Mrs|Prof|Col|Lt|Sgt)\.?\s+(?:[A-Z]\.\s*)*[A-Z][a-z]+"
    )

    # Stage 3: titlecase spans (1+ words, min 3 chars).
    _titlecase_span = re.compile(r"\b([A-Z][a-z]{2,}(?:[ \t]+[A-Z][a-z]+)*)\b")

    def __init__(
        self,
        resources_dir: Path | None = None,
        config: "DomainConfig | None" = None,
    ) -> None:
        # Always operate on a DomainConfig. When none is supplied, load the named
        # resources dir (or the Turnbull corpus explicitly) — never a silent
        # default. This collapses the former config/resources/bare branches and
        # removes the silent module-level Turnbull lexicon caches.
        if config is None:
            from gemynd.core.domain_config import load_domain_config
            config = load_domain_config(
                resources_dir if resources_dir is not None else TURNBULL_RESOURCES_DIR
            )
        self._known_ocr_errors = config.ocr_corrections
        self._negative_forms = config.negative_lexicon
        hints: dict[str, list[str]] = {}
        for entity in config.seed_entities:
            hints.setdefault(entity.name.lower(), []).append(entity.entity_type)
        self._lexicon_hints = hints
        self._lexicon_terms = sorted(hints, key=len, reverse=True)

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        spans = self._detect(paragraph_text)
        spans = resolve_overlapping_spans(spans)
        return sorted(
            (self._classify(paragraph_text, span) for span in spans),
            key=lambda d: d.start_offset,
        )

    # ── Detection stage ───────────────────────────────────────────────────────

    def _detect(self, paragraph_text: str) -> list[CandidateSpan]:
        """Emit raw CandidateSpan objects from all three detection passes.

        No OCR flags or entity type hints are computed here — those are
        classification-stage concerns.  Confidence scores are base values;
        the near-seed-term bonus is applied in _classify().
        """
        spans: list[CandidateSpan] = []

        # Stage 1: seed lexicon scan.
        for surface, start, end in self._scan_terms(paragraph_text):
            if surface.lower() in self._negative_forms:
                continue
            spans.append(CandidateSpan(
                surface_form=surface,
                start_offset=start,
                end_offset=end,
                detection_confidence=0.92,
                detection_source="seed_lexicon",
                matched_patterns=[f"seed_lexicon:{surface.lower()}"],
                fallback_used=False,
            ))

        # Stage 2: acronyms and initialized names.
        for pattern, pattern_name in (
            (self._acronym, "acronym_re"),
            (self._initialized_name, "initialized_name_re"),
        ):
            source = "acronym" if pattern_name == "acronym_re" else "initialized_name"
            for match in pattern.finditer(paragraph_text):
                surface = match.group(0).strip()
                if len(surface) <= 1:
                    continue
                if surface.lower() in self._negative_forms:
                    continue
                spans.append(CandidateSpan(
                    surface_form=surface,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    detection_confidence=0.82,
                    detection_source=source,
                    matched_patterns=[pattern_name],
                    fallback_used=False,
                ))

        # Stage 3: titlecase spans.
        for match in self._titlecase_span.finditer(paragraph_text):
            surface = match.group(0).strip()
            words = surface.split()
            if words[0].lower() in self._STOPWORDS:
                continue
            if surface.lower() in self._negative_forms:
                continue
            normalized_surface = normalize_for_match(surface)
            fuzzy_hit: str | None = None
            if len(words) == 1 and self._is_sentence_initial(paragraph_text, match.start()):
                if normalized_surface not in self._known_ocr_errors:
                    fuzzy_hit = _fuzzy_match_seed(normalized_surface, self._lexicon_hints)
                    if not fuzzy_hit:
                        continue
            spans.append(CandidateSpan(
                surface_form=surface,
                start_offset=match.start(),
                end_offset=match.end(),
                detection_confidence=0.74,
                detection_source="proper_noun",
                matched_patterns=["titlecase_span"],
                fallback_used=fuzzy_hit is not None,
                fuzzy_hit=fuzzy_hit,
            ))

        return spans

    # ── Classification stage ──────────────────────────────────────────────────

    def _classify(self, paragraph_text: str, span: CandidateSpan) -> MentionDraft:
        """Materialise a full MentionDraft from a detected CandidateSpan.

        Computes OCR flags, entity type hints, applies the near-seed-term
        confidence bonus, and builds the audit trace.
        """
        surface = span.surface_form
        ocr_flags = self._get_ocr_flags(surface)
        near_seed_bonus = 0.05 if any(f.startswith("near_seed_term") for f in ocr_flags) else 0.0

        if span.detection_source == "seed_lexicon":
            entity_type_hints = self._lexicon_hints.get(surface.lower(), [])
            confidence = min(span.detection_confidence + near_seed_bonus, 1.0)
            trace = [
                "stage:seed_lexicon",
                f"entity_type:{','.join(entity_type_hints) or 'unknown'}",
                *ocr_flags,
            ]
        elif span.detection_source in ("acronym", "initialized_name"):
            entity_type_hints = []
            confidence = min(span.detection_confidence + near_seed_bonus, 0.92)
            trace = [f"stage:{span.detection_source}", *ocr_flags]
        else:  # proper_noun
            entity_type_hints = []
            confidence = min(span.detection_confidence + near_seed_bonus, 0.85)
            trace = ["stage:proper_noun"]
            if span.fuzzy_hit:
                trace.append(f"fuzzy_match:{span.fuzzy_hit}")
            trace.extend(ocr_flags)

        return MentionDraft(
            surface_form=surface,
            normalized_form=normalize_for_match(surface),
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            detection_confidence=confidence,
            ocr_flags=ocr_flags,
            entity_type_hints=entity_type_hints,
            detection_source=span.detection_source,
            decision_trace=trace,
            matched_patterns=list(span.matched_patterns),
            fallback_used=span.fallback_used,
        )

    def _scan_terms(self, paragraph_text: str) -> list[tuple[str, int, int]]:
        matches: list[tuple[str, int, int]] = []
        occupied: list[tuple[int, int]] = []
        for term in self._lexicon_terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", paragraph_text, flags=re.IGNORECASE):
                span = (match.start(), match.end())
                if any(not (span[1] <= left or span[0] >= right) for left, right in occupied):
                    continue
                occupied.append(span)
                matches.append((match.group(0), match.start(), match.end()))
        matches.sort(key=lambda row: row[1])
        return matches

    @staticmethod
    def _is_sentence_initial(paragraph_text: str, match_start: int) -> bool:
        prefix = paragraph_text[:match_start].rstrip()
        return not prefix or bool(re.search(r"[.!?]$", prefix))

    def _get_ocr_flags(self, surface: str) -> list[str]:
        return get_ocr_flags(
            surface,
            known_ocr_errors=self._known_ocr_errors,
            lexicon_hints=self._lexicon_hints,
        )

# ── GLiNER optional dependency (imported LAZILY) ───────────────────────────────
# Importing gliner pulls in transformers/torch (~2GB per process). The rule-based
# ingest path never uses GLiNER, so importing it at module load forced every
# ingest worker to load torch and OOM'd memory-constrained hosts. Check
# availability without importing (find_spec), and import GLiNER only when a
# GLiNERMentionAdapter is actually built.
import importlib.util as _importlib_util


def _gliner_available() -> bool:
    return _importlib_util.find_spec("gliner") is not None


def _import_gliner():
    from gliner import GLiNER  # type: ignore[import-untyped]
    return GLiNER


# ── Hybrid mention extractor ───────────────────────────────────────────────────

@dataclass
class HybridMentionTelemetry:
    """Counts of how rules and GLiNER contributed during a hybrid extraction.

    Mirrors HybridTelemetry in claim_extractor.py.
    Keys are (start_offset, end_offset) span tuples.
    """
    rules_only: list[tuple[int, int]]
    gliner_only: list[tuple[int, int]]
    overlapping: list[tuple[int, int]]
    confidence_deltas: list[tuple[tuple[int, int], float, float]]  # (key, rules_conf, gliner_conf)


class _NullMentionExtractor:
    """No-op extractor used when GLiNER is unavailable."""

    def extract(self, paragraph_text: str) -> list[MentionDraft]:  # noqa: ARG002
        return []


class GLiNERMentionAdapter:
    """Wraps a GLiNER model and adapts its output to list[MentionDraft].

    Entity labels are derived from seed_entities.csv entity_type values at
    construction time.  GLiNER returns paragraph-relative character offsets,
    which are already consistent with RuleBasedMentionExtractor offsets.

    Raises ImportError (with a helpful message) when gliner is not installed.
    """

    def __init__(
        self,
        model: object,
        entity_labels: list[str],
        threshold: float = 0.40,
    ) -> None:
        if not _gliner_available():
            raise ImportError(
                "GLiNER is not installed. Install it with: pip install gliner"
            )
        self._model = model
        self._entity_labels = entity_labels
        self._threshold = threshold

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "urchade/gliner_multi-v2.1",
        resources_dir: Path | None = None,
        threshold: float = 0.40,
    ) -> GLiNERMentionAdapter:
        if not _gliner_available():
            raise ImportError(
                "GLiNER is not installed. Install it with: pip install gliner"
            )
        model = _import_gliner().from_pretrained(model_name)
        seed = default_seed_entities(resources_dir or TURNBULL_RESOURCES_DIR)
        labels = sorted({e.entity_type for e in seed})
        return cls(model, labels, threshold)

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        entities = self._model.predict_entities(
            paragraph_text, self._entity_labels, threshold=self._threshold
        )
        drafts: list[MentionDraft] = []
        for ent in entities:
            surface: str = ent["text"]
            score: float = float(ent["score"])
            label: str = ent["label"]
            drafts.append(MentionDraft(
                surface_form=surface,
                normalized_form=normalize_for_match(surface),
                start_offset=int(ent["start"]),
                end_offset=int(ent["end"]),
                detection_confidence=round(score, 4),
                ocr_flags=[],
                entity_type_hints=[label],
                detection_source="gliner",
                decision_trace=[f"stage:gliner", f"label:{label}", f"score:{score:.4f}"],
                matched_patterns=["gliner_model"],
                fallback_used=False,
            ))
        return drafts


class HybridMentionExtractor:
    """Combines RuleBasedMentionExtractor with an optional GLiNER extractor.

    Mirrors the HybridClaimExtractor pattern from claim_extractor.py.
    The merger is keyed on (start_offset, end_offset):
    - Both agree on a span → higher confidence wins; winner inherits the
      other's ocr_flags / entity_type_hints if its own lists are empty.
    - GLiNER detects a span rules missed → include with detection_source="gliner".
    - Rules detect a span GLiNER missed → keep as-is.

    When no GLiNER extractor is provided (default), the output is identical to
    RuleBasedMentionExtractor.
    """

    def __init__(
        self,
        rules_extractor: MentionExtractor | None = None,
        gliner_extractor: MentionExtractor | None = None,
        resources_dir: Path | None = None,
        config: "DomainConfig | None" = None,
    ) -> None:
        self._rules: MentionExtractor = (
            rules_extractor if rules_extractor is not None
            else RuleBasedMentionExtractor(resources_dir=resources_dir, config=config)
        )
        self._gliner: MentionExtractor = (
            gliner_extractor if gliner_extractor is not None
            else _NullMentionExtractor()
        )
        self.last_telemetry: HybridMentionTelemetry | None = None

    def extract(self, paragraph_text: str) -> list[MentionDraft]:
        telemetry = HybridMentionTelemetry(
            rules_only=[], gliner_only=[], overlapping=[], confidence_deltas=[]
        )
        merged: dict[tuple[int, int], MentionDraft] = {}

        for draft in self._rules.extract(paragraph_text):
            merged[(draft.start_offset, draft.end_offset)] = draft

        for gliner_draft in self._gliner.extract(paragraph_text):
            key = (gliner_draft.start_offset, gliner_draft.end_offset)
            rule_draft = merged.get(key)

            if rule_draft is None:
                # GLiNER found a span rules missed.
                telemetry.gliner_only.append(key)
                merged[key] = gliner_draft
                continue

            rule_conf = rule_draft.detection_confidence
            gliner_conf = gliner_draft.detection_confidence
            telemetry.overlapping.append(key)
            telemetry.confidence_deltas.append((key, rule_conf, gliner_conf))

            if gliner_conf > rule_conf:
                # GLiNER wins: inherit OCR flags and type hints from rules if missing.
                if not gliner_draft.ocr_flags:
                    gliner_draft.ocr_flags = list(rule_draft.ocr_flags)
                if not gliner_draft.entity_type_hints:
                    gliner_draft.entity_type_hints = list(rule_draft.entity_type_hints)
                gliner_draft.detection_source = "hybrid"
                tag_hybrid_trace(
                    gliner_draft,
                    f"hybrid:gliner_won(conf={gliner_conf:.3f}>rule={rule_conf:.3f})",
                )
                merged[key] = gliner_draft
            else:
                # Rules win.
                rule_draft.detection_source = "hybrid"
                tag_hybrid_trace(
                    rule_draft,
                    f"hybrid:rules_won(conf={rule_conf:.3f}>={gliner_conf:.3f})",
                )

        for key in merged:
            if merged[key].detection_source not in ("hybrid", "gliner"):
                telemetry.rules_only.append(key)

        self.last_telemetry = telemetry
        return sorted(merged.values(), key=lambda d: d.start_offset)
