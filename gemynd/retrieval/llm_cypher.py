"""Text-to-Cypher LLM fallback (plan B2 + B6).

When the rule-based router and its full relaxation chain return zero rows,
the assembler asks :class:`LLMCypherGenerator` for a single read-only Cypher
query that targets the same row shape as the parameterised templates. The
generated query is run through :func:`gemynd.retrieval.cypher_validator.validate_cypher`
before any execution; failures degrade silently to fulltext rather than
raising into the response path.

This module is intentionally narrow:

* Construction is cheap — the schema-aware system prompt is built once per
  process from the live ``DomainConfig`` and stashed on the instance.
* :meth:`generate` returns ``str | None``. ``None`` means "the model declined,
  the validator rejected the output, or the LLM call errored" — the caller
  must treat all three identically and fall back to fulltext.
* The class never executes Cypher itself. Execution is always the assembler's
  responsibility, after a final validation pass.

The cache (B4) is owned by the caller; this class accepts an optional cache
handle and probes it before issuing the API call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from gemynd.core.domain_config import CompositeDomainConfig, DomainConfig

from .cypher_validator import (
    CypherSchema,
    ValidationResult,
    validate_cypher,
)
from .models import QueryIntent

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache protocol — kept structural so test fakes don't need a base class.
# ---------------------------------------------------------------------------


class _CypherCache(Protocol):
    def get(self, domain_version: str, normalized_query: str) -> str | None: ...
    def put(self, domain_version: str, normalized_query: str, cypher: str) -> None: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic model used for Cypher generation. Sonnet is the right cost/quality
# balance — Opus is overkill for this constrained generation task and Haiku
# trips on the schema injection more than is acceptable for the safety story.
_DEFAULT_MODEL = "claude-sonnet-4-6"

# Token cap for the model response. The expected output is a single Cypher
# query (~600 chars) plus optional reasoning that we strip; 1024 is plenty
# without leaving room for the model to wander.
_MAX_TOKENS = 1024

# Strip leading/trailing markdown fences (```cypher … ```) the model often
# emits despite explicit "no fences" instructions.
_FENCE_RE = re.compile(r"^\s*```(?:cypher|sql)?\s*|\s*```\s*$", re.IGNORECASE)

# Heuristic: collapse whitespace before hashing the query for cache lookup
# so trivial spacing differences don't blow the cache.
_WS_RE = re.compile(r"\s+")

# Hard rules block — appended verbatim to the system prompt. Numbered so the
# validator's rejection messages can reference the rule that fired.
_HARD_RULES = """\
HARD RULES (every query you produce MUST satisfy these — violations are rejected by an automated validator):

1. READ ONLY. You must never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD, FOREACH, USING, PERIODIC, or any apoc.* procedure.
2. ACCESS CONTROL. The query body MUST reference the parameters $institution_ids and $permitted_levels at least once each. Filter Document nodes by `d.institution_id IN $institution_ids AND d.access_level IN $permitted_levels`.
3. LATEST RUN. Filter to the most recent extraction run for every Document, mirroring the templates: `MATCH (d:Document)-[:PROCESSED_BY]->(r:ExtractionRun) WITH d, max(r.run_timestamp) AS latest_run MATCH (d)-[:PROCESSED_BY]->(r2:ExtractionRun {run_timestamp: latest_run})`.
4. ROW SHAPE. The final RETURN MUST bind these exact aliases: `c, d, pg, para`. You may also bind `obs, sp, y, measurements` as optional fields. The downstream consumer reads only those names — any other shape is rejected.
5. LABEL & RELATIONSHIP ALLOWLIST. Use only the labels and relationship types listed in this prompt. Anything else is rejected.
6. CLAIM TYPES. When filtering by `c.claim_type`, use only values from the allowed_claim_types enum below.
7. NO FREE-TEXT SCANS. Do not scan all Claim or Document nodes. Anchor every query on either an entity match (via $entity_id / $entity_ids), a corpus anchor relation, or the fulltext index `claim_normalized_sentence` via `CALL db.index.fulltext.queryNodes`.
8. PARAMETERIZE. Never inline user-supplied strings or numbers. Use parameters: $entity_id, $entity_ids, $year_min, $year_max, $claim_types, $search_text, $limit. The validator does not check parameter binding correctness, but inlined strings are a query-injection bug.
9. LIMIT. Always end with `LIMIT $limit` (the assembler supplies a sane integer).
10. OUTPUT FORMAT. Reply with the Cypher query and nothing else — no prose, no markdown fences, no explanation. If you cannot answer the question with a query that satisfies these rules, reply with the single token: NONE.
"""


# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------


def _render_schema_block(
    domain_labels: frozenset[str],
    structural_labels: frozenset[str],
    domain_edges: frozenset[str],
    structural_edges: frozenset[str],
    anchor_relations: frozenset[str],
    allowed_claim_types: frozenset[str],
    fulltext_indexes: frozenset[str],
) -> str:
    """Build the schema description the LLM sees inside the system prompt.

    Sorted for determinism so the prompt cache is stable across restarts.
    """
    lines: list[str] = ["GRAPH SCHEMA"]
    lines.append("")
    lines.append("Allowed node labels (use only these in `:Label` patterns):")
    lines.append(
        "  structural: " + ", ".join(sorted(structural_labels))
    )
    if domain_labels:
        lines.append("  domain:     " + ", ".join(sorted(domain_labels)))
    lines.append("")
    lines.append("Allowed relationship types (use only these in `[:REL]` patterns):")
    lines.append(
        "  structural: " + ", ".join(sorted(structural_edges))
    )
    if domain_edges:
        lines.append("  domain:     " + ", ".join(sorted(domain_edges)))
    if anchor_relations:
        lines.append(
            "  anchors:    " + ", ".join(sorted(anchor_relations))
        )
    lines.append("")
    if allowed_claim_types:
        lines.append("Allowed values for c.claim_type (the claim type enum):")
        # Wrap at ~80 chars for prompt readability.
        chunk: list[str] = []
        width = 0
        for ct in sorted(allowed_claim_types):
            piece = ct
            if width + len(piece) + 2 > 80 and chunk:
                lines.append("  " + ", ".join(chunk))
                chunk = []
                width = 0
            chunk.append(piece)
            width += len(piece) + 2
        if chunk:
            lines.append("  " + ", ".join(chunk))
        lines.append("")
    if fulltext_indexes:
        lines.append(
            "Fulltext indexes available to db.index.fulltext.queryNodes: "
            + ", ".join(sorted(fulltext_indexes))
        )
        lines.append("")
    return "\n".join(lines)


# A worked example pinned in the prompt — the model latches onto the row shape
# and the latest-run pattern much more reliably with a concrete sample than
# with prose alone. Kept abstract so it works across domains.
_SHAPE_EXEMPLAR = """\
SHAPE EXEMPLAR (an entity-anchored query with the required row shape):

MATCH (e:Entity {entity_id: $entity_id})<-[:REFERS_TO|POSSIBLY_REFERS_TO]-(c:Claim)
MATCH (para:Paragraph)-[:HAS_CLAIM]->(c)
MATCH (pg:Page)-[:HAS_PARAGRAPH]->(para)
MATCH (d:Document)-[:HAS_PAGE]->(pg)
MATCH (d)-[:PROCESSED_BY]->(r:ExtractionRun)
WITH d, c, para, pg, max(r.run_timestamp) AS latest_run
MATCH (d)-[:PROCESSED_BY]->(r2:ExtractionRun {run_timestamp: latest_run})
WHERE d.institution_id IN $institution_ids
  AND d.access_level   IN $permitted_levels
  AND ($claim_types IS NULL OR c.claim_type IN $claim_types)
OPTIONAL MATCH (c)-[:SUPPORTS]->(obs:Observation)
OPTIONAL MATCH (obs)-[:OF_SPECIES]->(sp:Species)
OPTIONAL MATCH (obs)-[:IN_YEAR]->(y:Year)
OPTIONAL MATCH (obs)-[:HAS_MEASUREMENT]->(m:Measurement)
WITH c, d, pg, para, obs, sp, y, collect(m) AS measurements
RETURN c, d, pg, para, obs, sp, y, measurements
ORDER BY c.extraction_confidence DESC
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class LLMCypherGenerator:
    """Constrained text-to-Cypher generator with mandatory output validation.

    Construction is cheap — the schema-aware system prompt is built once and
    reused for every call. The class is safe to share across requests; it
    keeps no per-call state beyond the cache it was given.

    Parameters
    ----------
    client:
        :class:`MeteredAnthropicClient` (per gemynd convention, all Anthropic
        calls flow through the metered wrapper). May be ``None`` to disable
        the LLM path entirely — :meth:`generate` then always returns ``None``.
    domain_config:
        :class:`DomainConfig` or :class:`CompositeDomainConfig` describing the
        labels, edges, claim types, and anchors the LLM is allowed to use.
        The schema is captured by reference at construction; if the underlying
        domain mutates after construction the prompt is *not* refreshed.
    schema:
        Pre-built :class:`CypherSchema` used by the validator. The same schema
        feeds the prompt and the validator so the LLM cannot legally produce a
        query that the validator would later reject for unknown labels.
    cache:
        Optional ``(get, put)`` cache. Hits skip the API call entirely.
    model:
        Override the default Anthropic model id.
    """

    def __init__(
        self,
        client: Any | None,
        domain_config: DomainConfig | CompositeDomainConfig,
        schema: CypherSchema,
        cache: _CypherCache | None = None,
        *,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._domain = domain_config
        self._schema = schema
        self._cache = cache
        self._model = model
        self._domain_version = self._compute_domain_version(domain_config, schema)
        self._system_prompt = self._build_system_prompt(domain_config, schema)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query_text: str,
        intent: QueryIntent | None = None,
    ) -> str | None:
        """Return a validated Cypher query for *query_text* or ``None``.

        ``None`` covers all failure modes:
          * client not configured (LLM path disabled)
          * cache hit that no longer validates against the current schema
          * the model returned NONE / empty / non-Cypher
          * the validator rejected the model's output
          * the API call raised
        """
        if self._client is None:
            return None
        if not query_text or not query_text.strip():
            return None

        normalized = self._normalize_query(query_text)

        # Cache lookup. Re-validate on hit — schema may have widened or
        # narrowed since the entry was written.
        if self._cache is not None:
            cached = None
            try:
                cached = self._cache.get(self._domain_version, normalized)
            except Exception as exc:
                _log.debug("cypher cache get failed: %s", exc)
            if cached:
                verdict = validate_cypher(cached, self._schema)
                if verdict.ok:
                    return cached
                _log.info(
                    "cypher cache hit failed re-validation: %s", verdict.reason
                )

        prompt = self._build_user_prompt(query_text, intent)
        try:
            cypher = self._call_model(prompt)
        except Exception as exc:
            _log.warning("LLM cypher call failed: %s", exc)
            return None

        if not cypher:
            return None

        verdict = validate_cypher(cypher, self._schema)
        if not verdict.ok:
            _log.info(
                "LLM cypher rejected by validator: %s (full=%s)",
                verdict.reason,
                verdict.violations,
            )
            return None

        if self._cache is not None:
            try:
                self._cache.put(self._domain_version, normalized, cypher)
            except Exception as exc:
                _log.debug("cypher cache put failed: %s", exc)

        return cypher

    @property
    def system_prompt(self) -> str:
        """Expose the cached system prompt — used by tests and diagnostics."""
        return self._system_prompt

    @property
    def domain_version(self) -> str:
        return self._domain_version

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_model(self, user_prompt: str) -> str | None:
        """Issue the Anthropic API call and return the cleaned Cypher string."""
        message = self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Anthropic content is a list of blocks; we only expect one text block
        # because the system prompt forbids any other shape.
        try:
            raw_text = message.content[0].text
        except (AttributeError, IndexError):
            return None
        cleaned = self._extract_cypher(raw_text)
        if not cleaned:
            return None
        if cleaned.strip().upper() == "NONE":
            return None
        return cleaned

    @staticmethod
    def _extract_cypher(raw: str) -> str:
        """Strip markdown fences and surrounding prose, leaving raw Cypher."""
        text = raw.strip()
        if not text:
            return ""
        # Handle fenced ```cypher … ``` blocks: keep the content inside the
        # first fence, drop everything else.
        if "```" in text:
            parts = text.split("```")
            # Pick the longest non-empty fenced segment as the most likely
            # Cypher body (model sometimes wraps the query in fences and
            # follows up with prose outside the fence).
            candidates = [p.strip() for p in parts if p.strip()]
            if candidates:
                # The very first candidate after a fence may start with a
                # language tag like 'cypher' on its own line; strip that.
                best = max(candidates, key=len)
                if best.lower().startswith("cypher\n"):
                    best = best[len("cypher\n"):]
                elif best.lower().startswith("sql\n"):
                    best = best[len("sql\n"):]
                text = best
        text = _FENCE_RE.sub("", text).strip()
        return text

    @staticmethod
    def _normalize_query(query_text: str) -> str:
        return _WS_RE.sub(" ", query_text.strip().lower())

    @classmethod
    def _compute_domain_version(
        cls,
        domain: DomainConfig | CompositeDomainConfig,
        schema: CypherSchema,
    ) -> str:
        """Derive a stable cache key prefix from the schema.

        Schema changes invalidate cached cypher because the validator's
        allowlist (and therefore the set of legal queries) shifts. We hash
        the sorted union of labels + edges + claim types — a coarse but
        sufficient proxy for "the LLM would now produce different output".
        """
        import hashlib

        try:
            allowed_types = sorted(domain.allowed_claim_types)
        except AttributeError:
            allowed_types = []
        bits = (
            "|".join(sorted(schema.labels))
            + "##"
            + "|".join(sorted(schema.edges))
            + "##"
            + "|".join(allowed_types)
        )
        return hashlib.sha1(bits.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _build_system_prompt(
        cls,
        domain: DomainConfig | CompositeDomainConfig,
        schema: CypherSchema,
    ) -> str:
        """Assemble the cached system prompt from schema + hard rules."""
        from .cypher_validator import _STRUCTURAL_LABELS, _STRUCTURAL_EDGES

        # Domain-specific labels are everything in the schema that isn't part
        # of the fixed structural set. Same for edges.
        domain_labels = frozenset(schema.labels - _STRUCTURAL_LABELS)
        domain_edges = frozenset(schema.edges - _STRUCTURAL_EDGES)

        # Anchor relations: read straight off the domain when available so
        # they get their own labelled bucket in the prompt. Failing that, the
        # domain edges union already covers them.
        anchor_relations: set[str] = set()
        members: list[DomainConfig]
        if isinstance(domain, CompositeDomainConfig):
            members = list(domain.members)
        else:
            members = [domain]
        for m in members:
            rel = m.anchor_relation
            if rel:
                anchor_relations.add(rel)

        try:
            allowed_claim_types = frozenset(domain.allowed_claim_types)
        except AttributeError:
            allowed_claim_types = frozenset()

        schema_block = _render_schema_block(
            domain_labels=domain_labels,
            structural_labels=_STRUCTURAL_LABELS,
            domain_edges=domain_edges - frozenset(anchor_relations),
            structural_edges=_STRUCTURAL_EDGES,
            anchor_relations=frozenset(anchor_relations),
            allowed_claim_types=allowed_claim_types,
            fulltext_indexes=schema.fulltext_indexes,
        )

        return (
            "You translate a single natural-language question about an "
            "archival knowledge graph into one read-only Neo4j Cypher query. "
            "The graph stores extracted CLAIMS linked to the DOCUMENTS, "
            "PARAGRAPHS, PAGES, and ENTITIES they came from. Your job is to "
            "produce a query whose result rows match the row shape used by "
            "the rest of the retrieval pipeline.\n\n"
            + schema_block
            + "\n"
            + _SHAPE_EXEMPLAR
            + "\n"
            + _HARD_RULES
        )

    def _build_user_prompt(
        self,
        query_text: str,
        intent: QueryIntent | None,
    ) -> str:
        """Per-call user message — short, structured, no chain-of-thought ask."""
        # Reuse the prompt-injection scrub from the assembler so a hostile
        # query string can't smuggle instructions into the LLM prompt.
        from .context_assembler import _PROMPT_INJECTION_RE

        if _PROMPT_INJECTION_RE.search(query_text):
            safe_query = "[QUERY REDACTED: potential prompt injection]"
        else:
            safe_query = query_text.strip()

        hints: list[str] = []
        if intent is not None:
            if intent.entities:
                hints.append("entities: " + ", ".join(intent.entities[:8]))
            if intent.year_min is not None or intent.year_max is not None:
                hints.append(
                    f"year_range: {intent.year_min or '?'}–{intent.year_max or '?'}"
                )
            if intent.claim_types:
                hints.append("claim_types: " + ", ".join(intent.claim_types[:8]))
            if intent.bucket:
                hints.append(f"bucket: {intent.bucket}")
        hint_block = ("\n\nHINTS (from upstream classifier):\n  " + "\n  ".join(hints)) if hints else ""

        return (
            "QUESTION:\n"
            f"  {safe_query}"
            + hint_block
            + "\n\nReturn one Cypher query that respects every HARD RULE, "
            "or the single token NONE if no compliant query can answer this question."
        )


# ---------------------------------------------------------------------------
# Tiny in-process cache implementation
# ---------------------------------------------------------------------------


class InMemoryCypherCache:
    """Trivial dict-backed cache. Used by tests; production wires the SQLite
    store from :mod:`gemynd.shared.database_manager` instead.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get(self, domain_version: str, normalized_query: str) -> str | None:
        return self._store.get((domain_version, normalized_query))

    def put(self, domain_version: str, normalized_query: str, cypher: str) -> None:
        self._store[(domain_version, normalized_query)] = cypher

    def clear(self) -> None:
        self._store.clear()
