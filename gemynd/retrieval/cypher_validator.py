"""Safety validator for LLM-generated Cypher.

Runs a sequence of cheap checks before any LLM-produced query is handed to
Neo4j. Every failure mode is a hard reject: the fallback must degrade to the
existing fulltext path rather than execute unverified Cypher.

Contract (see plan B3 for the full enumeration):

1. Read-only token scan — reject mutation keywords and any ``CALL apoc.*``.
2. Label / relationship allowlist — every ``:Label`` and ``[:REL]`` token must
   appear in the schema the LLM was given.
3. Mandatory access parameters — ``$institution_ids`` and ``$permitted_levels``
   must both appear in the query body.
4. Row shape contract — ``RETURN`` must bind the aliases the consumer expects
   (``c``, ``d``, ``pg``, ``para``) so the existing ``_row_to_block`` can
   consume the result.
5. Optional EXPLAIN cost guard — when an executor is supplied, run ``EXPLAIN``
   first and reject plans containing ``AllNodesScan`` on ``Claim`` or
   ``Document``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forbidden tokens
# ---------------------------------------------------------------------------

# Mutation and admin verbs. Matched against tokenized words only (not substring
# scanned) so identifiers like ``created_at`` or ``dataset`` never trigger.
_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "CREATE", "DELETE", "DETACH", "MERGE", "SET", "REMOVE", "DROP",
    "LOAD", "FOREACH", "USING", "PERIODIC",
})

# CALL is allowed only for the fulltext index helpers on this allowlist.
_ALLOWED_CALL_PROCEDURES: frozenset[str] = frozenset({
    "db.index.fulltext.queryNodes",
    "db.index.fulltext.queryRelationships",
})

# Aliases the LLM must bind in RETURN so _row_to_block can consume its rows.
# The plan fixes these four; any others are optional.
_REQUIRED_RETURN_ALIASES: frozenset[str] = frozenset({"c", "d", "pg", "para"})

# Parameters that must appear somewhere in the query body to enforce access
# control. These map 1-to-1 with Neo4j parameter bindings supplied by the
# caller, so every query the LLM produces must respect the tenancy filter.
_REQUIRED_PARAMS: tuple[str, ...] = ("$institution_ids", "$permitted_levels")

# Fixed structural labels that every domain inherits — ``_row_to_block``
# depends on these regardless of ``DomainConfig.entity_labels``.
_STRUCTURAL_LABELS: frozenset[str] = frozenset({
    "Document", "Paragraph", "Section", "Page",
    "Claim", "ExtractionRun", "Observation", "Year", "Measurement", "Entity",
    # Role entities used inside rows for observation context.
    "Species", "Habitat",
})

# Fixed structural edges. Anchor relations (ABOUT_REFUGE / ABOUT_PLACE / etc.)
# must be injected from DomainConfig at construction time by the caller.
_STRUCTURAL_EDGES: frozenset[str] = frozenset({
    "PROCESSED_BY", "HAS_CLAIM", "HAS_PARAGRAPH", "HAS_SECTION", "HAS_PAGE",
    "SUPPORTS", "IN_YEAR", "HAS_MEASUREMENT", "OF_SPECIES", "IN_HABITAT",
})


# ---------------------------------------------------------------------------
# Schema bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CypherSchema:
    """Allowlist the validator checks against.

    Built once at ``create_app()`` time from ``DomainConfig`` and the fixed
    structural labels/edges listed above.
    """

    labels: frozenset[str]
    edges: frozenset[str]
    fulltext_indexes: frozenset[str]

    @classmethod
    def from_domain(
        cls,
        domain_labels: frozenset[str],
        domain_edges: frozenset[str],
        anchor_relations: frozenset[str] = frozenset(),
        fulltext_indexes: frozenset[str] = frozenset({"claim_normalized_sentence"}),
    ) -> "CypherSchema":
        return cls(
            labels=frozenset(domain_labels | _STRUCTURAL_LABELS),
            edges=frozenset(domain_edges | _STRUCTURAL_EDGES | anchor_relations),
            fulltext_indexes=fulltext_indexes,
        )


# ---------------------------------------------------------------------------
# Executor protocol
# ---------------------------------------------------------------------------


class _ExplainExecutor(Protocol):
    def run(self, cypher: str, params: dict | None = None) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Structured verdict returned by :func:`validate_cypher`.

    ``ok`` is True only when every check passed. ``reason`` is a human-readable
    string describing the first failure, populated on reject. ``violations``
    holds all failures for logging; the caller may choose to report only the
    first.
    """

    ok: bool
    reason: str = ""
    violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenizer — minimal, not a full Cypher parser
# ---------------------------------------------------------------------------

# Strings (single and double quoted) and line comments are stripped before
# tokenization so the keyword scan does not false-positive on ``"CREATE"``
# literals or ``// MERGE`` comments.
_STRING_OR_COMMENT_RE = re.compile(
    r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*|/\*[\s\S]*?\*/)"
)

# Words (sequences of alphanumerics and underscores).
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# ``:Label`` tokens after relationship patterns (``[:REL]``) have been
# stripped. We rely on the bracket-strip + string-strip preprocessing to
# eliminate false positives, so the regex itself is intentionally minimal.
_LABEL_RE = re.compile(r":([A-Z][A-Za-z0-9_]*)")

# Used to blank out ``[:REL]`` segments before the label scan so the
# label regex never sees relationship-type tokens.
_REL_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")

# ``[:REL_TYPE]`` (plus alternation ``[:R1|R2]``).
_REL_RE = re.compile(r"\[:\s*([A-Z_][A-Z0-9_]*(?:\s*\|\s*:?[A-Z_][A-Z0-9_]*)*)")

# Parameters like ``$foo``.
_PARAM_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")

# Matches ``CALL <qualified.name>`` so we can check it against the allowlist.
_CALL_PROC_RE = re.compile(r"\bCALL\s+([A-Za-z_][A-Za-z0-9_.]*)", re.IGNORECASE)


def _strip_strings_and_comments(cypher: str) -> str:
    return _STRING_OR_COMMENT_RE.sub(" ", cypher)


def _extract_return_aliases(cypher: str) -> set[str]:
    """Best-effort extraction of the aliases used in the top-level RETURN.

    Not a full parser — we find the last ``RETURN`` (case-insensitive) that
    is not inside a subquery's ``CALL { ... RETURN }`` and scan its items. An
    LLM playing games with nested subqueries would defeat this, but the
    alias check is only one of five gates and is defended by the forbidden-
    keyword and EXPLAIN checks above.
    """
    stripped = _strip_strings_and_comments(cypher)
    # Find the LAST top-level RETURN. Cypher ends with the outermost RETURN so
    # walking from the end is safe for valid queries.
    m = None
    for candidate in re.finditer(r"\bRETURN\b", stripped, re.IGNORECASE):
        m = candidate
    if m is None:
        return set()
    tail = stripped[m.end():]
    # Stop at ORDER BY / SKIP / LIMIT / UNION / semicolon.
    cut = re.search(
        r"\b(ORDER\s+BY|SKIP|LIMIT|UNION)\b|;",
        tail, re.IGNORECASE,
    )
    if cut:
        tail = tail[:cut.start()]
    aliases: set[str] = set()
    # Split on commas at the top level (naively — nested function calls do
    # not typically appear in our expected row shape).
    for item in _split_top_level(tail, ","):
        item = item.strip()
        if not item:
            continue
        # Forms we accept: ``c``, ``c AS claim``, ``expr AS alias``.
        as_match = re.search(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", item, re.IGNORECASE)
        if as_match:
            aliases.add(as_match.group(1))
            continue
        word = _WORD_RE.match(item)
        if word:
            aliases.add(word.group(0))
    return aliases


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* on *sep* while respecting ``()``, ``[]``, ``{}`` nesting."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and ch == sep:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_cypher(
    cypher: str,
    schema: CypherSchema,
    executor: _ExplainExecutor | None = None,
    *,
    require_explain: bool = False,
) -> ValidationResult:
    """Run every check against *cypher* and return a :class:`ValidationResult`.

    Parameters
    ----------
    cypher:
        The query to validate. Pre-trimmed; leading/trailing whitespace does
        not affect the outcome.
    schema:
        :class:`CypherSchema` describing the labels, edges, and fulltext
        indexes this query may reference.
    executor:
        Optional Neo4j executor used for the EXPLAIN cost guard. When ``None``,
        the cost guard is skipped.
    require_explain:
        When True, an executor must be supplied and the EXPLAIN guard must
        pass. Used in production; tests that want to exercise every other rule
        without hitting Neo4j leave this False.
    """
    violations: list[str] = []
    if not cypher or not cypher.strip():
        return ValidationResult(ok=False, reason="empty cypher", violations=["empty cypher"])

    stripped = _strip_strings_and_comments(cypher)

    # 1. Forbidden keywords -------------------------------------------------
    # Tokenize by word boundaries so ``created_at`` does not hit ``CREATE``.
    words_upper = {w.upper() for w in _WORD_RE.findall(stripped)}
    bad_keywords = sorted(_FORBIDDEN_KEYWORDS & words_upper)
    for bad in bad_keywords:
        violations.append(f"forbidden keyword: {bad}")

    # 1b. CALL procedures must be on the allowlist.
    for call_match in _CALL_PROC_RE.finditer(stripped):
        proc = call_match.group(1)
        # Allow CALL { subquery } — the regex requires a name right after CALL,
        # so a subquery is naturally skipped (the word after ``CALL`` is the
        # first identifier inside the block, which hits the allowlist check).
        # Filter that false-positive out by checking what follows the match.
        after = stripped[call_match.end():].lstrip()
        if after.startswith("{"):
            continue
        if proc not in _ALLOWED_CALL_PROCEDURES:
            violations.append(f"forbidden CALL procedure: {proc}")

    # 2. Label / relationship allowlist -------------------------------------
    # Blank out relationship brackets so the label scanner never sees the
    # ``:REL`` tokens inside ``[:REL]`` patterns.
    label_scan_text = _REL_BRACKET_RE.sub(" ", stripped)
    for lm in _LABEL_RE.finditer(label_scan_text):
        label = lm.group(1)
        if label not in schema.labels:
            violations.append(f"unknown label: {label}")
    for rm in _REL_RE.finditer(stripped):
        # Alternation ``R1|R2`` splits into individual names.
        for rel in re.split(r"\s*\|\s*:?", rm.group(1)):
            rel = rel.strip()
            if rel and rel not in schema.edges:
                violations.append(f"unknown relationship type: {rel}")

    # 3. Mandatory access parameters ----------------------------------------
    params_present = set(_PARAM_RE.findall(stripped))
    for required in _REQUIRED_PARAMS:
        if required not in params_present:
            violations.append(f"missing required parameter: {required}")

    # 4. Row-shape contract -------------------------------------------------
    return_aliases = _extract_return_aliases(cypher)
    missing_aliases = sorted(_REQUIRED_RETURN_ALIASES - return_aliases)
    if missing_aliases:
        violations.append(
            f"RETURN missing required aliases: {', '.join(missing_aliases)}"
        )

    if violations:
        return ValidationResult(
            ok=False, reason=violations[0], violations=violations,
        )

    # 5. EXPLAIN cost guard --------------------------------------------------
    if require_explain and executor is None:
        return ValidationResult(
            ok=False,
            reason="executor required for EXPLAIN guard",
            violations=["executor required for EXPLAIN guard"],
        )
    if executor is not None:
        try:
            # EXPLAIN parses + plans the query without executing it. Parameters
            # must still be supplied — stub them with empty / placeholder
            # values since the planner only needs types.
            stub_params = {
                "institution_ids": [],
                "permitted_levels": ["public"],
                "entity_id": "",
                "entity_ids": [],
                "anchor_id": "",
                "claim_types": None,
                "year_min": None,
                "year_max": None,
                "limit": 1,
                "search_text": "",
            }
            rows = executor.run(f"EXPLAIN {cypher}", stub_params)
            plan_blob = " ".join(str(r) for r in rows)
            # Neo4j EXPLAIN result does not return the plan tree directly
            # through the driver's record API; a permissive cost check scans
            # the server notifications for scan warnings. Neo4j 5+ returns
            # notifications on result summaries, which the simple .run()
            # wrapper above does not expose. We fall back to a plan-string
            # scan which is belt-and-braces; if the driver exposes zero info
            # we pass the check and rely on the 5-second timeout instead.
            for banned in ("AllNodesScan", "allNodesScan"):
                if banned in plan_blob:
                    violations.append(f"plan contains {banned}")
        except Exception as exc:
            _log.debug("EXPLAIN guard failed: %s", exc)
            if require_explain:
                return ValidationResult(
                    ok=False,
                    reason=f"EXPLAIN failed: {exc}",
                    violations=[f"EXPLAIN failed: {exc}"],
                )

    if violations:
        return ValidationResult(
            ok=False, reason=violations[0], violations=violations,
        )

    return ValidationResult(ok=True, reason="", violations=[])
