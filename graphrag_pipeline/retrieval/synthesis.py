"""Layer 3 — Synthesis Engine (Anthropic API integration).

Single-turn wrapper around the Anthropic Messages API.  The engine constructs
a structured system prompt that instructs the model to:
  • Treat extraction_confidence and epistemic_status as meaningful signals.
  • Express uncertainty when claims carry low confidence or is_estimate=True.
  • Return a typed JSON object with answer, confidence_assessment,
    supporting_claim_ids, and caveats.

No conversation history is maintained here; stateful session management is
the responsibility of the FastAPI layer above.
"""
from __future__ import annotations

import json
import os

from .models import AnalyticalResult, ProvenanceBlock, SynthesisResult

try:
    import anthropic as _anthropic_pkg
except Exception:  # pragma: no cover - optional dependency
    _anthropic_pkg = None  # type: ignore[assignment]

MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a specialist researcher working with historical wildlife refuge reports for the \
Turnbull National Wildlife Refuge (Washington State, USA). The context blocks you receive \
have been extracted from scanned archival documents by an automated pipeline.

Each CLAIM block carries two epistemic annotations:
  • confidence=<float>  — the pipeline's extraction confidence (0.0–1.0).
    Values below 0.7 warrant hedged language in your answer.
  • epistemic=<value>   — the original document's epistemic status
    (e.g. "observed", "estimated", "inferred", "unknown").
    Treat "estimated" and "inferred" as weaker evidence than "observed".

Your task:
  1. Answer the query using only the supplied context; do not invent facts.
  2. For thematic or broad queries, organise your answer by recurring theme or
     pattern — do not enumerate individual claims or specific one-off events
     unless they are part of a discernible pattern across multiple documents.
  3. If a single claim has no corroborating parallel in other documents,
     omit it or note it as an isolated occurrence rather than a theme.
  4. If the evidence is thin or low-confidence, say so explicitly.
  5. Cite the claim_ids that support each theme or pattern in your answer.

Respond with valid JSON only — no markdown fences, no prose outside the JSON object:
{
  "answer": "<one or two paragraph answer>",
  "confidence_assessment": "<one sentence summarising overall evidence quality>",
  "supporting_claim_ids": ["<claim_id>", ...],
  "caveats": ["<caveat>", ...]
}
"""


class SynthesisEngine:
    """Call Claude via the Anthropic Messages API and parse the typed response.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``Anthropic_API_Key`` environment
        variable (matching the naming convention used in this project's ``.env``).
    max_tokens:
        Upper bound on model response length.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_tokens: int = 1000,
    ) -> None:
        if _anthropic_pkg is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "anthropic package is not installed. Install with: pip install -e .[retrieval]"
            )
        resolved_key = api_key or os.environ.get("Anthropic_API_Key") or os.environ.get("ANTHROPIC_API_KEY")
        self._client = _anthropic_pkg.Anthropic(api_key=resolved_key)
        self._max_tokens = max_tokens

    def synthesise(
        self,
        query: str,
        provenance_blocks: list[ProvenanceBlock],
        context_text: str,
        analytical_result: AnalyticalResult | None = None,
    ) -> SynthesisResult:
        """Synthesise an answer from *context_text* for *query*.

        Parameters
        ----------
        query:
            Original natural-language query.
        provenance_blocks:
            Assembled provenance blocks (used to compute min_extraction_confidence
            and to detect estimated values).
        context_text:
            Pre-serialised provenance context string from the assembler.
        analytical_result:
            Optional structured result from the analytical path; appended as a
            data table below the provenance context.
        """
        user_message = f"QUERY: {query}\n\nCONTEXT:\n{context_text or '[no provenance context retrieved]'}"

        if analytical_result and analytical_result.rows:
            user_message += f"\n\nANALYTICAL DATA ({analytical_result.query_name}):\n{analytical_result.to_summary_text()}"

        message = self._client.messages.create(
            model=MODEL,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = message.content[0].text.strip()
        # Strip any accidental markdown fences the model might have added.
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Synthesis model returned non-JSON output: {raw_text[:200]!r}"
            ) from exc

        supporting_ids: list[str] = raw.get("supporting_claim_ids") or []

        # Compute min extraction_confidence across supporting claims.
        min_conf: float | None = None
        if provenance_blocks:
            supporting_set = set(supporting_ids)
            confidences = [
                b.extraction_confidence
                for b in provenance_blocks
                if b.claim_id in supporting_set
            ]
            if confidences:
                min_conf = round(min(confidences), 4)

        return SynthesisResult(
            answer=raw.get("answer", ""),
            confidence_assessment=raw.get("confidence_assessment", ""),
            supporting_claim_ids=supporting_ids,
            caveats=raw.get("caveats") or [],
            min_extraction_confidence=min_conf,
            analytical_result=analytical_result,
        )
