"""graphrag_pipeline.retrieval — natural-language retrieval layer.

Four-layer architecture:
  Layer 0 — classifier.py     : query intent classification
  Layer 1 — entity_gateway.py : entity resolution
  Layer 2A — query_builder.py : analytical Cypher templates
  Layer 2B — context_assembler.py : provenance context assembly
  Layer 3  — synthesis.py     : Anthropic API synthesis
  Layer 4  — web/app.py       : FastAPI endpoints
"""
from __future__ import annotations

from graphrag_pipeline.retrieval.models import ProvenanceBlock, SynthesisResult
from graphrag_pipeline.shared.settings import Settings


class RetrievalService:
    """Programmatic interface to the retrieval pipeline.

    Usage::

        from graphrag_pipeline.shared.settings import Settings
        from graphrag_pipeline.retrieval import RetrievalService

        settings = Settings.from_env()
        svc = RetrievalService(settings)
        result = svc.query("How many species were surveyed in 1985?")
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor = None

    def _get_executor(self):
        if self._executor is None:
            from graphrag_pipeline.retrieval.executor import Neo4jQueryExecutor
            self._executor = Neo4jQueryExecutor(
                uri=self._settings.neo4j_uri,
                user=self._settings.neo4j_user,
                password=self._settings.neo4j_password,
                database=self._settings.neo4j_database,
                trust_mode=self._settings.neo4j_trust,
                ca_cert_path=self._settings.neo4j_ca_cert,
            )
        return self._executor

    def query(self, text: str, *, user_context=None) -> SynthesisResult:
        """Run the full four-layer retrieval pipeline and return a synthesised answer."""
        from graphrag_pipeline.retrieval.classifier import classify_query
        from graphrag_pipeline.retrieval.entity_gateway import EntityResolutionGateway
        from graphrag_pipeline.retrieval.query_builder import CypherQueryBuilder
        from graphrag_pipeline.retrieval.context_assembler import ProvenanceContextAssembler
        from graphrag_pipeline.retrieval.synthesis import SynthesisEngine

        executor = self._get_executor()
        intent = classify_query(text)
        gateway = EntityResolutionGateway()
        entity_ctx = gateway.resolve(intent.entities)
        assembler = ProvenanceContextAssembler(executor)
        blocks = assembler.assemble(intent, entity_ctx)
        engine = SynthesisEngine(
            api_key=self._settings.anthropic_api_key,
            model=self._settings.synthesis_model,
        )
        return engine.synthesise(text, blocks)

    def provenance(self, text: str, *, user_context=None) -> list[ProvenanceBlock]:
        """Return raw provenance blocks without LLM synthesis."""
        from graphrag_pipeline.retrieval.classifier import classify_query
        from graphrag_pipeline.retrieval.entity_gateway import EntityResolutionGateway
        from graphrag_pipeline.retrieval.context_assembler import ProvenanceContextAssembler

        executor = self._get_executor()
        intent = classify_query(text)
        gateway = EntityResolutionGateway()
        entity_ctx = gateway.resolve(intent.entities)
        assembler = ProvenanceContextAssembler(executor)
        return assembler.assemble(intent, entity_ctx)

    def close(self) -> None:
        """Release the Neo4j connection."""
        if self._executor is not None:
            self._executor.close()
            self._executor = None


__all__ = ["RetrievalService"]
