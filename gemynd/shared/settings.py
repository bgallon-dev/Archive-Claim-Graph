"""Single configuration object loaded once at startup."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Neo4j
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_trust: str = "system"
    neo4j_ca_cert: str | None = None
    # LLM
    anthropic_api_key: str = ""
    synthesis_model: str = "claude-sonnet-4-6"
    # Context ordering: "auto" | "proximity" | "kgr" | "confidence"
    # (see retrieval/context_ranker.py). kgr_root points at a
    # kettle-graph-reasoner checkout to enable the temporal reranker.
    context_ranker: str = "auto"
    kgr_root: str = ""
    synthesis_timeout_seconds: float = 60.0
    # Retries on the interactive path: a bare 429/529 from the provider
    # would otherwise fail a user's query outright. The SDK's exponential
    # backoff is free; 0 disables it.
    synthesis_max_retries: int = 2
    # Query-time LLM provider seam (synthesis + LLM-Cypher). Explicit
    # provider wins; else a set base_url selects the OpenAI-compatible
    # path (vLLM / Ollama / LM Studio / llama.cpp); else Anthropic.
    # See shared/openai_compat.py.
    synthesis_llm_provider: str = ""
    synthesis_llm_base_url: str = ""
    synthesis_llm_model: str = ""
    synthesis_llm_api_key: str = ""
    # JSON dict merged into OpenAI-compatible request bodies, e.g.
    # '{"reasoning_effort": "low"}' to curb a reasoning model's thinking.
    synthesis_llm_extra_body: str = ""
    # Claim-extraction provider switch (EXTRACTION PATH ONLY; synthesis/Cypher
    # use the synthesis_llm_* seam above).
    llm_provider: str = "anthropic"
    llm_model: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:1234/v1"
    # Auth
    jwt_secret_key: str = ""
    jwt_expire_hours: int = 24
    users_db: str = "data/users.db"
    cookie_secure: bool = False
    csrf_secret_key: str = ""
    # Rate limiting
    query_rate_limit_max: int = 20
    query_rate_limit_period: float = 60.0
    # Demo/kiosk: widen corpus breadth for non-admin roles WITHOUT widening the
    # sensitivity tier. Off by default — see `demo_all_corpora` in
    # retrieval/web/app.py:require_user_scoped.
    demo_all_corpora: bool = False
    # Persistence
    conv_log_db: str = "data/conversation_log.db"
    write_audit_db: str = "data/write_audit.db"
    review_db: str = "data/review.db"
    annotation_db: str = ""
    token_usage_db: str = "data/token_usage.db"
    ingest_db: str = "data/ingest_jobs.db"
    feedback_db: str = "data/feedback.db"
    corpus_registry_path: str = "data/corpus_registry.yaml"
    # API tokens (JSON-encoded env var)
    graphrag_api_tokens: dict = field(default_factory=dict)
    # OCR metadata
    ocr_engine: str = "tesseract"
    ocr_version: str = "5.x"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_tokens = os.environ.get("GRAPHRAG_API_TOKENS", "{}")
        try:
            tokens = json.loads(raw_tokens)
        except Exception:
            tokens = {}
        return cls(
            neo4j_uri=os.environ.get("NEO4J_URI", ""),
            # Accept NEO4J_USERNAME as an alias for NEO4J_USER — the neo4j
            # tooling/driver docs use USERNAME, so .env files commonly set that.
            neo4j_user=(os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME") or ""),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", ""),
            neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            neo4j_trust=os.environ.get("NEO4J_TRUST", "system"),
            neo4j_ca_cert=os.environ.get("NEO4J_CA_CERT") or None,
            anthropic_api_key=(
                os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("Anthropic_API_Key", "")
            ),
            synthesis_model=os.environ.get("SYNTHESIS_MODEL", "claude-sonnet-4-6"),
            context_ranker=os.environ.get("CONTEXT_RANKER", "auto"),
            kgr_root=os.environ.get("KGR_ROOT", ""),
            synthesis_llm_provider=os.environ.get("SYNTHESIS_LLM_PROVIDER", ""),
            synthesis_llm_base_url=os.environ.get("SYNTHESIS_LLM_BASE_URL", ""),
            synthesis_llm_model=os.environ.get("SYNTHESIS_LLM_MODEL", ""),
            synthesis_llm_api_key=os.environ.get("SYNTHESIS_LLM_API_KEY", ""),
            synthesis_llm_extra_body=os.environ.get("SYNTHESIS_LLM_EXTRA_BODY", ""),
            synthesis_timeout_seconds=float(os.environ.get("SYNTHESIS_TIMEOUT", "60")),
            synthesis_max_retries=int(os.environ.get("SYNTHESIS_MAX_RETRIES", "2")),
            llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
            llm_model=os.environ.get("LLM_MODEL", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            jwt_secret_key=os.environ.get("JWT_SECRET_KEY", ""),
            jwt_expire_hours=int(os.environ.get("JWT_EXPIRE_HOURS", "24")),
            users_db=os.environ.get("USERS_DB", "data/users.db"),
            cookie_secure=os.environ.get("COOKIE_SECURE", "false").lower() == "true",
            csrf_secret_key=os.environ.get("CSRF_SECRET_KEY", ""),
            query_rate_limit_max=int(os.environ.get("QUERY_RATE_LIMIT_MAX", "20")),
            query_rate_limit_period=float(os.environ.get("QUERY_RATE_LIMIT_PERIOD", "60")),
            demo_all_corpora=os.environ.get("DEMO_ALL_CORPORA", "false").lower() == "true",
            conv_log_db=os.environ.get("CONV_LOG_DB", "data/conversation_log.db"),
            write_audit_db=os.environ.get("WRITE_AUDIT_DB", "data/write_audit.db"),
            review_db=os.environ.get("REVIEW_DB", "data/review.db"),
            annotation_db=os.environ.get("ANNOTATION_DB", ""),
            token_usage_db=os.environ.get("TOKEN_USAGE_DB", "data/token_usage.db"),
            ingest_db=os.environ.get("INGEST_DB", "data/ingest_jobs.db"),
            feedback_db=os.environ.get("FEEDBACK_DB", "data/feedback.db"),
            corpus_registry_path=os.environ.get("CORPUS_REGISTRY_PATH", "data/corpus_registry.yaml"),
            graphrag_api_tokens=tokens,
            ocr_engine=os.environ.get("OCR_ENGINE", "tesseract"),
            ocr_version=os.environ.get("OCR_VERSION", "5.x"),
        )
