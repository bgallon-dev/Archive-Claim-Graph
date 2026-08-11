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
    synthesis_timeout_seconds: float = 60.0
    synthesis_max_retries: int = 0
    # Local inference must stay bounded on museum/demo hardware. These limits
    # do not affect Anthropic synthesis.
    synthesis_local_claim_budget: int = 30
    synthesis_local_max_tokens: int = 1200
    # Context ordering: auto | proximity | kgr | confidence.
    context_ranker: str = "auto"
    kgr_root: str = ""
    # Synthesis provider switch. "anthropic" (default) keeps the current path.
    # "lmstudio"/"openai"/"local" route synthesis to an OpenAI-compatible
    # endpoint (openai_base_url) for offline/local operation. The model and key
    # for that path come from synthesis_openai_model + a resolved local key.
    # "opencode" routes synthesis through the opencode agent platform, either
    # a long-running `opencode serve` HTTP server (mode="server") or a per-call
    # `opencode run` CLI subprocess (mode="cli"). It uses opencode's own API,
    # not the OpenAI-compatible endpoint.
    synthesis_provider: str = "anthropic"
    # Ordered fallback synthesis providers tried in sequence when the primary
    # fails at query time (e.g. "anthropic,opencode"). Empty list = current
    # single-provider behavior. Each entry must be a provider SynthesisEngine
    # supports; entries without a built engine are skipped at query time.
    synthesis_fallback_providers: list[str] = field(default_factory=list)
    synthesis_openai_model: str = ""
    # OpenCode synthesis transport ("server" | "cli"). Server mode talks to a
    # long-running `opencode serve` instance; cli mode spawns `opencode run` per
    # query. Server mode is the default because it avoids per-call cold boot.
    synthesis_opencode_mode: str = "server"
    # URL of the `opencode serve` instance (server mode only). Defaults to the
    # standard opencode serve port on loopback.
    synthesis_opencode_url: str = "http://127.0.0.1:4096"
    # HTTP basic auth credentials for `opencode serve` (server mode only).
    # Username defaults to "opencode"; password is required when the server has
    # OPENCODE_SERVER_PASSWORD set.
    synthesis_opencode_username: str = "opencode"
    synthesis_opencode_password: str = ""
    # Model passed to opencode (e.g. "anthropic/claude-sonnet-4-6"). Empty means
    # opencode uses its own configured default model.
    synthesis_opencode_model: str = ""
    # opencode agent to use for synthesis. The shipped `.opencode/agent/synthesis.md`
    # agent denies all tools so the backend returns only a JSON envelope.
    synthesis_opencode_agent: str = "synthesis"
    # Path to the opencode binary (cli mode only). Defaults to "opencode".
    synthesis_opencode_binary: str = "opencode"
    # Working directory for the opencode subprocess (cli mode only). Empty means
    # the server's current working directory.
    synthesis_opencode_dir: str = ""
    # Claim-extraction provider switch (EXTRACTION PATH ONLY; synthesis/Cypher stay Anthropic)
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
    deep_research_rate_limit_max: int = 4
    deep_research_rate_limit_period: float = 60.0
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
            neo4j_user=os.environ.get("NEO4J_USER", ""),
            neo4j_password=os.environ.get("NEO4J_PASSWORD", ""),
            neo4j_database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            neo4j_trust=os.environ.get("NEO4J_TRUST", "system"),
            neo4j_ca_cert=os.environ.get("NEO4J_CA_CERT") or None,
            anthropic_api_key=(
                os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("Anthropic_API_Key", "")
            ),
            synthesis_model=os.environ.get("SYNTHESIS_MODEL", "claude-sonnet-4-6"),
            synthesis_timeout_seconds=float(os.environ.get("SYNTHESIS_TIMEOUT", "60")),
            synthesis_max_retries=int(os.environ.get("SYNTHESIS_MAX_RETRIES", "0")),
            synthesis_local_claim_budget=int(
                os.environ.get("SYNTHESIS_LOCAL_CLAIM_BUDGET", "30")
            ),
            synthesis_local_max_tokens=int(
                os.environ.get("SYNTHESIS_LOCAL_MAX_TOKENS", "1200")
            ),
            context_ranker=os.environ.get("CONTEXT_RANKER", "auto"),
            kgr_root=os.environ.get("KGR_ROOT", ""),
            synthesis_provider=os.environ.get("SYNTHESIS_PROVIDER", "anthropic").strip().lower(),
            synthesis_fallback_providers=[
                p.strip().lower()
                for p in os.environ.get("SYNTHESIS_FALLBACK_PROVIDERS", "").split(",")
                if p.strip()
            ],
            synthesis_openai_model=os.environ.get("SYNTHESIS_OPENAI_MODEL", ""),
            synthesis_opencode_mode=os.environ.get("SYNTHESIS_OPENCODE_MODE", "server").strip().lower(),
            synthesis_opencode_url=os.environ.get("SYNTHESIS_OPENCODE_URL", "http://127.0.0.1:4096"),
            synthesis_opencode_username=os.environ.get("SYNTHESIS_OPENCODE_USERNAME", "opencode"),
            synthesis_opencode_password=os.environ.get("SYNTHESIS_OPENCODE_PASSWORD", ""),
            synthesis_opencode_model=os.environ.get("SYNTHESIS_OPENCODE_MODEL", ""),
            synthesis_opencode_agent=os.environ.get("SYNTHESIS_OPENCODE_AGENT", "synthesis"),
            synthesis_opencode_binary=os.environ.get("SYNTHESIS_OPENCODE_BINARY", "opencode"),
            synthesis_opencode_dir=os.environ.get("SYNTHESIS_OPENCODE_DIR", ""),

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
            deep_research_rate_limit_max=int(
                os.environ.get("DEEP_RESEARCH_RATE_LIMIT_MAX", "4")
            ),
            deep_research_rate_limit_period=float(
                os.environ.get("DEEP_RESEARCH_RATE_LIMIT_PERIOD", "60")
            ),
            demo_all_corpora=os.environ.get("DEMO_ALL_CORPORA", "false").lower() == "true",
            conv_log_db=os.environ.get("CONV_LOG_DB", "data/conversation_log.db"),
            write_audit_db=os.environ.get("WRITE_AUDIT_DB", "data/write_audit.db"),
            review_db=os.environ.get("REVIEW_DB", "data/review.db"),
            annotation_db=os.environ.get("ANNOTATION_DB", ""),
            token_usage_db=os.environ.get("TOKEN_USAGE_DB", "data/token_usage.db"),
            ingest_db=os.environ.get("INGEST_DB", "data/ingest_jobs.db"),
            corpus_registry_path=os.environ.get("CORPUS_REGISTRY_PATH", "data/corpus_registry.yaml"),
            graphrag_api_tokens=tokens,
            ocr_engine=os.environ.get("OCR_ENGINE", "tesseract"),
            ocr_version=os.environ.get("OCR_VERSION", "5.x"),
        )
