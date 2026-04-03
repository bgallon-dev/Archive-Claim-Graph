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
    # Auth
    jwt_secret_key: str = ""
    jwt_expire_hours: int = 24
    users_db: str = "data/users.db"
    cookie_secure: bool = False
    # Persistence
    conv_log_db: str = "data/conversation_log.db"
    write_audit_db: str = "data/write_audit.db"
    review_db: str = "data/review.db"
    annotation_db: str = ""
    token_usage_db: str = "data/token_usage.db"
    ingest_db: str = "data/ingest_jobs.db"
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
            jwt_secret_key=os.environ.get("JWT_SECRET_KEY", ""),
            jwt_expire_hours=int(os.environ.get("JWT_EXPIRE_HOURS", "24")),
            users_db=os.environ.get("USERS_DB", "data/users.db"),
            cookie_secure=os.environ.get("COOKIE_SECURE", "false").lower() == "true",
            conv_log_db=os.environ.get("CONV_LOG_DB", "data/conversation_log.db"),
            write_audit_db=os.environ.get("WRITE_AUDIT_DB", "data/write_audit.db"),
            review_db=os.environ.get("REVIEW_DB", "data/review.db"),
            annotation_db=os.environ.get("ANNOTATION_DB", ""),
            token_usage_db=os.environ.get("TOKEN_USAGE_DB", "data/token_usage.db"),
            ingest_db=os.environ.get("INGEST_DB", "data/ingest_jobs.db"),
            graphrag_api_tokens=tokens,
            ocr_engine=os.environ.get("OCR_ENGINE", "tesseract"),
            ocr_version=os.environ.get("OCR_VERSION", "5.x"),
        )
