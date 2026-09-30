"""Payment verification and durable idempotency for the Kettle MCP service."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PaymentError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PaymentVerification:
    provider_reference: str


class PaymentVerifier(Protocol):
    def verify(
        self,
        *,
        payment_receipt: str,
        idempotency_key: str,
        tenant_id: str,
        operation: str,
        price_usd: float,
    ) -> PaymentVerification: ...


class RejectingPaymentVerifier:
    def verify(self, **_kwargs: Any) -> PaymentVerification:
        raise PaymentError("payment_unavailable", "Payment verification is not configured")


class HTTPPaymentVerifier:
    """Verify opaque receipts through an isolated signing/payment process."""

    def __init__(self, url: str, *, token: str | None = None, timeout: float = 15.0) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout

    def verify(
        self,
        *,
        payment_receipt: str,
        idempotency_key: str,
        tenant_id: str,
        operation: str,
        price_usd: float,
    ) -> PaymentVerification:
        payload = {
            "payment_receipt": payment_receipt,
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
            "operation": operation,
            "price_usd": round(price_usd, 2),
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            # Never include response bodies, URLs with query secrets, or tokens.
            raise PaymentError(
                "payment_verification_failed", "Payment verifier could not authorize the receipt"
            ) from exc
        if result.get("authorized") is not True:
            raise PaymentError("payment_denied", "Payment receipt was not authorized")
        reference = str(result.get("provider_reference") or "").strip()
        if not reference:
            raise PaymentError("payment_invalid", "Payment verifier returned no reference")
        return PaymentVerification(provider_reference=reference)


class PaymentLedger:
    """SQLite ledger enforcing unique idempotency keys and payment receipts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        # Serializes check-then-insert so a concurrent retry replays instead of
        # failing as a reused receipt.
        self._lock = threading.RLock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS payment_authorizations (
                idempotency_key TEXT PRIMARY KEY,
                payment_receipt TEXT NOT NULL UNIQUE,
                payment_receipt_sha256 TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                price_usd REAL NOT NULL,
                snapshot_id TEXT,
                provider_reference TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def existing(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT tenant_id, operation, price_usd, snapshot_id, provider_reference, "
                "payment_receipt_sha256 FROM payment_authorizations WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "tenant_id": row[0],
            "operation": row[1],
            "price_usd": float(row[2]),
            "snapshot_id": row[3],
            "provider_reference": row[4],
            "payment_receipt_sha256": row[5],
        }

    def record(
        self,
        *,
        idempotency_key: str,
        payment_receipt: str,
        tenant_id: str,
        operation: str,
        price_usd: float,
        snapshot_id: str | None,
        provider_reference: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._record(
                idempotency_key=idempotency_key,
                payment_receipt=payment_receipt,
                tenant_id=tenant_id,
                operation=operation,
                price_usd=price_usd,
                snapshot_id=snapshot_id,
                provider_reference=provider_reference,
            )

    def _record(
        self,
        *,
        idempotency_key: str,
        payment_receipt: str,
        tenant_id: str,
        operation: str,
        price_usd: float,
        snapshot_id: str | None,
        provider_reference: str,
    ) -> dict[str, Any]:
        existing = self.existing(idempotency_key)
        if existing is not None:
            if (
                existing["tenant_id"] != tenant_id
                or existing["operation"] != operation
                or existing["price_usd"] != price_usd
            ):
                raise PaymentError("idempotency_conflict", "Idempotency key was used for another request")
            return {**existing, "replayed": True}
        digest = hashlib.sha256(payment_receipt.encode("utf-8")).hexdigest()
        try:
            self.connection.execute(
                "INSERT INTO payment_authorizations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    idempotency_key,
                    payment_receipt,
                    digest,
                    tenant_id,
                    operation,
                    price_usd,
                    snapshot_id,
                    provider_reference,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as exc:
            raise PaymentError("payment_receipt_reused", "Payment receipt was already used") from exc
        return {
            "tenant_id": tenant_id,
            "operation": operation,
            "price_usd": price_usd,
            "snapshot_id": snapshot_id,
            "provider_reference": provider_reference,
            "payment_receipt_sha256": digest,
            "replayed": False,
        }

    def close(self) -> None:
        self.connection.close()
