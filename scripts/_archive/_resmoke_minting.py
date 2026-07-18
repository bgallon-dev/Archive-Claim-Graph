"""Cached re-smoke to measure entity-minting (workstream 1) impact.

Runs the pipeline over the already-staged 300 gemynd payloads with the
newspaper config. The LLM claim-extraction cache serves every paragraph, so
this is ~free; mention resolution, claim-link resolution, and the new minting
path all re-run fresh.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("CLAIM_EXTRACTION_MODEL", "claude-haiku-4-5")

REPO = Path(r"c:\Users\Benjamin\Desktop\Gemynd\Archive-Claim-Graph")
sys.path.insert(0, str(REPO))

# Load .env so the Anthropic claim-extraction cache is consulted (cache HITs
# make this re-smoke ~free; without a key the adapter degrades to NullLLMAdapter
# and no LLM claim-links flow). The secret is never printed.
_env = REPO / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
print("anthropic key present:", bool(os.environ.get("ANTHROPIC_API_KEY")))

from gemynd.ingest.pipeline import run_e2e  # noqa: E402

STAGED = REPO / "out" / "newspapers_smoke300_v3" / "_gemynd_input"
OUT = REPO / "out" / "_resmoke_mint300"
RES = REPO / "gemynd" / "resources_newspaper"

def main() -> None:
    inputs = sorted(str(p) for p in STAGED.glob("*.json"))
    print(f"inputs: {len(inputs)}")

    run_e2e(inputs, OUT, backend="memory", workers=4, resources_dir=RES, no_llm=False)

    bundles = sorted(OUT.glob("*.semantic.json"))
    tot = Counter()
    diag_codes = Counter()
    minted = 0
    minted_types = Counter()
    for b in bundles:
        j = json.loads(b.read_text(encoding="utf-8"))
        ents = j.get("entities", []) or []
        tot["entities"] += len(ents)
        tot["claim_entity_links"] += len(j.get("claim_entity_links", []) or [])
        tot["claim_location_links"] += len(j.get("claim_location_links", []) or [])
        tot["claims"] += len(j.get("claims", []) or [])
        for e in ents:
            if (e.get("properties") or {}).get("discovery_source") == "claim_link_minting":
                minted += 1
                minted_types[e.get("entity_type", "?")] += 1
        for d in j.get("claim_link_diagnostics", []) or []:
            diag_codes[d.get("diagnostic_code", "?")] += 1

    print(f"\nbundles: {len(bundles)}")
    print(f"TOTALS: {dict(tot)}")
    print(f"minted entities: {minted}  by_type={dict(minted_types.most_common())}")
    print(f"diagnostics by code: {dict(diag_codes.most_common())}")


if __name__ == "__main__":
    main()
