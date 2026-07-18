"""Aggregate audit of the 300-article newspaper smoke output."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import statistics

OUT = Path(r"c:\Users\Benjamin\Desktop\Gemynd\Archive-Claim-Graph\out\newspapers_smoke300")
_TURNBULL_LABELS = {"Refuge", "Species", "Habitat", "SurveyMethod"}
_TURNBULL_CONCEPTS = {
    "concept_nesting_success", "concept_breeding_success", "concept_drought_condition",
    "concept_habitat_condition", "concept_survey_result", "concept_seasonal_condition",
    "concept_population_decline", "concept_population_count", "concept_breeding_activity",
}

bundles = sorted(OUT.glob("*.semantic.json"))
print(f"bundles: {len(bundles)}")

tot = Counter()
claims_per_doc: list[int] = []
entities_per_doc: list[int] = []
entity_types = Counter()
claim_types = Counter()
sources = Counter()
certainty = Counter()
quarantined = 0
obs_role_keys = Counter()
concept_ids = Counter()
relations = Counter()
zero_claim_docs = 0
leak_label_docs = 0

for b in bundles:
    j = json.loads(b.read_text(encoding="utf-8"))
    claims = j.get("claims", []) or []
    mentions = j.get("mentions", []) or []
    entities = j.get("entities", []) or []
    observations = j.get("observations", []) or []
    measurements = j.get("measurements", []) or []
    links = j.get("claim_entity_links", []) or []
    concept_links = j.get("claim_concept_links", []) or []

    tot["claims"] += len(claims)
    tot["mentions"] += len(mentions)
    tot["entities"] += len(entities)
    tot["observations"] += len(observations)
    tot["measurements"] += len(measurements)
    tot["claim_entity_links"] += len(links)
    tot["concept_links"] += len(concept_links)

    claims_per_doc.append(len(claims))
    entities_per_doc.append(len(entities))
    if not claims:
        zero_claim_docs += 1

    doc_leak = False
    for e in entities:
        et = e.get("entity_type", "?")
        entity_types[et] += 1
        if et in _TURNBULL_LABELS:
            doc_leak = True
    if doc_leak:
        leak_label_docs += 1

    for c in claims:
        claim_types[c.get("claim_type", "?")] += 1
        sources[c.get("extraction_source", "?")] += 1
        certainty[c.get("epistemic_status") or c.get("certainty") or "?"] += 1
        if c.get("quarantine_status") == "quarantined":
            quarantined += 1
    for o in observations:
        for k in (o.get("role_entities") or {}):
            obs_role_keys[k] += 1
    for cl in concept_links:
        concept_ids[cl.get("concept_id", "?")] += 1
    for ln in links:
        relations[ln.get("relation_type", "?")] += 1


def dist(xs: list[int]) -> str:
    xs = sorted(xs)
    if not xs:
        return "n/a"
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    return f"min={xs[0]} p25={p(.25)} median={statistics.median(xs)} p75={p(.75)} p90={p(.9)} max={xs[-1]}"


print(f"\nTOTALS: {dict(tot)}")
print(f"claims/doc:   {dist(claims_per_doc)}")
print(f"entities/doc: {dist(entities_per_doc)}")
print(f"docs with 0 claims: {zero_claim_docs}/{len(bundles)}")
print(f"quarantined claims: {quarantined}")

print(f"\nentity_type distribution: {dict(entity_types.most_common())}")
print(f"\n*** TURNBULL LABEL LEAK CHECK ***")
leaked = {k: v for k, v in entity_types.items() if k in _TURNBULL_LABELS}
print(f"  Turnbull labels present: {leaked or 'NONE'}  (docs affected: {leak_label_docs})")
leaked_concepts = {k: v for k, v in concept_ids.items() if k in _TURNBULL_CONCEPTS}
print(f"  Turnbull concept IDs present: {leaked_concepts or 'NONE'}")
print(f"  observation 'species' role: {obs_role_keys.get('species', 0)}")

print(f"\nclaim_type distribution (top 25): {dict(claim_types.most_common(25))}")
print(f"\nextraction_source: {dict(sources)}")
print(f"epistemic_status: {dict(certainty)}")
print(f"\nclaim_entity relation types: {dict(relations.most_common())}")
print(f"observation role keys: {dict(obs_role_keys.most_common())}")
print(f"concept IDs (top 15): {dict(concept_ids.most_common(15))}")
