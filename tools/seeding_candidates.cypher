// Seeding-candidate ranking: entities NOT already in seed_entities.csv,
// scored by a novelty heuristic that rewards breadth over raw volume.
//
// Inputs (parameters):
//   $seed_names      — list of lowercased names + aliases already in the
//                      active seed_entities.csv files. Entities whose name
//                      (lowercased) matches any of these are excluded. Pass
//                      `[]` to disable the filter and see the raw ranking.
//   $min_doc_spread  — minimum number of distinct documents (default 2).
//   $min_claim_types — minimum number of distinct claim_types (default 2).
//   $exclude_periods — boolean, drop Period (date-bucket) entities (default
//                      true).
//
// Schema notes:
//   * Document -> Claim is a 4-hop path:
//       (Document)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)
//                 -[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(:Claim)
//   * Claim -> Entity uses a dynamic relation_type. The unlabelled `<-[]-`
//     matches any of MANAGEMENT_TARGET, SUBJECT_OF_CLAIM, TOPIC_OF_CLAIM,
//     OCCURRED_AT, REFERS_TO, ...
//   * Document year property is `report_year` (integer).
//
// Filters (applied to keep candidates that look like genuine new seeds):
//   * Already-seeded: case-insensitive `name` match against $seed_names.
//   * Period entities: date buckets are never useful as retrieval seeds.
//   * Singletons: document_spread > 1 AND distinct_claim_types > 1
//     (single-doc / single-claim-type entries are usually import artifacts —
//     e.g. the `STATE OF WASHINGTON` row with 1 claim type, 1 document.)
//   * Quarantined claims excluded from the count.
//
// Score:
//   novelty_score = claim_count * distinct_claim_types * log10(document_spread + 1)
//   The log10 dampens raw volume so an entity referenced in 5 claim types
//   across 100 docs ranks above one with 700 claims confined to 1 doc.
//
// Caveat — alias drift: this only filters by name. If a corpus entity is a
// variant of a seeded one but never aliased (e.g. the graph has
// "NORTHERN PACIFIC RAILROAD CO" while the seed is "Northern Pacific Railway"),
// it will still appear here. That's an entity-resolution gap worth surfacing,
// not a seeding gap.

MATCH (e:Entity)<-[]-(c:Claim)<-[:HAS_CLAIM]-(p:Paragraph)
MATCH (p)<-[:HAS_PARAGRAPH]-(:Section)<-[:HAS_SECTION]-(:Page)<-[:HAS_PAGE]-(d:Document)
WHERE (c.quarantine_status IS NULL OR c.quarantine_status <> 'quarantined')
  AND (NOT $exclude_periods OR NOT 'Period' IN labels(e))
  AND NOT toLower(coalesce(e.name, '')) IN $seed_names
WITH e,
     count(DISTINCT c)            AS claim_count,
     count(DISTINCT c.claim_type) AS distinct_claim_types,
     count(DISTINCT d)            AS document_spread,
     min(d.report_year)           AS earliest_year,
     max(d.report_year)           AS latest_year
WHERE document_spread >= $min_doc_spread AND distinct_claim_types >= $min_claim_types
RETURN e.entity_id                                AS entity_id,
       [l IN labels(e) WHERE l <> 'Entity'][0]    AS entity_type,
       e.name                                     AS name,
       claim_count,
       distinct_claim_types,
       document_spread,
       earliest_year,
       latest_year,
       (latest_year - earliest_year)              AS year_span,
       toInteger(claim_count * distinct_claim_types * log10(document_spread + 1)) AS novelty_score
ORDER BY novelty_score DESC
LIMIT 200;
