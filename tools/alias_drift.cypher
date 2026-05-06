// Alias-drift candidates: high-volume entities whose name is NOT an exact
// match for any seed name, but DOES contain a multi-word seed name as a
// substring (case-insensitive). These are likely variants of seeded entities
// that the entity-resolver missed and should be merged at the resolver layer
// rather than added as new seeds.
//
// Example: seeded "Northern Pacific Railway" (with aliases "Northern Pacific",
// "N. P.") is unlikely to be merged with the corpus entity
// "NORTHERN PACIFIC RAILROAD CO" by exact-match resolution. This query
// surfaces that gap.
//
// Inputs (parameters):
//   $seed_names — full list of lowercased names + aliases (same shape as the
//                 list passed to seeding_candidates.cypher). Multi-word names
//                 (those containing a space) are used as drift probes.
//
// Filters:
//   * Excludes Period entities and quarantined claims.
//   * Skips entities that exact-match a seed name (those would already be
//     filtered as seeded).
//   * Requires the matched seed to contain a space. Single-word seeds
//     ("Spokane", "Idaho") would be too noisy as substrings — every
//     "Spokane River", "Spokane County" would match.
//   * Requires the matched seed to be at least 8 characters long, to keep
//     short multi-word aliases like "U. S." from triggering broad matches.
//
// Output: claim_count and the matched seed(s), so you can see *which* seed
// the entity is likely a variant of.

WITH [s IN $seed_names WHERE s CONTAINS ' ' AND size(s) >= 8] AS multi_word_seeds
MATCH (e:Entity)<-[]-(c:Claim)
WHERE (c.quarantine_status IS NULL OR c.quarantine_status <> 'quarantined')
  AND NOT 'Period' IN labels(e)
  AND NOT toLower(coalesce(e.name, '')) IN $seed_names
  AND ANY(seed IN multi_word_seeds WHERE toLower(e.name) CONTAINS seed)
WITH e,
     multi_word_seeds,
     count(DISTINCT c) AS claim_count
WITH e,
     claim_count,
     [seed IN multi_word_seeds WHERE toLower(e.name) CONTAINS seed] AS matched_seeds
RETURN e.entity_id                                AS entity_id,
       [l IN labels(e) WHERE l <> 'Entity'][0]    AS entity_type,
       e.name                                     AS name,
       claim_count,
       matched_seeds[..3]                         AS likely_canonical_seed
ORDER BY claim_count DESC
LIMIT 100;
