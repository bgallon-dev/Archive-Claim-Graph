// Under-extracted documents: docs whose claim count is below the average
// for documents of similar length AND from the same institution. Surfaces
// sources worth a deep dive — the claim extractor likely missed signal
// that's in the text.
//
// Why bucket by (length, institution): claim density varies both with
// document size and with genre. A USFWS narrative report has a different
// natural density than a newspaper article, so a single corpus-wide average
// would unfairly flag verbose narratives or spare newspapers. Comparing
// each doc only to its same-length, same-institution peers gives a sharper
// "under-extracted" signal.
//
// Length bucketing: log10(char_count) rounded to 0.5 — ~half-decade buckets.
//   100 chars       -> bucket 2.0
//   1,000 chars     -> bucket 3.0
//   10,000 chars    -> bucket 4.0
//   100,000 chars   -> bucket 5.0
//
// Inputs (parameters):
//   $min_chars         — minimum total char_count for a doc to be considered
//                        (default 1000; below that the ratio is too noisy).
//   $min_bucket_docs   — minimum docs per (length_bucket, institution_id)
//                        peer group to compute a meaningful median
//                        (default 3).
//   $deficit_ratio_max — only return docs whose actual/expected ratio is at
//                        most this value (default 0.6 — 40% below expected).
//                        Set to 1.0 to see all under-extracted docs, however
//                        slightly.
//
// Why median, not mean: peer-group claim-density distributions are right-
// skewed — a couple of unusually rich docs pull the mean upward and make
// the rest look deficient. The median (percentileCont, 0.5) is robust to
// those outliers, so deficit_ratio reflects "how this doc compares to a
// typical peer" rather than "how it compares to the average peer."
//
// Output: doc_id, title, year, institution, length, claim_count, expected,
// deficit, density vs peer-group avg. Order by deficit_claims DESC so the
// biggest absolute extraction gaps surface first.
//
// Caveat 1: char_count is used as the length proxy because Paragraph.char_count
// is already stored. Word-count would be slightly more intuitive but requires
// scanning text. The ranking is the same either way for English.
//
// Caveat 2: peer groups smaller than $min_bucket_docs are skipped entirely —
// a doc with no peers in its length band has no fair comparison and is
// dropped from the output.

MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)
WITH d, sum(coalesce(p.char_count, 0)) AS char_count, count(DISTINCT p) AS paragraph_count
WHERE char_count >= $min_chars

OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)-[:HAS_PARAGRAPH]->(:Paragraph)-[:HAS_CLAIM]->(c:Claim)
WHERE c.quarantine_status IS NULL OR c.quarantine_status <> 'quarantined'
WITH d, char_count, paragraph_count, count(DISTINCT c) AS claim_count

WITH d, char_count, paragraph_count, claim_count,
     coalesce(d.institution_id, 'unknown')              AS institution_id,
     toFloat(toInteger(log10(char_count) * 2)) / 2.0    AS length_bucket,
     toFloat(claim_count) * 1000.0 / char_count         AS density_per_kchar

// Group by (length_bucket, institution) and compute the peer median density.
WITH length_bucket,
     institution_id,
     collect({d: d,
              char_count: char_count,
              paragraph_count: paragraph_count,
              claim_count: claim_count,
              density: density_per_kchar})  AS docs_in_peer_group,
     percentileCont(density_per_kchar, 0.5) AS peer_median_density,
     count(*)                               AS n_in_peer_group
WHERE n_in_peer_group >= $min_bucket_docs

UNWIND docs_in_peer_group AS row
WITH row.d                       AS d,
     row.char_count              AS char_count,
     row.paragraph_count         AS paragraph_count,
     row.claim_count              AS claim_count,
     row.density                 AS density_per_kchar,
     length_bucket,
     institution_id,
     peer_median_density,
     n_in_peer_group,
     row.char_count * peer_median_density / 1000.0 AS expected_claims

WITH d, char_count, paragraph_count, claim_count, density_per_kchar,
     length_bucket, institution_id, peer_median_density, n_in_peer_group,
     expected_claims,
     CASE
       WHEN expected_claims > 0
         THEN toFloat(claim_count) / expected_claims
       ELSE 1.0
     END AS deficit_ratio
WHERE deficit_ratio <= $deficit_ratio_max

RETURN d.doc_id                                  AS doc_id,
       d.title                                   AS title,
       d.report_year                             AS report_year,
       institution_id,
       char_count,
       paragraph_count,
       claim_count,
       toInteger(expected_claims)                AS expected_claims,
       toInteger(expected_claims - claim_count)  AS deficit_claims,
       toFloat(toInteger(deficit_ratio * 1000)) / 1000.0    AS deficit_ratio,
       toFloat(toInteger(density_per_kchar * 1000)) / 1000.0 AS density_per_kchar,
       toFloat(toInteger(peer_median_density * 1000)) / 1000.0 AS peer_median_density,
       length_bucket,
       n_in_peer_group
ORDER BY deficit_claims DESC
LIMIT 100;
