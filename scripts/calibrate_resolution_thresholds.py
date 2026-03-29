"""
Calibrate entity resolution thresholds from processed semantic bundles.

Loads all *.semantic.json files in a directory, extracts a feature matrix
from EntityResolutionRecord objects, fits a logistic regression, and reports
suggested ResolutionPolicy threshold values with bootstrap confidence intervals.

Optionally writes a calibrated ResolutionPolicy to a YAML file that can be
loaded by DictionaryFuzzyResolver.

Requirements (not in core dependencies):
    pip install scikit-learn numpy

Usage:
    python scripts/calibrate_resolution_thresholds.py \\
        --bundles-dir out/ \\
        [--min-samples 50] \\
        [--out-policy calibrated_policy.yaml] \\
        [--verbose]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def extract_features(bundles_dir: Path) -> tuple[list[list[float]], list[int], int]:
    """Load bundles and build feature matrix.

    Returns:
        rows: list of feature vectors
        labels: 1=REFERS_TO, 0=POSSIBLY_REFERS_TO
        skipped: count of records excluded (non-REFERS_TO/POSSIBLY_REFERS_TO)
    """
    try:
        from graphrag_pipeline.shared.io_utils import load_semantic_bundle
        from graphrag_pipeline.shared.resource_loader import load_claim_relation_compatibility
    except ImportError as exc:
        sys.exit(f"Error: graphrag_pipeline package not importable: {exc}")

    # Load preferred_entity_types for the entity_type_match feature.
    try:
        compat = load_claim_relation_compatibility()
        preferred_entity_types: dict[str, list[str]] = compat.get("preferred_entity_types", {})
    except Exception:
        preferred_entity_types = {}

    bundle_paths = sorted(bundles_dir.glob("*.semantic.json"))
    if not bundle_paths:
        return [], [], 0

    rows: list[list[float]] = []
    labels: list[int] = []
    skipped = 0

    for path in bundle_paths:
        try:
            bundle = load_semantic_bundle(path)
        except Exception as exc:
            print(f"Warning: could not load {path.name}: {exc}", file=sys.stderr)
            continue

        mention_by_id = {m.mention_id: m for m in bundle.mentions}
        entity_by_id = {e.entity_id: e for e in bundle.entities}

        # Claim types per paragraph (for entity_type_match feature).
        claim_types_by_para: dict[str, set[str]] = defaultdict(set)
        for c in bundle.claims:
            claim_types_by_para[c.paragraph_id].add(c.claim_type)

        for res in bundle.entity_resolutions:
            if res.relation_type == "REFERS_TO":
                label = 1
            elif res.relation_type == "POSSIBLY_REFERS_TO":
                label = 0
            else:
                skipped += 1
                continue

            mention = mention_by_id.get(res.mention_id)
            entity = entity_by_id.get(res.entity_id)
            if mention is None or entity is None:
                skipped += 1
                continue

            # Feature: entity_type_match
            para_claim_types = claim_types_by_para.get(mention.paragraph_id, set())
            entity_type_match = 0
            for ct in para_claim_types:
                if entity.entity_type in preferred_entity_types.get(ct, []):
                    entity_type_match = 1
                    break

            row = [
                float(res.match_score),
                float(entity_type_match),
                float(1 if getattr(mention, "ocr_suspect", False) else 0),
                _length_ratio(mention.normalized_form, entity.normalized_form),
                _token_overlap(mention.normalized_form, entity.normalized_form),
            ]
            rows.append(row)
            labels.append(label)

    return rows, labels, skipped


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_threshold_ci(
    X_scaled: object,
    y: list[int],
    clf: object,
    feature_idx: int = 0,
    n_iter: int = 200,
    alpha: float = 0.95,
) -> tuple[float, float]:
    """Estimate CI for the decision boundary projected onto feature *feature_idx*."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = len(y)
    boundaries: list[float] = []

    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        X_b = X_scaled[idx]  # type: ignore[index]
        y_b = np.array(y)[idx]
        if len(set(y_b)) < 2:
            continue
        from sklearn.linear_model import LogisticRegression
        clf_b = LogisticRegression(max_iter=1000, random_state=42)
        clf_b.fit(X_b, y_b)
        coef = clf_b.coef_[0]
        intercept = clf_b.intercept_[0]
        if abs(coef[feature_idx]) > 1e-9:
            # boundary for feature j: coef[0]*x[0] + ... = -intercept
            # holding others at 0 (scaled space), solve for feature_idx:
            boundary = -intercept / coef[feature_idx]
            boundaries.append(float(boundary))

    if len(boundaries) < 10:
        return float("nan"), float("nan")

    lo = float(np.percentile(boundaries, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(boundaries, (1 + alpha) / 2 * 100))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate resolution thresholds from semantic bundles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--bundles-dir", required=True, type=Path,
                        help="Directory containing *.semantic.json files")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Minimum resolution records required to fit model (default: 50)")
    parser.add_argument("--out-policy", type=Path, default=None,
                        help="Write calibrated ResolutionPolicy YAML to this path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-fold CV metrics")
    args = parser.parse_args()

    # Check sklearn early so the error is clear.
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, cross_val_predict
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        sys.exit(
            "Error: scikit-learn and numpy are required for threshold calibration.\n"
            "Install them with:  pip install scikit-learn numpy"
        )

    bundles_dir = args.bundles_dir
    if not bundles_dir.is_dir():
        sys.exit(f"Error: --bundles-dir is not a directory: {bundles_dir}")

    print(f"Loading bundles from {bundles_dir} …")
    rows, labels, skipped = extract_features(bundles_dir)

    n_bundles = len(list(bundles_dir.glob("*.semantic.json")))
    n_samples = len(rows)
    print(f"  {n_bundles} bundle(s), {n_samples} resolutions ({skipped} skipped)")

    if n_samples < args.min_samples:
        sys.exit(
            f"Error: only {n_samples} samples found (minimum {args.min_samples}).\n"
            "Process more documents or lower --min-samples."
        )

    n_pos = sum(labels)
    n_neg = n_samples - n_pos
    print(f"  REFERS_TO: {n_pos}  POSSIBLY_REFERS_TO: {n_neg}")

    X = np.array(rows)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    if args.verbose:
        print(f"\nCross-validation ROC AUC per fold: {[round(s, 3) for s in cv_scores]}")

    print(f"\nCross-validation ROC AUC: {cv_mean:.3f} ± {cv_std:.3f}")

    # Feature names
    feature_names = [
        "match_score",
        "entity_type_match",
        "ocr_suspect",
        "length_ratio",
        "token_overlap",
    ]

    coef = clf.coef_[0]
    print("\nFeature importances (coefficient magnitude):")
    for name, c in sorted(zip(feature_names, coef), key=lambda x: -abs(x[1])):
        print(f"  {name:<22}  {c:+.4f}")

    # Suggest thresholds: find the match_score value at the decision boundary
    # by holding all other features at their median values in scaled space.
    X_median_scaled = np.median(X_scaled, axis=0)
    match_coef_scaled = coef[0]
    other_contribution = float(np.dot(coef[1:], X_median_scaled[1:]) + clf.intercept_[0])

    if abs(match_coef_scaled) > 1e-9:
        boundary_scaled = -other_contribution / match_coef_scaled
        # Convert back to original match_score space.
        match_scale = scaler.scale_[0]
        match_mean = scaler.mean_[0]
        refers_to_threshold = round(float(boundary_scaled * match_scale + match_mean), 3)
    else:
        print("Warning: match_score has near-zero coefficient; threshold suggestion unreliable.")
        refers_to_threshold = 0.85

    # maybe_threshold: use the 5th percentile of REFERS_TO match scores as a conservative floor.
    refers_to_scores = X[np.array(labels) == 1, 0]
    maybe_threshold = round(float(np.percentile(refers_to_scores, 5)), 3) if len(refers_to_scores) > 0 else 0.65
    # Enforce ordering.
    maybe_threshold = min(maybe_threshold, refers_to_threshold - 0.05)
    maybe_threshold = max(maybe_threshold, 0.40)

    # Bootstrap CI for refers_to_threshold (in match_score space).
    lo_scaled, hi_scaled = _bootstrap_threshold_ci(X_scaled, labels, clf, feature_idx=0)
    if not math.isnan(lo_scaled):
        lo = round(lo_scaled * scaler.scale_[0] + scaler.mean_[0], 3)
        hi = round(hi_scaled * scaler.scale_[0] + scaler.mean_[0], 3)
        ci_str = f"  (95% CI: {lo}–{hi})"
    else:
        ci_str = "  (CI unavailable — too few bootstrap samples)"

    print(f"\nSuggested refers_to_threshold: {refers_to_threshold}{ci_str}")
    print(f"Suggested maybe_threshold:      {maybe_threshold}")

    if args.out_policy is not None:
        import yaml  # PyYAML is a core dep of the pipeline
        policy_data = {
            "refers_to_threshold": refers_to_threshold,
            "maybe_threshold": maybe_threshold,
            "uniqueness_gap": 0.05,  # not estimated — kept at default
        }
        header = (
            f"# Calibrated by calibrate_resolution_thresholds.py\n"
            f"# Bundles: {n_bundles} file(s), {n_samples} resolutions\n"
            f"# ROC AUC: {cv_mean:.3f} \u00b1 {cv_std:.3f}\n"
        )
        args.out_policy.parent.mkdir(parents=True, exist_ok=True)
        with args.out_policy.open("w", encoding="utf-8") as fh:
            fh.write(header)
            yaml.safe_dump(policy_data, fh, default_flow_style=False)
        print(f"\nCalibrated policy written to: {args.out_policy}")


if __name__ == "__main__":
    main()
