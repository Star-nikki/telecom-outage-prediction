"""
Priority 5, Candidate 1: leaf-based training-support diagnostics.

READ-ONLY / DIAGNOSTIC ONLY -- same guarantee as the other diagnostic
scripts. Does not modify decision_agent.py, does not retrain anything,
does not touch predict_with_confidence() or the production confidence
formula, does not write anything back to models/. This is a HYPOTHESIS
TEST for Candidate 1, not an implementation of it. If the results here are
weak, we move to Candidate 2 (conformal prediction) rather than forcing
this one into production.

WHERE TO SAVE:
    Same folder as decision_agent.py, calibration_diagnostics.py, and
    confidence_diagnostics.py (agents/). Imports from both of those, so
    they must already be sitting next to it -- they already are.

HOW TO RUN:
    From the project root:      python -m agents.leaf_support_diagnostics
    From inside agents/:        python leaf_support_diagnostics.py

WHAT "LEAF SUPPORT" MEANS HERE (precise definition):
    For a saved LightGBM model, every tree partitions the feature space
    into leaves. For a given test row, look up which leaf it lands in on
    EVERY tree in the model (model.predict(X, pred_leaf=True) -- this is
    the model's own learned structure, not a re-derivation from probability
    or margin). Separately, for a reference population of TRAINING rows,
    precompute, per tree, how many training rows land in each leaf, and how
    many of those were actual positives (outages). A test row's leaf
    support is then: across all trees, how many training rows shared its
    leaf, and what fraction of those were positives. This is a genuine
    training-data-density signal, structurally unrelated to the model's own
    probability output for that row (a row can have a high predicted
    probability while landing in leaves that saw almost no positive
    training examples -- that mismatch is exactly what this is trying to
    surface).

IMPORTANT APPROXIMATION, STATED UP FRONT:
    Production bagging trains each of the 3 ensemble members on a
    different random undersampled subset (positives + up to 10x negatives,
    per train_bagged_ensemble()). Which exact rows went into which
    member's bag is not saved anywhere, so this script cannot reconstruct
    it. Instead, leaf-support tables are built from a single reference
    sample drawn from the full available training split (all positives +
    a capped random sample of negatives, see TRAIN_REFERENCE_MAX_ROWS
    below) and applied identically against each bagged model's own tree
    structure. This is an approximation of "how much relevant training
    data backs this prediction," not an exact replay of each member's
    bootstrap. It is a reasonable approximation for a diagnostic, but
    should be named as such if this ever gets written up.

BACKEND SUPPORT:
    LightGBM only. model.predict(X, pred_leaf=True) is a LightGBM
    sklearn-API feature; XGBoost's sklearn API exposes the equivalent via
    model.apply(X) instead, which isn't wired up here. If you need this for
    the xgboost backend later, that's a small separate addition, not a
    redesign.
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
from config import PREDICTION_HORIZONS
from calibration_diagnostics import load_features, time_split
from confidence_diagnostics import (
    _resolve_model_dir, tune_threshold_on_val, compute_row_level_stats, BACKEND,
)

pd.set_option("display.width", 160)

if BACKEND != "lightgbm":
    raise SystemExit(
        f"leaf_support_diagnostics.py only supports the lightgbm backend "
        f"(pred_leaf=True). confidence_diagnostics.BACKEND is currently "
        f"'{BACKEND}'. Set it to 'lightgbm' in confidence_diagnostics.py, "
        f"or extend this script for xgboost via model.apply(X) first."
    )

# Reference population used to build leaf -> (count, positive_count) tables.
# All positives are always kept; negatives are capped and randomly sampled
# to this total, purely for runtime/memory practicality on an 8GB machine.
# This is the approximation described in the module docstring above.
TRAIN_REFERENCE_MAX_ROWS = 300_000

# Test-side sampling: all positives are always kept (they're the rare,
# important class); negatives are capped so the script finishes in
# reasonable time. This does NOT affect the leaf-support TABLES (built from
# the training reference above) -- only which test rows we score and report.
TEST_MAX_NEGATIVES = 150_000

# A row is flagged "insufficient evidence" (not scored, not force-fit into
# a 0-1 number) if, across the ensemble, the average total training support
# in its matched leaves falls below this many rows.
MIN_SUPPORT_FOR_SCORING = 5

# Same idea, for positive class support specifically: a predicted-positive
# row whose matched leaves saw fewer than this many positive training
# examples is flagged as having essentially no positive-class evidence,
# regardless of how many negatives happen to share the leaf.
MIN_POSITIVE_SUPPORT_FLAG = 1

RNG_SEED = 42


def build_reference_sample(train_df, target_col):
    y = train_df[target_col]
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index
    n_neg_keep = max(0, TRAIN_REFERENCE_MAX_ROWS - len(pos_idx))
    if n_neg_keep < len(neg_idx):
        rng = np.random.RandomState(RNG_SEED)
        neg_idx = pd.Index(rng.choice(neg_idx, size=n_neg_keep, replace=False))
    sel = pos_idx.union(neg_idx)
    return train_df.loc[sel]


def build_test_sample(test_df, target_col):
    y = test_df[target_col]
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index
    if len(neg_idx) > TEST_MAX_NEGATIVES:
        rng = np.random.RandomState(RNG_SEED + 1)
        neg_idx = pd.Index(rng.choice(neg_idx, size=TEST_MAX_NEGATIVES, replace=False))
    sel = pos_idx.union(neg_idx)
    return test_df.loc[sel]


def build_leaf_tables(model, X_ref, y_ref_arr):
    """Per tree: total training-row count and positive-row count per leaf id.
    Returns two lists of 1D numpy arrays, one array per tree, indexed by
    leaf id."""
    leaf_ref = model.predict(X_ref, pred_leaf=True)  # (n_ref, n_trees)
    n_trees = leaf_ref.shape[1]
    total_tables, pos_tables = [], []
    for t in range(n_trees):
        leaves_t = leaf_ref[:, t]
        max_leaf = int(leaves_t.max()) + 1
        total = np.bincount(leaves_t, minlength=max_leaf).astype(np.float64)
        pos = np.bincount(leaves_t, weights=y_ref_arr, minlength=max_leaf)
        total_tables.append(total)
        pos_tables.append(pos)
    return total_tables, pos_tables


def score_leaf_support(model, X, total_tables, pos_tables):
    """Per test row, per tree: total support and positive support in the
    matched leaf. Returns two (n_rows, n_trees) matrices -- raw counts,
    no normalization at this stage."""
    leaf_test = model.predict(X, pred_leaf=True)
    n_rows, n_trees = leaf_test.shape
    support = np.zeros((n_rows, n_trees), dtype=np.float64)
    pos_support = np.zeros((n_rows, n_trees), dtype=np.float64)
    for t in range(n_trees):
        total = total_tables[t]
        pos = pos_tables[t]
        leaves_t = np.clip(leaf_test[:, t], 0, len(total) - 1)
        support[:, t] = total[leaves_t]
        pos_support[:, t] = pos[leaves_t]
    return support, pos_support


def run_horizon(label, hours, train_df, val_df, test_df, feature_columns, model_dir):
    model_path = os.path.join(model_dir, f"decision_model_{BACKEND}_{label}.joblib")
    if not os.path.exists(model_path):
        print(f"\n[{label}] No saved model at {model_path} -- skipping.")
        return None

    target_col = f"outage_in_next_{label}"
    if target_col not in train_df.columns:
        print(f"\n[{label}] No column {target_col} -- skipping.")
        return None

    models = joblib.load(model_path)
    X_val, y_val = val_df[feature_columns], val_df[target_col]
    if y_val.sum() == 0 or test_df[target_col].sum() == 0:
        print(f"\n[{label}] No positives in val or test split -- skipping.")
        return None

    threshold = tune_threshold_on_val(models, X_val, y_val)

    ref_df = build_reference_sample(train_df, target_col)
    X_ref = ref_df[feature_columns]
    y_ref_arr = ref_df[target_col].values.astype(np.float64)

    test_sample = build_test_sample(test_df, target_col)
    X_test = test_sample[feature_columns]
    y_test_arr = test_sample[target_col].values

    print(f"\n{'='*78}\nHORIZON {label} ({hours}h)  backend={BACKEND}  n_bagging={len(models)}  "
          f"tuned_threshold={threshold:.4f}\n{'='*78}")
    print(f"Reference (training) sample for leaf tables: {len(ref_df):,} rows "
          f"({int(y_ref_arr.sum()):,} positive, {len(ref_df) - int(y_ref_arr.sum()):,} negative)")
    print(f"Test sample scored: {len(test_sample):,} rows "
          f"({int(y_test_arr.sum()):,} positive [all kept], "
          f"{len(test_sample) - int(y_test_arr.sum()):,} negative [capped/sampled])")

    # ---- Per bagged model: build tables, score, then combine across models ----
    per_model_support = []
    per_model_pos_support = []
    for m in models:
        total_tables, pos_tables = build_leaf_tables(m, X_ref, y_ref_arr)
        support, pos_support = score_leaf_support(m, X_test, total_tables, pos_tables)
        per_model_support.append(support.mean(axis=1))       # mean across trees, this model
        per_model_pos_support.append(pos_support.mean(axis=1))

    per_model_support = np.array(per_model_support)      # (n_bagging, n_test)
    per_model_pos_support = np.array(per_model_pos_support)

    mean_support = per_model_support.mean(axis=0)          # avg across ensemble members
    min_support = per_model_support.min(axis=0)             # worst-case member -- conservative
    mean_pos_support = per_model_pos_support.mean(axis=0)
    pos_fraction_in_leaf = mean_pos_support / np.clip(mean_support, 1e-9, None)

    # Normalized (0-1) score, kept explicitly separate from the raw counts:
    # percentile rank of mean_support within this horizon's scored test set.
    # This is a RELATIVE score (useful for bucketing/ranking) -- it is not
    # a claim about absolute reliability, which is what the raw counts and
    # the sparse-evidence flag below are for.
    support_percentile = pd.Series(mean_support).rank(pct=True).values

    # Existing production confidence, for direct comparison
    probability, raw_std, existing_confidence = compute_row_level_stats(models, X_test)

    pred = (probability >= threshold).astype(int)
    correct = (pred == y_test_arr).astype(int)
    sq_err = (probability - y_test_arr) ** 2

    insufficient_evidence = mean_support < MIN_SUPPORT_FOR_SCORING
    frac_insufficient = insufficient_evidence.mean()
    print(f"\nRows flagged INSUFFICIENT EVIDENCE (mean training support < {MIN_SUPPORT_FOR_SCORING} "
          f"across matched leaves): {int(insufficient_evidence.sum()):,} / {len(mean_support):,} "
          f"({frac_insufficient*100:.2f}%) -- these are not scored below, not forced into a number.")

    predicted_positive_no_pos_evidence = (pred == 1) & (mean_pos_support < MIN_POSITIVE_SUPPORT_FLAG)
    print(f"Rows predicted POSITIVE with essentially no positive-class training support "
          f"(mean_pos_support < {MIN_POSITIVE_SUPPORT_FLAG}): "
          f"{int(predicted_positive_no_pos_evidence.sum()):,} / {int(pred.sum()):,} positive predictions "
          f"({(predicted_positive_no_pos_evidence.sum()/max(pred.sum(),1))*100:.2f}%) -- "
          f"this is exactly the pattern (confident-looking prediction, ~zero real positive evidence) "
          f"leaf support is meant to catch.")

    valid_mask = ~insufficient_evidence
    n_valid = int(valid_mask.sum())
    print(f"Scorable rows (evidence sufficient): {n_valid:,} / {len(mean_support):,}")

    if n_valid < 30:
        print(f"\n>> Verdict for {label}: INSUFFICIENT EVIDENCE -- too few scorable rows "
              f"({n_valid}) to draw a reliable conclusion for this horizon.")
        return {
            "horizon": label, "hours": hours, "n_test_scored": len(mean_support),
            "n_valid": n_valid, "frac_insufficient_evidence": float(frac_insufficient),
            "verdict": "INSUFFICIENT EVIDENCE",
        }, None

    ms, mps, pfl, sp = mean_support[valid_mask], mean_pos_support[valid_mask], \
        pos_fraction_in_leaf[valid_mask], support_percentile[valid_mask]
    corr_v, prob_v, ec_v, rs_v = correct[valid_mask], probability[valid_mask], \
        existing_confidence[valid_mask], raw_std[valid_mask]
    y_v, sq_v = y_test_arr[valid_mask], sq_err[valid_mask]

    # ---- 1/2. Distribution of the leaf-support signal ----
    print(f"\n--- 1/2. Leaf-support distribution (raw counts, scorable rows only) ---")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"  mean_support:      min={ms.min():.1f}  max={ms.max():.1f}  mean={ms.mean():.1f}  "
          f"median={np.median(ms):.1f}")
    print("    percentiles: " + "  ".join(f"p{p}={np.percentile(ms, p):.1f}" for p in pcts))
    print(f"  mean_pos_support:  min={mps.min():.2f}  max={mps.max():.2f}  mean={mps.mean():.2f}  "
          f"median={np.median(mps):.2f}")
    print("    percentiles: " + "  ".join(f"p{p}={np.percentile(mps, p):.2f}" for p in pcts))
    print(f"  pos_fraction_in_leaf: mean={pfl.mean():.5f}  median={np.median(pfl):.5f}  "
          f"max={pfl.max():.5f}")
    print(f"  support_percentile (normalized 0-1, relative rank -- NOT an absolute reliability score): "
          f"by construction spans 0-1 uniformly; reported for reference only.")

    # ---- 3. Class-aware support ----
    print(f"\n--- 3. Class-aware support check ---")
    n_high_total_low_pos = ((ms > np.percentile(ms, 75)) & (mps < MIN_POSITIVE_SUPPORT_FLAG)).sum()
    print(f"  Rows in the TOP quartile of total support but with essentially no positive-class "
          f"support (mean_pos_support < {MIN_POSITIVE_SUPPORT_FLAG}): {int(n_high_total_low_pos):,} "
          f"({n_high_total_low_pos/n_valid*100:.2f}% of scorable rows). "
          f"This is the exact case raw total-support alone would get wrong -- high overall density, "
          f"almost entirely negative examples.")

    # ---- 4. Relationship with correctness, bucketed ----
    print(f"\n--- 4. Buckets by mean_support (quartile-based, scorable rows only) ---")
    try:
        qbins = pd.qcut(ms, q=[0, .25, .5, .75, 1.0], duplicates="drop")
    except ValueError:
        qbins = pd.qcut(ms, q=2, duplicates="drop")
    bucket_rows = []
    for interval in sorted(qbins.unique()):
        mask = qbins == interval
        n = int(mask.sum())
        bucket_rows.append({
            "horizon": label,
            "support_bucket": str(interval),
            "n": n,
            "accuracy": float(corr_v[mask].mean()),
            "brier": float(sq_v[mask].mean()),
            "positive_rate": float(y_v[mask].mean()),
            "mean_predicted_prob": float(prob_v[mask].mean()),
            "mean_support": float(ms[mask].mean()),
            "mean_pos_support": float(mps[mask].mean()),
        })
    bucket_df = pd.DataFrame(bucket_rows)
    print(bucket_df.to_string(index=False))

    spearman_support_correct = pd.Series(ms).corr(pd.Series(corr_v), method="spearman")
    accs = bucket_df["accuracy"].values
    diffs = np.diff(accs)
    trend = "increasing" if (diffs >= -1e-9).all() else ("decreasing" if (diffs <= 1e-9).all() else "non-monotonic")
    print(f"\n  Spearman correlation(mean_support, correct): {spearman_support_correct:.4f}")
    print(f"  Bucket accuracy trend across increasing support: {trend}")

    # ---- 6. Comparison against existing confidence ----
    print(f"\n--- 6. Comparison against existing ensemble-spread confidence ---")
    spearman_existing_correct = pd.Series(ec_v).corr(pd.Series(corr_v), method="spearman")
    spearman_support_vs_existing = pd.Series(ms).corr(pd.Series(ec_v), method="spearman")
    print(f"  Spearman(existing_confidence, correct) on this same scorable set: {spearman_existing_correct:.4f}")
    print(f"  Spearman(leaf_support, correct):                                   {spearman_support_correct:.4f}")
    print(f"  Spearman(leaf_support, existing_confidence): {spearman_support_vs_existing:.4f} "
          f"(low = leaf support is telling us something existing confidence doesn't; "
          f"high = largely redundant with it)")

    # ---- 7 already reported above (insufficient evidence / sparse cases) ----

    # ---- 8. Verdict ----
    if abs(spearman_support_correct) >= 0.15 and trend == "increasing":
        verdict = "MEANINGFULLY DISCRIMINATIVE"
    elif abs(spearman_support_correct) >= 0.05:
        verdict = "WEAK"
    else:
        verdict = "ABSENT / NOT RELIABLE"
    print(f"\n>> Verdict for {label}: leaf support is {verdict} "
          f"(existing confidence was: see confidence_diagnostics_summary.csv for comparison)")

    summary_row = {
        "horizon": label, "hours": hours,
        "n_test_scored": len(mean_support), "n_valid": n_valid,
        "frac_insufficient_evidence": float(frac_insufficient),
        "n_pos_pred_no_pos_evidence": int(predicted_positive_no_pos_evidence.sum()),
        "n_pos_pred_total": int(pred.sum()),
        "mean_support_mean": float(ms.mean()), "mean_support_median": float(np.median(ms)),
        "mean_pos_support_mean": float(mps.mean()),
        "spearman_support_vs_correct": float(spearman_support_correct),
        "spearman_existing_conf_vs_correct": float(spearman_existing_correct),
        "spearman_support_vs_existing_conf": float(spearman_support_vs_existing),
        "bucket_accuracy_trend": trend,
        "verdict": verdict,
    }
    return summary_row, bucket_df


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    train_df, val_df, test_df, train_end, val_end = time_split(df)
    model_dir = _resolve_model_dir()

    print(f"Backend under diagnosis: {BACKEND}")
    print(f"Model dir resolved to: {os.path.abspath(model_dir)}")
    print(f"Train: {len(train_df):,} rows (< {train_end})  [reference sample capped at "
          f"{TRAIN_REFERENCE_MAX_ROWS:,} rows -- see APPROXIMATION note in the file docstring]")
    print(f"Val:   {len(val_df):,} rows (threshold tuning only, same as confidence_diagnostics.py)")
    print(f"Test:  {len(test_df):,} rows (>= {val_end})  [negatives capped at "
          f"{TEST_MAX_NEGATIVES:,} for scoring; all positives always kept]\n")

    all_summaries, all_buckets = [], []
    for label, hours in PREDICTION_HORIZONS:
        result = run_horizon(label, hours, train_df, val_df, test_df, feature_columns, model_dir)
        if result is None:
            continue
        summary, bucket_df = result
        all_summaries.append(summary)
        if bucket_df is not None:
            all_buckets.append(bucket_df)

    if not all_summaries:
        print("\nNo horizons produced results.")
        return

    print(f"\n\n{'='*78}\nSUMMARY ACROSS ALL HORIZONS\n{'='*78}")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("leaf_support_diagnostics_summary.csv", index=False)
    print("\nSaved leaf_support_diagnostics_summary.csv")

    if all_buckets:
        bucket_all_df = pd.concat(all_buckets, ignore_index=True)
        bucket_all_df.to_csv("leaf_support_diagnostics_buckets.csv", index=False)
        print("Saved leaf_support_diagnostics_buckets.csv")

    print(f"\n\n{'='*78}\nOVERALL CONCLUSION\n{'='*78}")
    counts = summary_df["verdict"].value_counts()
    print(counts.to_string())
    n_meaningful = counts.get("MEANINGFULLY DISCRIMINATIVE", 0)
    n_weak = counts.get("WEAK", 0)
    n_absent = counts.get("ABSENT / NOT RELIABLE", 0)
    n_insuff = counts.get("INSUFFICIENT EVIDENCE", 0)
    total = len(summary_df)

    if n_meaningful >= total / 2:
        print("\n>> Leaf support shows a meaningfully discriminative relationship with "
              "correctness across most horizons -- worth pursuing further (proper held-out "
              "validation, then a design for how it would be surfaced) before considering it "
              "for production.")
    elif n_absent + n_insuff >= total / 2:
        print("\n>> Leaf support does NOT show a reliable relationship with correctness across "
              "most horizons (or evidence is too sparse to tell). Recommend moving to Candidate "
              "2 (conformal prediction) rather than forcing this one.")
    else:
        print("\n>> Mixed result: leaf support helps on some horizons, not others. Worth "
              "checking per-horizon whether it beats existing confidence specifically where "
              "existing confidence was weakest (see the Spearman comparison column), before "
              "deciding whether a horizon-conditional approach makes sense.")


if __name__ == "__main__":
    run()
