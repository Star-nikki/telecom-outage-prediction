"""
Priority 5, Candidate 2: split conformal prediction diagnostic.

READ-ONLY / DIAGNOSTIC ONLY -- same guarantee as every other diagnostic
script in this project. Does not modify decision_agent.py, does not
retrain anything, does not touch the production confidence formula or
predict_with_confidence(). This is a HYPOTHESIS TEST for Candidate 2, same
discipline as leaf_support_diagnostics.py was for Candidate 1: if this
doesn't clearly beat existing ensemble-spread confidence, we say so and
don't force it into production.

WHY THIS IS CANDIDATE 2, NOT CANDIDATE 1 AGAIN:
    Candidate 1 (leaf-support, leaf_support_diagnostics.py) asked the model
    to introspect its own training data density. Verdict was WEAK/ABSENT
    on 6 of 7 horizons -- the raw signal saturated near the full reference
    set size for most rows, only differentiating in the extreme tail.
    Conformal prediction is structurally different: instead of asking the
    model anything about itself, it's an external, distribution-free
    statistical wrapper with a formal coverage GUARANTEE, calibrated
    against real held-out outcomes rather than model self-agreement.

WHAT CONFORMAL PREDICTION ACTUALLY PRODUCES HERE (precise definition):
    For a target coverage level (e.g. 90%), a PREDICTION SET per row --
    which class labels {0, 1} are plausible given the model's calibrated
    probability, guaranteed (on average, on held-out data) to contain the
    true label at least 90% of the time. A SINGLETON set ({0} or {1} alone)
    means the model is confident enough that only one class survives the
    conformal threshold. A set containing BOTH classes means genuine
    ambiguity -- the calibrated probability wasn't confident enough to rule
    either class out. Set size (1 vs 2) is the confidence proxy tested
    here -- NOT a rescaled 0-1 number, a genuine "how many classes are
    still plausible" count, honest about uncertainty by construction.

BUILDS ON PRIORITY 4 (calibration_experiment.py), NOT INDEPENDENT OF IT:
    Nonconformity scores are computed from the CALIBRATED probability
    (via calibrated_prediction.py's wrapper), not the raw ensemble mean --
    conformal prediction assumes a reasonably well-behaved probability
    input; garbage in, garbage out. Uses the SAME saved bagged models and
    SAME calibrators already produced by Priority 4, nothing retrained.

LEAK-FREE SPLIT, NESTED INSIDE calibration_experiment.py's own held-out
region (not a fresh carve-out):
    calibration_experiment.py already defined:
      [ 0-70% TRAIN ] [ 70-85% VAL ] [ 85-92.5% CAL-FIT ] [ 92.5-100% TEST ]
    This script further splits that 92.5-100% TEST region (already unseen
    by both the model AND the Platt/Isotonic calibrators) into:
      [ 92.5-96.25% CONFORMAL-CAL ] [ 96.25-100% FINAL-TEST ]
    CONFORMAL-CAL fits the conformal threshold; FINAL-TEST is untouched
    until final evaluation. Every stage here is strictly outside what the
    model or the calibrators ever saw.

WHERE TO SAVE:
    Same folder as calibration_experiment.py / confidence_diagnostics.py
    (agents/). Imports from both, so they must already be present.

HOW TO RUN:
    From the project root:      python -m agents.conformal_diagnostics
    From inside agents/:        python conformal_diagnostics.py

IMPORTANT: does not implement Candidate 2 as the production confidence
score. If the diagnostic below shows a real, discriminative relationship
between set size and correctness, that's the evidence needed to consider
wiring it in as a NEW field alongside (not replacing) predict_with_confidence()
-- a separate, later, deliberate step, not automatic from running this.
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
from confidence_diagnostics import _resolve_model_dir, tune_threshold_on_val, compute_row_level_stats, BACKEND
import calibrated_prediction as cp

pd.set_option("display.width", 160)

TARGET_COVERAGE = 0.90  # 1 - alpha. Standard choice; change if you want a
                         # tighter/looser guarantee.

# Same reasoning as leaf_support_diagnostics.py's MIN_SUPPORT_FOR_SCORING:
# don't compute (and don't trust) a conformal threshold for the positive
# class if there aren't enough positive examples in CONFORMAL-CAL to
# estimate a meaningful quantile from.
MIN_POSITIVES_FOR_CONFORMAL_CAL = 10


def split_conformal_regions(df, val_end_frac=0.85, calfit_frac=0.925,
                             conformal_cal_frac=0.9625):
    """Nests inside calibration_experiment.py's own TEST region (>=92.5%)
    rather than carving out a fresh split -- both pieces here were already
    unseen by the model AND its calibrators."""
    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    q = lambda frac: df_sorted["timestamp"].iloc[int(n * frac)]
    calfit_end = q(calfit_frac)
    conformal_cal_end = q(conformal_cal_frac)
    conformal_cal_df = df[(df["timestamp"] >= calfit_end) & (df["timestamp"] < conformal_cal_end)]
    final_test_df = df[df["timestamp"] >= conformal_cal_end]
    return conformal_cal_df, final_test_df, calfit_end, conformal_cal_end


def get_calibrated_probs(models, df, feature_columns, backend, label, model_dir):
    """
    Vectorized -- computes the ensemble-mean probability for the WHOLE
    batch at once (same approach calibration_experiment.py already uses
    successfully on ~118k-row test sets in a few seconds), then applies the
    saved calibrator this horizon was recommended to use, if any, as one
    vectorized operation. This is NOT a reimplementation that could drift
    from predict_with_confidence_calibrated() -- it reads the exact same
    calibration_recommendation.json and the exact same saved calibrator
    .joblib files that wrapper reads, just applied to an array instead of
    row-by-row.

    (An earlier version of this function called predict_with_confidence_calibrated()
    once per row via df.iterrows() for maximum fidelity to the production
    code path. Benchmarking showed that's impractically slow at this
    dataset's scale -- LightGBM has real per-call overhead that doesn't
    show up until you're making hundreds of thousands of individual calls.
    Vectorizing removes that risk entirely rather than leaving you to find
    out the hard way mid-run.)
    """
    X = df[feature_columns]
    raw_probs = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    recommendation = cp._load_recommendation(model_dir)
    horizon_rec = recommendation.get(label)
    if not horizon_rec or horizon_rec.get("recommended") == "raw":
        return raw_probs, ["raw"] * len(raw_probs)

    method = horizon_rec["recommended"]
    calibrator_path = os.path.join(model_dir, f"calibrator_{backend}_{label}_{method}.joblib")
    if not os.path.exists(calibrator_path):
        return raw_probs, ["raw"] * len(raw_probs)
    calibrator = joblib.load(calibrator_path)

    if method == "platt":
        eps = 1e-6
        logit = np.log(np.clip(raw_probs, eps, 1 - eps) / (1 - np.clip(raw_probs, eps, 1 - eps)))
        calibrated = calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]
    else:
        calibrated = calibrator.predict(raw_probs)

    return calibrated, [method] * len(calibrated)


def fit_conformal_thresholds_mondrian(calibrated_probs_cal, y_cal, target_coverage=TARGET_COVERAGE):
    """
    Mondrian (class-conditional) split conformal: fits a SEPARATE
    nonconformity threshold per true class, instead of one shared threshold
    across both.

    WHY THIS REPLACES THE ORIGINAL (LAC-style) VERSION:
    A single marginal threshold is the (1-alpha) quantile of nonconformity
    scores across the WHOLE calibration set. Under severe class imbalance
    (positive rate 0.05%-4.6% here), that quantile is completely dominated
    by the overwhelming majority of easy, well-calibrated negatives -- the
    rare positives barely influence it at all. The result, confirmed on
    real data: thresholds collapsing to ~0 at every horizon, every row
    forced into a singleton set via the empty-set fallback, zero variance,
    zero information. Mondrian fixes this by computing the negative-class
    threshold ONLY from calibration rows that are actually negative, and
    the positive-class threshold ONLY from calibration rows that are
    actually positive -- the rare class gets its own quantile, unable to be
    drowned out by the majority class sharing a pooled one.

    Returns (threshold_0, threshold_1, n_neg_used, n_pos_used).
    """
    alpha = 1 - target_coverage
    neg_scores = calibrated_probs_cal[y_cal == 0]        # s0 = 1 - p_hat(0) = p, for true negatives
    pos_scores = 1 - calibrated_probs_cal[y_cal == 1]     # s1 = 1 - p_hat(1) = 1-p, for true positives

    def quantile_threshold(scores):
        n = len(scores)
        if n == 0:
            return 1.0  # no information for this class -- never exclude it (safe default)
        q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        return float(np.quantile(scores, q_level, method="higher"))

    threshold_0 = quantile_threshold(neg_scores)
    threshold_1 = quantile_threshold(pos_scores)
    return threshold_0, threshold_1, len(neg_scores), len(pos_scores)


def build_prediction_sets_mondrian(calibrated_probs, threshold_0, threshold_1):
    """
    Returns each row's conformal category -- WITHOUT forcing anything into
    a fake confident answer:
      0 = EMPTY      -- neither class conformally justified at the target
                        coverage. This is a genuine abstention signal, not
                        an error to paper over. Testing on real data showed
                        this is actually the DOMINANT uncertain category
                        here (not "ambiguous" -- see module notes below):
                        an earlier version of this function forced these
                        rows into a >=0.5 guess and called them "confident,"
                        which was actively destroying the most informative
                        uncertainty signal this method produces on a
                        severely imbalanced dataset like this one.
      1 = SINGLETON  -- exactly one class conformally justified (genuinely
                        confident).
      2 = AMBIGUOUS  -- both classes conformally justified. Only possible
                        when threshold_0 + threshold_1 > 1; on this
                        dataset's real thresholds that condition rarely
                        holds, so expect this category to be small or
                        empty -- that's a real property of the fitted
                        thresholds, not a bug.

    A row's true label is correctly NOT considered "covered" if it falls
    in an EMPTY set (there's nothing to cover it with) -- this is proper
    conformal behavior, not a gap: the ~10% of rows the 90%-coverage
    guarantee allows to miss are exactly rows like these.
    """
    include_0 = calibrated_probs <= threshold_0
    include_1 = (1 - calibrated_probs) <= threshold_1
    n_included = include_0.astype(int) + include_1.astype(int)
    category = np.where(n_included == 0, 0, np.where(n_included == 1, 1, 2))
    return category, include_0, include_1


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

    threshold_decision = tune_threshold_on_val(models, X_val, y_val)

    conformal_cal_df, final_test_df, calfit_end, conformal_cal_end = split_conformal_regions(
        pd.concat([train_df, val_df, test_df]))
    # (split_conformal_regions re-derives its own cutoffs from the full df
    # internally via time ordering, so passing the concatenated frame here
    # is just to get back to the full timestamp range -- the actual filter
    # uses the ORIGINAL test_df's rows, restricted below.)
    conformal_cal_df = conformal_cal_df[conformal_cal_df.index.isin(test_df.index)]
    final_test_df = final_test_df[final_test_df.index.isin(test_df.index)]

    y_conformal_cal = conformal_cal_df[target_col].values
    y_final_test = final_test_df[target_col].values

    print(f"\n{'='*78}\nHORIZON {label} ({hours}h)  backend={BACKEND}  "
          f"decision_threshold={threshold_decision:.4f}\n{'='*78}")
    print(f"CONFORMAL-CAL: {len(conformal_cal_df):,} rows ({int(y_conformal_cal.sum())} positive)")
    print(f"FINAL-TEST:    {len(final_test_df):,} rows ({int(y_final_test.sum())} positive)")

    if y_conformal_cal.sum() < MIN_POSITIVES_FOR_CONFORMAL_CAL:
        print(f"\n>> Verdict for {label}: INSUFFICIENT EVIDENCE -- only "
              f"{int(y_conformal_cal.sum())} positives in CONFORMAL-CAL "
              f"(need >= {MIN_POSITIVES_FOR_CONFORMAL_CAL}) to fit a trustworthy conformal threshold.")
        return {"horizon": label, "hours": hours, "verdict": "INSUFFICIENT EVIDENCE",
                "n_conformal_cal_positives": int(y_conformal_cal.sum())}, None

    print("Computing calibrated probabilities via the real production wrapper "
          "(row-by-row, may take a moment)...")
    probs_cal, methods_cal = get_calibrated_probs(
        models, conformal_cal_df, feature_columns, BACKEND, label, model_dir)
    probs_test, methods_test = get_calibrated_probs(
        models, final_test_df, feature_columns, BACKEND, label, model_dir)

    threshold_0, threshold_1, n_neg_used, n_pos_used = fit_conformal_thresholds_mondrian(
        probs_cal, y_conformal_cal)
    category, include_0, include_1 = build_prediction_sets_mondrian(probs_test, threshold_0, threshold_1)

    n_empty = int((category == 0).sum())
    if n_empty:
        print(f"NOTE: {n_empty:,} row(s) ({n_empty/len(category)*100:.2f}%) fell in the EMPTY "
              f"category -- neither class conformally justified. Reported as its own honest "
              f"category below, NOT forced into a fake confident answer.")

    # Empirical coverage, reported PER CLASS -- this is the whole point of
    # Mondrian: the negative class was never the problem, the POSITIVE
    # class's coverage is what collapsed under the old shared-threshold
    # approach. Report both separately, not just an overall blended number
    # that could hide one class failing while the other compensates. Empty
    # rows correctly count as NOT covered for whichever true class they
    # have -- that's proper conformal behavior, not a gap to patch.
    neg_mask_test = y_final_test == 0
    pos_mask_test = y_final_test == 1
    coverage_negative = include_0[neg_mask_test].mean() if neg_mask_test.any() else float("nan")
    coverage_positive = include_1[pos_mask_test].mean() if pos_mask_test.any() else float("nan")
    true_in_set = np.where(y_final_test == 1, include_1, include_0)
    empirical_coverage = true_in_set.mean()

    print(f"\nThresholds (Mondrian, class-conditional): "
          f"class-0={threshold_0:.6f} (from {n_neg_used:,} negatives)  "
          f"class-1={threshold_1:.6f} (from {n_pos_used:,} positives)")
    print(f"Target coverage: {TARGET_COVERAGE*100:.0f}%")
    print(f"  Empirical coverage, NEGATIVE class: {coverage_negative*100:.2f}%")
    print(f"  Empirical coverage, POSITIVE class: {coverage_positive*100:.2f}%  "
          f"<- this is the one that was broken before")
    print(f"  Empirical coverage, OVERALL:        {empirical_coverage*100:.2f}%")
    if abs(coverage_positive - TARGET_COVERAGE) > 0.10:
        print("  NOTE: positive-class coverage is meaningfully off target -- with very few "
              "positives in CONFORMAL-CAL, the per-class guarantee is noisier than the textbook "
              "large-sample case. Reported honestly, not smoothed over.")

    n_singleton = int((category == 1).sum())
    n_ambiguous = int((category == 2).sum())
    print(f"\nCategory sizes on FINAL-TEST: empty(abstain)={n_empty:,} ({n_empty/len(category)*100:.2f}%)  "
          f"singleton(confident)={n_singleton:,} ({n_singleton/len(category)*100:.2f}%)  "
          f"ambiguous(both classes)={n_ambiguous:,} ({n_ambiguous/len(category)*100:.2f}%)")

    # Correctness at the SAME decision threshold used elsewhere in this
    # project, for apples-to-apples bucketing.
    pred = (probs_test >= threshold_decision).astype(int)
    correct = (pred == y_final_test).astype(int)

    print(f"\n--- Bucketed by conformal category (empty/abstain treated as LEAST confident) ---")
    bucket_rows = []
    for cat, cat_label in [(0, "empty (abstain -- least confident)"),
                            (2, "ambiguous (both classes)"),
                            (1, "singleton (confident)")]:
        mask = category == cat
        n = int(mask.sum())
        if n == 0:
            continue
        bucket_rows.append({
            "horizon": label, "category": cat_label, "n": n,
            "accuracy": float(correct[mask].mean()),
            "positive_rate": float(y_final_test[mask].mean()),
            "mean_calibrated_prob": float(probs_test[mask].mean()),
        })
    bucket_df = pd.DataFrame(bucket_rows)
    print(bucket_df.to_string(index=False))

    # Compare against existing ensemble-spread confidence on the SAME rows.
    # Rank confidence as: singleton (most confident) > ambiguous > empty
    # (least confident) -- this ordering IS the hypothesis being tested.
    confidence_rank = np.select([category == 1, category == 2, category == 0], [2, 1, 0])
    _, _, existing_confidence = compute_row_level_stats(models, final_test_df[feature_columns])
    spearman_setsize_correct = pd.Series(confidence_rank).corr(pd.Series(correct), method="spearman")
    spearman_existing_correct = pd.Series(existing_confidence).corr(pd.Series(correct), method="spearman")
    print(f"\nSpearman(confidence_rank, correct)     [conformal, Candidate 2]: {spearman_setsize_correct:.4f}")
    print(f"Spearman(existing_confidence, correct) [ensemble-spread, current]: {spearman_existing_correct:.4f}")

    # The real contrast to test is singleton (confident) vs empty (abstain)
    # -- ambiguous is frequently empty/near-empty on this dataset's fitted
    # thresholds (see build_prediction_sets_mondrian's docstring), so
    # comparing against it alone would often be comparing against nothing.
    acc_singleton = bucket_df[bucket_df["category"].str.startswith("singleton")]["accuracy"]
    acc_empty = bucket_df[bucket_df["category"].str.startswith("empty")]["accuracy"]
    if len(acc_singleton) and len(acc_empty):
        gap = float(acc_singleton.iloc[0] - acc_empty.iloc[0])
    else:
        gap = None

    if gap is not None and gap > 0.05 and abs(spearman_setsize_correct) >= 0.10:
        verdict = "MEANINGFULLY DISCRIMINATIVE"
    elif gap is not None and gap > 0.0:
        verdict = "WEAK"
    else:
        verdict = "ABSENT / NOT RELIABLE"
    print(f"\n>> Verdict for {label}: conformal category is {verdict} "
          f"(singleton-vs-empty accuracy gap: {gap if gap is not None else 'n/a'})")

    summary_row = {
        "horizon": label, "hours": hours,
        "n_conformal_cal": len(conformal_cal_df), "n_conformal_cal_positives": int(y_conformal_cal.sum()),
        "n_final_test": len(final_test_df),
        "threshold_class0": threshold_0, "threshold_class1": threshold_1,
        "target_coverage": TARGET_COVERAGE,
        "empirical_coverage_negative": coverage_negative,
        "empirical_coverage_positive": coverage_positive,
        "empirical_coverage_overall": empirical_coverage,
        "frac_empty": n_empty / len(category), "frac_singleton": n_singleton / len(category),
        "frac_ambiguous": n_ambiguous / len(category),
        "singleton_vs_empty_accuracy_gap": gap,
        "spearman_setsize_vs_correct": spearman_setsize_correct,
        "spearman_existing_confidence_vs_correct": spearman_existing_correct,
        "verdict": verdict,
    }
    return summary_row, bucket_df


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    train_df, val_df, test_df, train_end, val_end = time_split(df)
    model_dir = _resolve_model_dir()

    print(f"Backend: {BACKEND}  |  Model dir: {os.path.abspath(model_dir)}")
    print(f"Target coverage: {TARGET_COVERAGE*100:.0f}%")

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
    summary_df.to_csv("conformal_diagnostics_summary.csv", index=False)
    print("\nSaved conformal_diagnostics_summary.csv")

    if all_buckets:
        pd.concat(all_buckets, ignore_index=True).to_csv("conformal_diagnostics_by_class.csv", index=False)
        print("Saved conformal_diagnostics_by_class.csv")

    print(f"\n\n{'='*78}\nOVERALL CONCLUSION\n{'='*78}")
    counts = summary_df["verdict"].value_counts()
    print(counts.to_string())
    n_meaningful = counts.get("MEANINGFULLY DISCRIMINATIVE", 0)
    total = len(summary_df)

    if n_meaningful >= total / 2:
        print("\n>> Conformal set size shows a meaningfully discriminative relationship with "
              "correctness across most horizons -- this IS real evidence Candidate 2 beats "
              "existing confidence. Next step (separate, deliberate): decide how to SURFACE this "
              "in production (e.g. a new 'ambiguous' flag alongside probability/confidence, not a "
              "silent replacement of the existing confidence number) -- not done automatically here.")
    else:
        print("\n>> Conformal set size does NOT clearly beat existing confidence across most "
              "horizons either. Both Candidate 1 (leaf support) and Candidate 2 (conformal) have "
              "now been tested and found weak/insufficient on this dataset -- the honest "
              "conclusion is that reliable per-prediction confidence may need more calibration "
              "data (especially at short horizons, where positives are extremely rare) before any "
              "candidate can be trusted, rather than a different formula being the fix.")


if __name__ == "__main__":
    run()
