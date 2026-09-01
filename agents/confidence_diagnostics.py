"""
Priority 5: confidence diagnostics.

READ-ONLY / DIAGNOSTIC ONLY -- same guarantee as calibration_diagnostics.py
and calibration_experiment.py. Does not modify decision_agent.py, does not
retrain anything, does not touch predict_with_confidence() or the
production confidence formula. Loads the already-saved bagged ensembles
via joblib.load() and reuses the real feature pipeline (decision_agent.py)
and data-loading/time-split logic (calibration_diagnostics.py) so results
reflect the project's actual pipeline, not a reimplementation.

WHERE TO SAVE:
    Same folder as decision_agent.py, calibration_diagnostics.py and
    calibration_experiment.py (agents/). Imports from calibration_diagnostics.py,
    so that file must already be sitting next to it -- it already is.

HOW TO RUN:
    From the project root:      python -m agents.confidence_diagnostics
    From inside agents/:        python confidence_diagnostics.py

WHAT THIS DOES NOT DO:
    Does not propose or apply a new confidence formula. This script only
    answers: (1) what does the current 1 - 2*std confidence value actually
    look like across real predictions, (2) is the raw ensemble disagreement
    it's derived from meaningfully spread out or clustered near zero, and
    (3) does higher confidence actually correspond to more reliable
    predictions. Any change to the formula is a separate, later step.

REQUIRES:
    The bagged ensemble .joblib files must already exist in MODEL_DIR
    (produced by `python decision_agent.py --mode train ...`). This script
    does not train anything -- if a horizon's model file is missing, it is
    skipped with a message, not silently substituted.
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
from config import PREDICTION_HORIZONS
from calibration_diagnostics import load_features, brier_score, time_split

pd.set_option("display.width", 160)

BACKEND = "lightgbm"  # which saved model backend to diagnose (must match a
                       # backend you actually trained -- change if needed)

# Confidence buckets to report on. These are illustrative ranges, not a
# claim about where "good" confidence should fall -- that's exactly what
# this script is trying to find out.
CONFIDENCE_BUCKETS = [
    (0.00, 0.50, "0.00-0.50 (low)"),
    (0.50, 0.70, "0.50-0.70 (low-mid)"),
    (0.70, 0.85, "0.70-0.85 (mid)"),
    (0.85, 0.95, "0.85-0.95 (high)"),
    (0.95, 1.001, "0.95-1.00 (very high)"),
]


def _resolve_model_dir():
    """Same problem as calibration_diagnostics.py's _resolve_data_path():
    da.MODEL_DIR is relative, so it only resolves correctly if this script
    is run from the exact directory decision_agent.py itself expects. Try
    da.MODEL_DIR as-is first, then common alternates, so this works whether
    you run `python -m agents.confidence_diagnostics` from the project root
    or `python confidence_diagnostics.py` from inside agents/."""
    candidates = [
        da.MODEL_DIR,
        os.path.join("models"),
        os.path.join("agents", "models"),
        os.path.join("..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Could not find a models/ directory. Tried: {candidates}. "
        f"Edit _resolve_model_dir() or set MODEL_DIR_OVERRIDE below directly."
    )


def tune_threshold_on_val(models, X_val, y_val):
    """Mirrors decision_agent.py's train_all_horizons() threshold-tuning
    block exactly (same logic, just not retraining) so the threshold used
    here for 'correctness' matches what production would actually pick."""
    from sklearn.metrics import precision_recall_curve

    val_proba = da.ensemble_predict_proba(models, X_val)
    if y_val.sum() == 0:
        return 0.5

    prec_v, rec_v, thresholds_v = precision_recall_curve(y_val, val_proba)
    if len(thresholds_v) == 0:
        return 0.5
    f1_v = np.where((prec_v + rec_v) > 0, 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12), 0)
    mask = prec_v[:-1] >= 0.5
    if mask.any():
        idx = np.argmax(rec_v[:-1][mask])
        return float(thresholds_v[mask][idx])
    return float(thresholds_v[np.argmax(f1_v[:-1])])


def compute_row_level_stats(models, X):
    """
    Per-row: probability (ensemble mean), raw_std (raw ensemble
    disagreement, no transform), confidence (the ACTUAL production
    formula: 1 - 2*std for n_bagging>1, clipped to [0,1]).
    """
    member_preds = np.array([m.predict_proba(X)[:, 1] for m in models])  # shape (n_bagging, n_rows)
    probability = member_preds.mean(axis=0)
    raw_std = member_preds.std(axis=0)

    if len(models) > 1:
        agreement = 1 - (raw_std * 2)
        confidence = np.clip(agreement, 0, 1)
    else:
        confidence = np.clip(np.abs(probability - 0.5) * 2, 0, 1)

    return probability, raw_std, confidence


def bucket_label(conf_value):
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= conf_value < hi:
            return label
    return CONFIDENCE_BUCKETS[-1][2]  # catches confidence == 1.0 exactly


def run_horizon(label, hours, train_df, val_df, test_df, feature_columns, model_dir):
    model_path = os.path.join(model_dir, f"decision_model_{BACKEND}_{label}.joblib")
    if not os.path.exists(model_path):
        print(f"\n[{label}] No saved model at {model_path} -- skipping.")
        return None

    models = joblib.load(model_path)
    n_bagging = len(models)
    if n_bagging <= 1:
        print(f"\n[{label}] Only {n_bagging} model(s) in the saved ensemble -- confidence here "
              f"is the margin-based fallback, not ensemble-agreement. Diagnostics below still "
              f"run, but the '1 - 2*std' analysis doesn't apply to this horizon's saved file.")

    target_col = f"outage_in_next_{label}"
    if target_col not in train_df.columns:
        print(f"\n[{label}] No column {target_col} in the loaded data -- skipping.")
        return None

    X_val, y_val = val_df[feature_columns], val_df[target_col]
    X_test, y_test = test_df[feature_columns], test_df[target_col]

    if y_test.sum() == 0:
        print(f"\n[{label}] No positives in test split -- skipping (can't evaluate reliability).")
        return None

    threshold = tune_threshold_on_val(models, X_val, y_val)

    probability, raw_std, confidence = compute_row_level_stats(models, X_test)
    y_test_arr = y_test.values
    pred = (probability >= threshold).astype(int)
    correct = (pred == y_test_arr).astype(int)
    sq_err = (probability - y_test_arr) ** 2  # per-row Brier contribution

    print(f"\n{'='*78}\nHORIZON {label} ({hours}h)  backend={BACKEND}  n_bagging={n_bagging}  "
          f"tuned_threshold={threshold:.4f}\n{'='*78}")
    print(f"Test rows: {len(y_test_arr):,} | positives: {int(y_test_arr.sum())} | "
          f"overall accuracy at threshold: {correct.mean():.4f} | overall Brier: {sq_err.mean():.6f}")

    # ---- 1. Confidence distribution ----
    print(f"\n--- 1. Confidence distribution ---")
    print(f"  min={confidence.min():.4f}  max={confidence.max():.4f}  "
          f"mean={confidence.mean():.4f}  median={np.median(confidence):.4f}  "
          f"std={confidence.std():.4f}")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  Percentiles: " + "  ".join(f"p{p}={np.percentile(confidence, p):.4f}" for p in pcts))
    for lo, hi, blabel in CONFIDENCE_BUCKETS:
        frac = ((confidence >= lo) & (confidence < hi)).mean()
        print(f"  fraction in [{blabel:>20}]: {frac*100:6.2f}%")

    # ---- 2. Raw ensemble disagreement ----
    print(f"\n--- 2. Raw ensemble disagreement (pre-transform) ---")
    print(f"  mean_probability={probability.mean():.4f}")
    print(f"  raw_std: min={raw_std.min():.5f}  max={raw_std.max():.5f}  "
          f"mean={raw_std.mean():.5f}  median={np.median(raw_std):.5f}")
    print("  raw_std percentiles: " + "  ".join(
        f"p{p}={np.percentile(raw_std, p):.5f}" for p in pcts))
    theoretical_max_std = 0.5
    frac_near_zero = (raw_std < 0.02).mean()
    frac_below_10pct_of_max = (raw_std < 0.1 * theoretical_max_std).mean()
    print(f"  fraction with raw_std < 0.02 (near-total agreement): {frac_near_zero*100:.2f}%")
    print(f"  fraction with raw_std < {0.1*theoretical_max_std:.2f} "
          f"(<10% of the theoretical max std of {theoretical_max_std}): {frac_below_10pct_of_max*100:.2f}%")

    # ---- 3. Confidence buckets ----
    print(f"\n--- 3. Confidence buckets: count, accuracy, Brier, mean prob, mean raw_std ---")
    bucket_rows = []
    labels_arr = np.array([bucket_label(c) for c in confidence])
    for lo, hi, blabel in CONFIDENCE_BUCKETS:
        mask = labels_arr == blabel
        n = int(mask.sum())
        if n == 0:
            bucket_rows.append({
                "horizon": label, "confidence_bucket": blabel, "n": 0,
                "accuracy": np.nan, "brier": np.nan,
                "mean_predicted_prob": np.nan, "mean_raw_std": np.nan,
                "n_positive": 0,
            })
            continue
        bucket_rows.append({
            "horizon": label,
            "confidence_bucket": blabel,
            "n": n,
            "accuracy": float(correct[mask].mean()),
            "brier": float(sq_err[mask].mean()),
            "mean_predicted_prob": float(probability[mask].mean()),
            "mean_raw_std": float(raw_std[mask].mean()),
            "n_positive": int(y_test_arr[mask].sum()),
        })
    bucket_df = pd.DataFrame(bucket_rows)
    print(bucket_df.to_string(index=False))

    # ---- 4. Confidence vs correctness ----
    print(f"\n--- 4. Confidence vs correctness ---")
    mean_conf_correct = confidence[correct == 1].mean() if (correct == 1).any() else float("nan")
    mean_conf_incorrect = confidence[correct == 0].mean() if (correct == 0).any() else float("nan")
    print(f"  mean confidence on CORRECT predictions:   {mean_conf_correct:.4f}")
    print(f"  mean confidence on INCORRECT predictions: {mean_conf_incorrect:.4f}")
    print(f"  gap (correct - incorrect): {mean_conf_correct - mean_conf_incorrect:+.4f}")

    # Spearman correlation between confidence and correctness (rank-based,
    # robust to the fact that 'correct' is binary and confidence is skewed).
    spearman_corr = pd.Series(confidence).corr(pd.Series(correct), method="spearman")
    print(f"  Spearman correlation(confidence, correct): {spearman_corr:.4f}")

    # Is bucket accuracy actually monotonic (or close to it) across
    # increasing confidence, using only non-empty buckets?
    non_empty = bucket_df.dropna(subset=["accuracy"])
    monotonic_trend = None
    if len(non_empty) >= 2:
        accs = non_empty["accuracy"].values
        diffs = np.diff(accs)
        monotonic_trend = "increasing" if (diffs >= -1e-9).all() else (
            "decreasing" if (diffs <= 1e-9).all() else "non-monotonic")
    print(f"  bucket accuracy trend across increasing confidence: {monotonic_trend}")

    if abs(spearman_corr) >= 0.15 and monotonic_trend == "increasing":
        verdict = "MEANINGFULLY DISCRIMINATIVE"
    elif abs(spearman_corr) >= 0.05:
        verdict = "WEAK"
    else:
        verdict = "ABSENT / NOT RELIABLE"
    print(f"  >> Verdict for {label}: confidence-correctness relationship is {verdict}")

    # ---- 5. Compression check ----
    print(f"\n--- 5. Compression check ---")
    frac_above_90 = (confidence > 0.90).mean()
    frac_above_95 = (confidence > 0.95).mean()
    frac_above_99 = (confidence > 0.99).mean()
    print(f"  fraction of predictions with confidence > 0.90: {frac_above_90*100:.2f}%")
    print(f"  fraction of predictions with confidence > 0.95: {frac_above_95*100:.2f}%")
    print(f"  fraction of predictions with confidence > 0.99: {frac_above_99*100:.2f}%")
    print(f"  (raw_std distribution above shows whether this is because the ensemble genuinely\n"
          f"   agrees almost all the time, or because 1 - 2*std compresses even moderate\n"
          f"   disagreement into the high range -- e.g. raw_std=0.15 alone still yields\n"
          f"   confidence=0.70, and raw_std=0.05 already yields confidence=0.90.)")
    compressed = frac_above_90 > 0.80
    print(f"  >> Compression flag for {label}: "
          f"{'LIKELY COMPRESSED (>80% of predictions land above 0.90 confidence)' if compressed else 'not obviously compressed'}")

    row_summary = {
        "horizon": label, "hours": hours, "n_bagging": n_bagging,
        "n_test": len(y_test_arr), "n_positive_test": int(y_test_arr.sum()),
        "threshold": threshold, "overall_accuracy": float(correct.mean()),
        "overall_brier": float(sq_err.mean()),
        "confidence_mean": float(confidence.mean()), "confidence_median": float(np.median(confidence)),
        "confidence_std": float(confidence.std()),
        "raw_std_mean": float(raw_std.mean()), "raw_std_median": float(np.median(raw_std)),
        "frac_confidence_gt_090": float(frac_above_90), "frac_confidence_gt_095": float(frac_above_95),
        "mean_conf_correct": float(mean_conf_correct), "mean_conf_incorrect": float(mean_conf_incorrect),
        "spearman_corr_conf_vs_correct": float(spearman_corr),
        "bucket_accuracy_trend": monotonic_trend,
        "verdict": verdict,
        "compression_flag": compressed,
    }
    return row_summary, bucket_df


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    train_df, val_df, test_df, train_end, val_end = time_split(df)

    model_dir = _resolve_model_dir()

    print(f"Backend under diagnosis: {BACKEND}")
    print(f"Model dir resolved to: {os.path.abspath(model_dir)}")
    print(f"Train: {len(train_df):,} rows (< {train_end})")
    print(f"Val:   {len(val_df):,} rows ({train_end} to {val_end}) -- used only to tune threshold, same as production")
    print(f"Test:  {len(test_df):,} rows (>= {val_end}) -- all diagnostics below run on this split\n")

    all_summaries = []
    all_bucket_rows = []
    for label, hours in PREDICTION_HORIZONS:
        result = run_horizon(label, hours, train_df, val_df, test_df, feature_columns, model_dir)
        if result is None:
            continue
        summary, bucket_df = result
        all_summaries.append(summary)
        all_bucket_rows.append(bucket_df)

    if not all_summaries:
        print(f"\nNo horizons produced results -- check that models exist in "
              f"{os.path.abspath(model_dir)}/ for backend '{BACKEND}'.")
        return

    print(f"\n\n{'='*78}\nSUMMARY ACROSS ALL HORIZONS\n{'='*78}")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("confidence_diagnostics_summary.csv", index=False)
    print("\nSaved confidence_diagnostics_summary.csv")

    bucket_all_df = pd.concat(all_bucket_rows, ignore_index=True)
    bucket_all_df.to_csv("confidence_diagnostics_buckets.csv", index=False)
    print("Saved confidence_diagnostics_buckets.csv")

    # ---- overall conclusion across horizons ----
    print(f"\n\n{'='*78}\nOVERALL CONCLUSION\n{'='*78}")
    n_meaningful = (summary_df["verdict"] == "MEANINGFULLY DISCRIMINATIVE").sum()
    n_weak = (summary_df["verdict"] == "WEAK").sum()
    n_absent = (summary_df["verdict"] == "ABSENT / NOT RELIABLE").sum()
    n_compressed = summary_df["compression_flag"].sum()
    total = len(summary_df)
    print(f"Horizons evaluated: {total}")
    print(f"  Meaningfully discriminative: {n_meaningful}")
    print(f"  Weak relationship:           {n_weak}")
    print(f"  Absent / not reliable:       {n_absent}")
    print(f"  Flagged as compressed (>80% of predictions above 0.90 confidence): {n_compressed} / {total}")

    if n_meaningful == total and n_compressed == 0:
        print("\n>> Current confidence is meaningfully discriminative and higher confidence "
              "corresponds to higher reliability, across all evaluated horizons.")
    elif n_absent >= total / 2 or n_compressed >= total / 2:
        print("\n>> Current confidence is highly compressed and/or does not reliably distinguish "
              "correct from incorrect predictions, across most evaluated horizons.")
    else:
        print("\n>> Mixed result: confidence is discriminative for some horizons but weak, absent, "
              "or compressed for others. See the per-horizon verdicts above and the saved CSVs "
              "before deciding whether/how to change the formula.")


if __name__ == "__main__":
    run()
