"""
Priority 5: stacked meta-confidence model.

RATIONALE: Candidate 1 (leaf support) and Candidate 2 (conformal) were each
tested independently and found weak (leaf support) or actively unreliable
in places (conformal, negative correlation at 12h/24h). But leaf support
was never WORSE than existing confidence at any horizon, and the two
signals measure genuinely different things (training-data density vs.
ensemble self-agreement). Weak, non-redundant signals can combine into
something stronger than any single one -- this script tests that directly
via a small logistic regression per horizon, rather than picking a single
"winner" candidate.

READ-ONLY / DIAGNOSTIC ONLY -- same guarantee as every other script here.
Trains ONLY a small logistic regression meta-model (4 features, one per
horizon) -- does not retrain decision_agent.py's models, does not modify
production code, does not touch predict_with_confidence().

FEATURES FED INTO THE META-MODEL (all already computed elsewhere, nothing
reimplemented):
    1. calibrated_prob        -- Priority 4's calibrated probability
    2. existing_confidence    -- current production confidence (1 - 2*std)
    3. raw_ensemble_std       -- pre-transform ensemble disagreement
    4. mean_leaf_support      -- Candidate 1's training-data density signal
    5. mean_pos_support       -- Candidate 1's positive-class-specific density

    Conformal (Candidate 2) is deliberately EXCLUDED -- it was negatively
    correlated with correctness at 12h/24h, i.e. actively worse than
    nothing at those horizons. Feeding a harmful signal into a stacker
    risks it dragging the combination down rather than only failing to help.

TARGET: "correct" (decision at the tuned threshold matches the true label),
same as every other confidence diagnostic in this project, PLUS a separate
report restricted to the PREDICTED-POSITIVE subset, since a confidence
score is operationally meaningless on obvious negatives -- nobody needs to
know how confident the model is that a clearly healthy tower is healthy.
The predicted-positive subset is where a trustworthy confidence signal
would actually change what an operator does.

LEAK-FREE SPLIT: reuses conformal_diagnostics.py's exact nested split
(META-CAL = 92.5-96.25%, META-TEST = 96.25-100%), both regions already
outside what the decision models AND the Priority 4 calibrators ever saw.
The logistic regression is fit on META-CAL only and evaluated once on
META-TEST.

WHERE TO SAVE:
    Same folder as confidence_diagnostics.py, leaf_support_diagnostics.py,
    conformal_diagnostics.py, calibrated_prediction.py (agents/). Imports
    from all four.

HOW TO RUN:
    From the project root:      python -m agents.meta_confidence_diagnostics
    From inside agents/:        python meta_confidence_diagnostics.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
from config import PREDICTION_HORIZONS
from calibration_diagnostics import load_features, time_split
from confidence_diagnostics import _resolve_model_dir, tune_threshold_on_val, compute_row_level_stats, BACKEND
from leaf_support_diagnostics import build_reference_sample, build_leaf_tables, score_leaf_support
from conformal_diagnostics import split_conformal_regions, get_calibrated_probs

pd.set_option("display.width", 160)

MIN_POSITIVES_FOR_META_FIT = 10


def run_horizon(label, hours, train_df, val_df, test_df, full_df, feature_columns, model_dir):
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
    if y_val.sum() == 0:
        print(f"\n[{label}] No positives in val split -- skipping.")
        return None
    threshold_decision = tune_threshold_on_val(models, X_val, y_val)

    meta_cal_df, meta_test_df, cal_end, test_end = split_conformal_regions(full_df)
    meta_cal_df = meta_cal_df[meta_cal_df.index.isin(test_df.index)]
    meta_test_df = meta_test_df[meta_test_df.index.isin(test_df.index)]

    y_cal = meta_cal_df[target_col].values
    y_test = meta_test_df[target_col].values
    if y_cal.sum() < MIN_POSITIVES_FOR_META_FIT or y_test.sum() == 0:
        print(f"\n[{label}] Only {int(y_cal.sum())} positives in META-CAL -- insufficient to fit "
              f"a trustworthy meta-model (need >= {MIN_POSITIVES_FOR_META_FIT}).")
        return {"horizon": label, "hours": hours, "verdict": "INSUFFICIENT EVIDENCE",
                "n_meta_cal_positives": int(y_cal.sum())}, None

    print(f"\n{'='*78}\nHORIZON {label} ({hours}h)  decision_threshold={threshold_decision:.4f}\n{'='*78}")
    print(f"META-CAL: {len(meta_cal_df):,} rows ({int(y_cal.sum())} positive)  |  "
          f"META-TEST: {len(meta_test_df):,} rows ({int(y_test.sum())} positive)")

    # ---- Feature 1: calibrated probability (vectorized, Priority 4) ----
    prob_cal, _ = get_calibrated_probs(models, meta_cal_df, feature_columns, BACKEND, label, model_dir)
    prob_test, _ = get_calibrated_probs(models, meta_test_df, feature_columns, BACKEND, label, model_dir)

    # ---- Features 2/3: existing confidence + raw ensemble std ----
    _, raw_std_cal, existing_conf_cal = compute_row_level_stats(models, meta_cal_df[feature_columns])
    _, raw_std_test, existing_conf_test = compute_row_level_stats(models, meta_test_df[feature_columns])

    # ---- Features 4/5: leaf support (Candidate 1), reference built from TRAIN only ----
    ref_df = build_reference_sample(train_df, target_col)
    X_ref = ref_df[feature_columns]
    y_ref_arr = ref_df[target_col].values.astype(np.float64)

    print("Building leaf-support lookup tables (once per bagged model, reused for CAL/TEST)...")
    leaf_tables_per_model = [build_leaf_tables(m, X_ref, y_ref_arr) for m in models]

    def leaf_features_for(df_subset):
        per_model_support, per_model_pos = [], []
        for m, (total_tables, pos_tables) in zip(models, leaf_tables_per_model):
            support, pos_support = score_leaf_support(m, df_subset[feature_columns], total_tables, pos_tables)
            per_model_support.append(support.mean(axis=1))
            per_model_pos.append(pos_support.mean(axis=1))
        mean_support = np.mean(per_model_support, axis=0)
        mean_pos_support = np.mean(per_model_pos, axis=0)
        return mean_support, mean_pos_support

    print("Computing leaf-support features for META-CAL and META-TEST...")
    support_cal, pos_support_cal = leaf_features_for(meta_cal_df)
    support_test, pos_support_test = leaf_features_for(meta_test_df)

    # ---- Assemble feature matrices ----
    X_meta_cal = np.column_stack([prob_cal, existing_conf_cal, raw_std_cal,
                                   np.log1p(support_cal), np.log1p(pos_support_cal)])
    X_meta_test = np.column_stack([prob_test, existing_conf_test, raw_std_test,
                                    np.log1p(support_test), np.log1p(pos_support_test)])
    feature_names = ["calibrated_prob", "existing_confidence", "raw_ensemble_std",
                      "log_mean_leaf_support", "log_mean_pos_support"]

    pred_cal = (prob_cal >= threshold_decision).astype(int)
    correct_cal = (pred_cal == y_cal).astype(int)
    pred_test = (prob_test >= threshold_decision).astype(int)
    correct_test = (pred_test == y_test).astype(int)

    if len(np.unique(correct_cal)) < 2:
        print(f"\n[{label}] META-CAL has only one 'correct' outcome value -- cannot fit a "
              f"discriminative meta-model (nothing to learn from).")
        return {"horizon": label, "hours": hours, "verdict": "INSUFFICIENT EVIDENCE",
                "n_meta_cal_positives": int(y_cal.sum())}, None

    scaler = StandardScaler().fit(X_meta_cal)
    X_meta_cal_s = scaler.transform(X_meta_cal)
    X_meta_test_s = scaler.transform(X_meta_test)

    meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta_model.fit(X_meta_cal_s, correct_cal)
    meta_confidence_test = meta_model.predict_proba(X_meta_test_s)[:, 1]

    print("\nMeta-model coefficients, fit on FULL population (standardized):")
    for name, coef in zip(feature_names, meta_model.coef_[0]):
        print(f"  {name:24s}: {coef:+.4f}")

    # ---- Evaluate: full population (reference only -- see caveat below) ----
    spearman_meta_full = pd.Series(meta_confidence_test).corr(pd.Series(correct_test), method="spearman")
    spearman_existing_full = pd.Series(existing_conf_test).corr(pd.Series(correct_test), method="spearman")
    try:
        auc_meta_full = roc_auc_score(correct_test, meta_confidence_test) if len(np.unique(correct_test)) > 1 else float("nan")
    except ValueError:
        auc_meta_full = float("nan")

    print(f"\n--- Full population (META-TEST, n={len(y_test):,}) -- REFERENCE ONLY, see caveat below ---")
    print(f"  Spearman(meta_confidence, correct):     {spearman_meta_full:.4f}")
    print(f"  Spearman(existing_confidence, correct): {spearman_existing_full:.4f}")
    print(f"  CAVEAT: this population is >99% trivial true-negatives -- the same accuracy-under-")
    print(f"  imbalance distortion flagged in every earlier diagnostic. NOT used for the verdict below.")

    # ---- The evaluation that actually matters: fit AND evaluate on the
    # ALERT-ONLY subset (rows the decision threshold actually flags). A
    # model fit on the full population can learn a relationship that's
    # true on trivial negatives but INVERTS among alerts -- exactly what
    # happened on first pass here (full-pop fit looked fine, but scored
    # -0.40 to -0.29 Spearman on the alert subset it would actually be
    # used on). Refitting specifically on alerts, mirroring how this would
    # actually be deployed, is the honest test.
    pos_pred_mask_cal = pred_cal == 1
    pos_pred_mask_test = pred_test == 1
    n_pos_pred_cal = int(pos_pred_mask_cal.sum())
    n_pos_pred_test = int(pos_pred_mask_test.sum())

    result_pos_subset = None
    verdict = "INSUFFICIENT ALERTS TO EVALUATE"
    artifacts = None
    if (n_pos_pred_cal >= 30 and n_pos_pred_test >= 10
            and len(np.unique(correct_cal[pos_pred_mask_cal])) > 1
            and len(np.unique(correct_test[pos_pred_mask_test])) > 1):
        alert_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        alert_model.fit(X_meta_cal_s[pos_pred_mask_cal], correct_cal[pos_pred_mask_cal])
        meta_confidence_alerts = alert_model.predict_proba(X_meta_test_s[pos_pred_mask_test])[:, 1]

        spearman_meta_pos = pd.Series(meta_confidence_alerts).corr(
            pd.Series(correct_test[pos_pred_mask_test]), method="spearman")
        spearman_existing_pos = pd.Series(existing_conf_test[pos_pred_mask_test]).corr(
            pd.Series(correct_test[pos_pred_mask_test]), method="spearman")

        print(f"\nMeta-model coefficients, fit ON ALERTS ONLY ({n_pos_pred_cal:,} rows, standardized):")
        for name, coef in zip(feature_names, alert_model.coef_[0]):
            print(f"  {name:24s}: {coef:+.4f}")

        print(f"\n--- ALERT-ONLY subset (n={n_pos_pred_test:,}) -- fit AND evaluated on alerts, "
              f"the operationally relevant comparison ---")
        print(f"  Spearman(meta_confidence, correct):     {spearman_meta_pos:.4f}")
        print(f"  Spearman(existing_confidence, correct): {spearman_existing_pos:.4f}")

        result_pos_subset = {"n_pos_pred": n_pos_pred_test, "spearman_meta_pos": spearman_meta_pos,
                              "spearman_existing_pos": spearman_existing_pos}

        if spearman_meta_pos >= spearman_existing_pos + 0.05:
            verdict = "BEATS EXISTING ON ALERTS"
        elif spearman_meta_pos >= spearman_existing_pos - 0.05:
            verdict = "COMPARABLE TO EXISTING ON ALERTS"
        else:
            verdict = "WORSE THAN EXISTING ON ALERTS"

        if verdict == "BEATS EXISTING ON ALERTS":
            artifacts = {
                "alert_model": alert_model, "scaler": scaler,
                "leaf_tables_per_model": leaf_tables_per_model,
                "threshold_decision": threshold_decision, "feature_names": feature_names,
                "spearman_meta_alerts": spearman_meta_pos, "spearman_existing_alerts": spearman_existing_pos,
                "n_train_alerts": n_pos_pred_cal,
            }
    else:
        print(f"\n--- ALERT-ONLY subset: too few alerts in META-CAL ({n_pos_pred_cal}) or META-TEST "
              f"({n_pos_pred_test}), or a single outcome class -- cannot fit/evaluate a trustworthy "
              f"alert-specific model for this horizon.")

    print(f"\n>> Verdict for {label}: {verdict}")

    summary_row = {
        "horizon": label, "hours": hours, "n_meta_cal_positives": int(y_cal.sum()),
        "n_meta_test": len(y_test), "n_pos_pred_cal": n_pos_pred_cal, "n_pos_pred_test": n_pos_pred_test,
        "spearman_meta_full": spearman_meta_full, "spearman_existing_full": spearman_existing_full,
        "auc_meta_full": auc_meta_full,
        "spearman_meta_alerts": result_pos_subset["spearman_meta_pos"] if result_pos_subset else np.nan,
        "spearman_existing_alerts": result_pos_subset["spearman_existing_pos"] if result_pos_subset else np.nan,
        "verdict": verdict,
    }
    for name, coef in zip(feature_names, meta_model.coef_[0]):
        summary_row[f"coef_fullpop_{name}"] = coef
    return summary_row, artifacts


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    train_df, val_df, test_df, train_end, val_end = time_split(df)
    model_dir = _resolve_model_dir()

    print(f"Backend: {BACKEND}  |  Model dir: {os.path.abspath(model_dir)}")

    all_summaries = []
    recommendation = {}
    for label, hours in PREDICTION_HORIZONS:
        result = run_horizon(label, hours, train_df, val_df, test_df, df, feature_columns, model_dir)
        if result is None:
            continue
        summary, artifacts = result
        all_summaries.append(summary)

        if artifacts is not None:
            model_path = os.path.join(model_dir, f"meta_confidence_model_{BACKEND}_{label}.joblib")
            scaler_path = os.path.join(model_dir, f"meta_confidence_scaler_{BACKEND}_{label}.joblib")
            leaf_path = os.path.join(model_dir, f"meta_confidence_leaftables_{BACKEND}_{label}.joblib")
            joblib.dump(artifacts["alert_model"], model_path)
            joblib.dump(artifacts["scaler"], scaler_path)
            joblib.dump(artifacts["leaf_tables_per_model"], leaf_path)
            print(f"\nSaved meta-confidence artifacts for {label}: {model_path}")
            recommendation[label] = {
                "use_meta": True,
                "feature_names": artifacts["feature_names"],
                "threshold_decision": artifacts["threshold_decision"],
                "spearman_meta_alerts": artifacts["spearman_meta_alerts"],
                "spearman_existing_alerts": artifacts["spearman_existing_alerts"],
                "n_train_alerts": artifacts["n_train_alerts"],
                "reason": (f"Alert-only Spearman {artifacts['spearman_meta_alerts']:.4f} vs existing "
                           f"{artifacts['spearman_existing_alerts']:.4f}, fit on {artifacts['n_train_alerts']} "
                           f"alerts."),
            }
        else:
            recommendation[label] = {"use_meta": False, "reason": summary.get("verdict", "not evaluated")}

    if not all_summaries:
        print("\nNo horizons produced results.")
        return

    print(f"\n\n{'='*78}\nSUMMARY ACROSS ALL HORIZONS\n{'='*78}")
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("meta_confidence_diagnostics_summary.csv", index=False)
    print("\nSaved meta_confidence_diagnostics_summary.csv")

    rec_path = os.path.join(model_dir, "meta_confidence_recommendation.json")
    with open(rec_path, "w") as f:
        json.dump(recommendation, f, indent=2)
    print(f"Saved recommendation file: {rec_path}")
    print("meta_confidence_prediction.py will automatically read this file and apply the "
          "alert-only model only where it's recommended above -- no manual step needed.")

    print(f"\n\n{'='*78}\nOVERALL CONCLUSION\n{'='*78}")
    if "verdict" in summary_df.columns:
        print(summary_df["verdict"].value_counts().to_string())
    n_beats = (summary_df["verdict"] == "BEATS EXISTING ON ALERTS").sum()
    n_comparable = (summary_df["verdict"] == "COMPARABLE TO EXISTING ON ALERTS").sum()
    n_worse = (summary_df["verdict"] == "WORSE THAN EXISTING ON ALERTS").sum()
    total_evaluable = n_beats + n_comparable + n_worse

    if total_evaluable == 0:
        print("\n>> No horizon had enough alerts in both META-CAL and META-TEST to run the "
              "operationally-relevant evaluation at all -- inconclusive, not evidence either way.")
    elif n_worse > total_evaluable / 2:
        print(f"\n>> Stacking is WORSE than existing confidence on the alert-only subset at "
              f"{n_worse}/{total_evaluable} evaluable horizons -- fitting on the full population "
              f"(dominated by trivial negatives) does not transfer to the alerts a confidence score "
              f"would actually be used on. The honest conclusion: this particular stacked model should "
              f"NOT replace existing confidence. Existing ensemble-spread confidence, weak as it is, "
              f"remains the least-bad option tested across Candidates 1, 2, and this stacking attempt.")
    elif n_beats >= total_evaluable / 2:
        print(f"\n>> Stacking beats existing confidence on the alert-only subset at "
              f"{n_beats}/{total_evaluable} evaluable horizons -- real evidence worth pursuing "
              f"further before considering production use.")
    else:
        print(f"\n>> Mixed/comparable result on alerts ({n_beats} beat, {n_comparable} comparable, "
              f"{n_worse} worse out of {total_evaluable} evaluable horizons) -- no clear winner. "
              f"Artifacts were still saved for any horizon that individually beat existing confidence, "
              f"so production wiring only activates where it earned it.")


if __name__ == "__main__":
    run()
