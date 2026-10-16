"""
Priority 4, FIXED: calibration experiment -- no retraining.

This replaces the earlier calibration_experiment.py, which retrained
LightGBM from scratch (fit_raw_model()/train_one_horizon()) purely to run
calibration analysis -- wasted effort, since calibration only needs an
already-trained model's predict_proba() output on held-out data. This
version LOADS the saved bagged ensembles via joblib.load() and never
trains anything.

WHY THIS IS STILL LEAK-FREE WITHOUT RETRAINING:
    Production's train_all_horizons() splits data 70% train / 15% val /
    15% test by TIME (see decision_agent.py's own train_end/val_end
    cutoffs). The saved models only ever saw the first 85% of the
    timeline. This script re-derives that exact same 85% cutoff, then
    splits the REMAINING 15% (which the model never trained on) into two
    further time-ordered pieces:

        [ 0-70% TRAIN ] [ 70-85% VAL ] [ 85-92.5% CAL-FIT ] [ 92.5-100% TEST ]
         (model saw this)  (model saw this)   |                      |
                                       fit Platt/Isotonic      FINAL, evaluated
                                       calibrators here         exactly once here

    Both CAL-FIT and TEST are entirely outside what the saved model
    trained on, so this is genuinely leak-free -- it just reuses the
    model's own held-out region instead of carving out a fresh one via
    another training run.

WHAT IT DOES:
    For each horizon, loads the saved bagged ensemble, computes raw
    (uncalibrated) ensemble-mean probability on CAL-FIT and TEST, fits
    Platt and Isotonic calibrators on CAL-FIT, evaluates raw vs Platt vs
    Isotonic on TEST (Brier / ECE / LogLoss / AUC / PR-AUC -- same metrics
    as before).

    Then, automatically (this is the "jugaad" part, by design, not
    accidental): decides per horizon whether calibration is worth using in
    production, using the same rule stated back in the Priority 4
    diagnosis: a calibrated variant is recommended only if it improves
    Brier score AND does not meaningfully hurt discrimination (AUC drop
    tolerance below). If neither Platt nor Isotonic clears that bar, "raw"
    (no calibration) is recommended -- calibration is never forced on a
    horizon it doesn't help.

    Saves:
      - calibration_experiment_summary.csv       (same as before)
      - agents/models/calibrator_{backend}_{label}_platt.joblib     (per horizon, if fitted)
      - agents/models/calibrator_{backend}_{label}_isotonic.joblib  (per horizon, if fitted)
      - agents/models/calibration_recommendation.json  (per-horizon decision + why)

    calibrated_prediction.py (separate file) reads that JSON + the saved
    calibrator files at inference time. decision_agent.py itself is NOT
    modified -- calibration is applied via an opt-in wrapper, so nothing
    about existing behavior changes unless you explicitly call the new
    wrapper function.

HOW TO RUN:
    From the project root:      python -m agents.calibration_experiment
    From inside agents/:        python calibration_experiment.py
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
from config import PREDICTION_HORIZONS
from calibration_diagnostics import (
    load_features, brier_score, calibration_curve_manual, expected_calibration_error,
)

pd.set_option("display.width", 160)

BACKEND = "lightgbm"

# AUC is allowed to drop by at most this much for a calibrated variant to
# still be "recommended" -- a small drop is expected/harmless (calibration
# reshapes the probability scale, it doesn't rerank much), but a real drop
# means the calibrator distorted the model's actual discrimination, which
# is worse than doing nothing.
MAX_ACCEPTABLE_AUC_DROP = 0.01


def _resolve_model_dir():
    candidates = [
        da.MODEL_DIR, "models", os.path.join("agents", "models"), os.path.join("..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Could not find a models/ directory. Tried: {candidates}")


def ensemble_proba(models, X):
    preds = np.array([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=0)


def split_for_calibration(df, val_end_frac=0.85, calfit_frac=0.925):
    """Re-derives production's own 85% train+val cutoff, then splits the
    remaining 15% into CAL-FIT (85-92.5%) and TEST (92.5-100%). Both are
    guaranteed outside what the saved model trained on."""
    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    q = lambda frac: df_sorted["timestamp"].iloc[int(n * frac)]
    val_end = q(val_end_frac)
    calfit_end = q(calfit_frac)
    calfit_df = df[(df["timestamp"] >= val_end) & (df["timestamp"] < calfit_end)]
    test_df = df[df["timestamp"] >= calfit_end]
    return calfit_df, test_df, val_end, calfit_end


def fit_calibrators(raw_cal_proba, y_cal):
    eps = 1e-6
    logit = np.log(np.clip(raw_cal_proba, eps, 1 - eps) / (1 - np.clip(raw_cal_proba, eps, 1 - eps)))
    platt = LogisticRegression()
    platt.fit(logit.reshape(-1, 1), y_cal)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal_proba, y_cal)
    return platt, iso


def apply_platt(platt, raw_proba):
    eps = 1e-6
    logit = np.log(np.clip(raw_proba, eps, 1 - eps) / (1 - np.clip(raw_proba, eps, 1 - eps)))
    return platt.predict_proba(logit.reshape(-1, 1))[:, 1]


def report_variant(name, y_test, proba):
    bs = brier_score(y_test, proba)
    cal_df = calibration_curve_manual(y_test, proba, n_bins=10)
    ece = expected_calibration_error(cal_df, len(y_test))
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")
    pr_auc = average_precision_score(y_test, proba)
    ll = log_loss(y_test, np.clip(proba, 1e-7, 1 - 1e-7))
    print(f"\n--- {name} ---")
    print(f"  Brier={bs:.6f}  ECE={ece:.6f}  LogLoss={ll:.6f}  AUC={auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"  min={proba.min():.6f}  max={proba.max():.6f}  mean={proba.mean():.6f}  "
          f"p50={np.percentile(proba,50):.6f}  p99={np.percentile(proba,99):.6f}")
    return {"variant": name, "brier": bs, "ece": ece, "log_loss": ll, "auc": auc, "pr_auc": pr_auc}


def decide_recommendation(raw_row, platt_row, iso_row):
    """The automated 'is calibration worth it' decision, applied
    per-horizon. Picks whichever calibrated variant improves Brier the
    most among those that don't hurt AUC beyond the tolerance; falls back
    to 'raw' (no calibration) if neither clears the bar."""
    candidates = []
    for name, row in [("platt", platt_row), ("isotonic", iso_row)]:
        if row is None:
            continue
        auc_drop = raw_row["auc"] - row["auc"] if pd.notna(raw_row["auc"]) and pd.notna(row["auc"]) else 999
        brier_improvement = raw_row["brier"] - row["brier"]
        if brier_improvement > 0 and auc_drop <= MAX_ACCEPTABLE_AUC_DROP:
            candidates.append((name, brier_improvement, row))
    if not candidates:
        return "raw", "Neither Platt nor Isotonic improved Brier score without hurting AUC beyond tolerance."
    candidates.sort(key=lambda c: c[1], reverse=True)
    best_name, improvement, best_row = candidates[0]
    return best_name, (f"Improved Brier by {improvement:.6f} "
                        f"({raw_row['brier']:.6f} -> {best_row['brier']:.6f}) "
                        f"with AUC change {best_row['auc'] - raw_row['auc']:+.4f}.")


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    model_dir = _resolve_model_dir()
    calfit_df, test_df, val_end, calfit_end = split_for_calibration(df)

    print(f"Model dir: {os.path.abspath(model_dir)}")
    print(f"CAL-FIT: {len(calfit_df):,} rows ({val_end} to {calfit_end}) -- fits Platt/Isotonic")
    print(f"TEST:    {len(test_df):,} rows (>= {calfit_end}) -- untouched until final evaluation\n")

    all_rows = []
    recommendation = {}

    for label, hours in PREDICTION_HORIZONS:
        target_col = f"outage_in_next_{label}"
        if target_col not in df.columns:
            continue
        model_path = os.path.join(model_dir, f"decision_model_{BACKEND}_{label}.joblib")
        if not os.path.exists(model_path):
            print(f"\n[{label}] No saved model at {model_path} -- skipping.")
            continue

        t0 = time.time()
        models = joblib.load(model_path)

        X_cal, y_cal = calfit_df[feature_columns], calfit_df[target_col].values
        X_test, y_test = test_df[feature_columns], test_df[target_col].values

        if y_cal.sum() < 5:
            print(f"\n{label}: too few positives in CAL-FIT ({int(y_cal.sum())}) to fit a reliable "
                  f"calibrator -- reporting raw only, recommending 'raw'.")
            raw_test = ensemble_proba(models, X_test)
            row = report_variant(f"{label} RAW (uncalibrated)", y_test, raw_test)
            row.update({"horizon": label, "hours": hours})
            all_rows.append(row)
            recommendation[label] = {"recommended": "raw",
                                      "reason": "Too few CAL-FIT positives to calibrate."}
            continue

        raw_cal = ensemble_proba(models, X_cal)
        raw_test = ensemble_proba(models, X_test)
        platt, iso = fit_calibrators(raw_cal, y_cal)
        platt_test = apply_platt(platt, raw_test)
        iso_test = iso.predict(raw_test)

        print(f"\n{'='*70}\nHORIZON {label} ({hours}h) -- loaded (not trained) in {time.time()-t0:.2f}s  "
              f"| test positives: {int(y_test.sum())} / {len(y_test):,}\n{'='*70}")

        raw_row = report_variant(f"{label} RAW (uncalibrated)", y_test, raw_test)
        platt_row = report_variant(f"{label} PLATT/sigmoid-calibrated", y_test, platt_test)
        iso_row = report_variant(f"{label} ISOTONIC-calibrated", y_test, iso_test)

        for row in (raw_row, platt_row, iso_row):
            row.update({"horizon": label, "hours": hours})
            all_rows.append(row)

        rec_choice, rec_reason = decide_recommendation(raw_row, platt_row, iso_row)
        print(f"\n  >> Recommendation for {label}: {rec_choice.upper()} -- {rec_reason}")
        recommendation[label] = {"recommended": rec_choice, "reason": rec_reason}

        if rec_choice != "raw":
            calibrator = platt if rec_choice == "platt" else iso
            out_path = os.path.join(model_dir, f"calibrator_{BACKEND}_{label}_{rec_choice}.joblib")
            joblib.dump(calibrator, out_path)
            print(f"  Saved calibrator: {out_path}")

    print(f"\n\n{'='*70}\nSUMMARY: raw vs Platt vs isotonic, all horizons\n{'='*70}")
    summary_df = pd.DataFrame(all_rows)[["horizon", "hours", "variant", "brier", "ece", "log_loss", "auc", "pr_auc"]]
    print(summary_df.to_string(index=False))
    summary_df.to_csv("calibration_experiment_summary.csv", index=False)
    print("\nSaved calibration_experiment_summary.csv")

    rec_path = os.path.join(model_dir, "calibration_recommendation.json")
    with open(rec_path, "w") as f:
        json.dump(recommendation, f, indent=2)
    print(f"Saved recommendation file: {rec_path}")

    print(f"\n{'='*70}\nFINAL RECOMMENDATION PER HORIZON\n{'='*70}")
    for label, rec in recommendation.items():
        print(f"  {label:>4}: {rec['recommended']:10s} -- {rec['reason']}")
    print("\ncalibrated_prediction.py will automatically read this file and apply "
          "calibration only where it's recommended above -- no manual step needed.")


if __name__ == "__main__":
    run()
