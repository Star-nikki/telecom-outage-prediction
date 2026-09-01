"""
Priority 4B: calibration experiment.

READ-ONLY / DIAGNOSTIC ONLY -- same guarantee as calibration_diagnostics.py.
Does not modify decision_agent.py, does not touch models/, does not change
any production probability. Imports decision_agent (via calibration_diagnostics.py)
only to reuse the real feature/model-building code.

WHERE TO SAVE:
    Same folder as calibration_diagnostics.py and decision_agent.py (agents/).
    This file imports from calibration_diagnostics.py, so that file must
    already be sitting next to it -- you already have it there.

HOW TO RUN:
    From the project root:      python -m agents.calibration_experiment
    From inside agents/:        python calibration_experiment.py

WHAT IT DOES
    For each horizon, uses a 4-way TIME-ORDERED split (nothing later ever
    used to inform anything earlier):

        [ TRAIN-FIT ] [ EARLY-STOP ] [ CAL-FIT ] [ THRESH-SELECT ] [ TEST ]
             |               |            |              |           |
        model.fit()    eval_set for   fit Platt &    pick an        FINAL,
                        early stop    Isotonic       operating      touched
                                      calibrators    threshold      exactly
                                                                     once

    Then, on the untouched TEST slice only, reports Brier / ECE / log loss /
    AUC / PR-AUC / a reliability table for THREE versions of the same raw
    model output: raw (uncalibrated), Platt/sigmoid-calibrated, and
    isotonic-calibrated -- so you can see directly whether calibration
    actually fixes the overconfidence found in Priority 4, and by how much,
    for every horizon including 15m.

    NOTE on 15m specifically: the Priority 4 run showed the 15m model
    early-stopping at iteration 5 with ~99% of predictions collapsing to
    one value -- a sign of a genuinely weak/undertrained model, not just a
    miscalibrated one. This script still runs the same experiment on 15m,
    but if AUC/PR-AUC stay low even after calibration, that confirms the
    problem there is discrimination (the model isn't ranking well), which
    calibration cannot fix -- it can only reshape a probability scale that
    is already ranking correctly.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
from config import PREDICTION_HORIZONS
from calibration_diagnostics import (
    load_features, brier_score, calibration_curve_manual,
    expected_calibration_error,
)

pd.set_option("display.width", 160)


def four_way_split(df):
    """Same train/val/test time cutoffs as calibration_diagnostics.py
    (70/15/15), with the training portion further split off a small
    early-stopping tail, and the val portion split into a calibration-fit
    half and a threshold-selection half. Every boundary is a timestamp
    cutoff, so nothing in a later stage can influence an earlier one."""
    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    q = lambda frac: df_sorted["timestamp"].iloc[int(n * frac)]

    train_fit_end = q(0.65)
    earlystop_end = q(0.70)
    calfit_end = q(0.775)
    thresh_end = q(0.85)

    train_fit_df = df[df["timestamp"] < train_fit_end]
    earlystop_df = df[(df["timestamp"] >= train_fit_end) & (df["timestamp"] < earlystop_end)]
    calfit_df = df[(df["timestamp"] >= earlystop_end) & (df["timestamp"] < calfit_end)]
    thresh_df = df[(df["timestamp"] >= calfit_end) & (df["timestamp"] < thresh_end)]
    test_df = df[df["timestamp"] >= thresh_end]
    return train_fit_df, earlystop_df, calfit_df, thresh_df, test_df


def fit_raw_model(train_fit_df, earlystop_df, feature_columns, target_col):
    X_train, y_train = train_fit_df[feature_columns], train_fit_df[target_col]
    X_es, y_es = earlystop_df[feature_columns], earlystop_df[target_col]
    if y_train.sum() == 0:
        return None

    pos_idx = y_train[y_train == 1].index
    neg_idx = y_train[y_train == 0].sample(
        n=min(len(pos_idx) * 10, (y_train == 0).sum()), random_state=42).index
    sel = pos_idx.union(neg_idx)
    Xb, yb = X_train.loc[sel], y_train.loc[sel]
    scale_pos_weight = (len(yb) - yb.sum()) / yb.sum() if yb.sum() > 0 else 1.0

    model = da.build_model("lightgbm", seed=42, scale_pos_weight=scale_pos_weight)
    model.fit(Xb, yb, eval_set=[(X_es, y_es)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    return model


def fit_calibrators(raw_cal_proba, y_cal):
    """Platt/sigmoid: logistic regression on the logit of the raw
    probability (the standard Platt-scaling formulation). Isotonic: a
    free-form monotonic map, better suited to the non-sigmoid-shaped
    overconfidence seen at 24h (big flat over-prediction in the middle of
    the range, not a simple uniform shift)."""
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
    print("  Reliability (predicted vs actual, by bin):")
    print("  " + cal_df.to_string(index=False).replace("\n", "\n  "))
    return {"variant": name, "brier": bs, "ece": ece, "log_loss": ll, "auc": auc, "pr_auc": pr_auc}


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    train_fit_df, earlystop_df, calfit_df, thresh_df, test_df = four_way_split(df)

    print(f"TRAIN-FIT:   {len(train_fit_df):,} rows")
    print(f"EARLY-STOP:  {len(earlystop_df):,} rows")
    print(f"CAL-FIT:     {len(calfit_df):,} rows")
    print(f"THRESH-SEL:  {len(thresh_df):,} rows")
    print(f"TEST:        {len(test_df):,} rows  (untouched until final evaluation)\n")

    all_rows = []
    for label, hours in PREDICTION_HORIZONS:
        target_col = f"outage_in_next_{label}"
        if target_col not in df.columns:
            continue

        t0 = time.time()
        model = fit_raw_model(train_fit_df, earlystop_df, feature_columns, target_col)
        if model is None:
            print(f"Skipping {label}: no positives in training split")
            continue

        raw_cal = model.predict_proba(calfit_df[feature_columns])[:, 1]
        y_cal = calfit_df[target_col].values
        raw_test = model.predict_proba(test_df[feature_columns])[:, 1]
        y_test = test_df[target_col].values

        if y_cal.sum() < 5:
            print(f"\n{label}: too few positives in CAL-FIT slice ({int(y_cal.sum())}) "
                  f"to fit a reliable calibrator -- skipping calibration for this horizon, "
                  f"reporting raw only.")
            row = report_variant(f"{label} RAW (uncalibrated)", y_test, raw_test)
            row.update({"horizon": label, "hours": hours})
            all_rows.append(row)
            continue

        platt, iso = fit_calibrators(raw_cal, y_cal)
        platt_test = apply_platt(platt, raw_test)
        iso_test = iso.predict(raw_test)

        print(f"\n{'='*70}\nHORIZON {label} ({hours}h) -- trained in {time.time()-t0:.1f}s  "
              f"| test positives: {int(y_test.sum())} / {len(y_test):,}\n{'='*70}")
        for name, proba in (("RAW (uncalibrated)", raw_test),
                             ("PLATT/sigmoid-calibrated", platt_test),
                             ("ISOTONIC-calibrated", iso_test)):
            row = report_variant(f"{label} {name}", y_test, proba)
            row.update({"horizon": label, "hours": hours})
            all_rows.append(row)

    print(f"\n\n{'='*70}\nSUMMARY: raw vs Platt vs isotonic, all horizons\n{'='*70}")
    summary_df = pd.DataFrame(all_rows)[["horizon", "hours", "variant", "brier", "ece", "log_loss", "auc", "pr_auc"]]
    print(summary_df.to_string(index=False))
    summary_df.to_csv("calibration_experiment_summary.csv", index=False)
    print("\nSaved calibration_experiment_summary.csv")


if __name__ == "__main__":
    run()
