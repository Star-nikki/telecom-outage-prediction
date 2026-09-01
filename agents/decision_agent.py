"""
Decision Agent -- v3: LightGBM / XGBoost instead of Random Forest.

WHY THE SWITCH: gradient boosting (LightGBM, XGBoost) generally handles
imbalanced tabular data like this better than Random Forest, and trains
faster. This version lets you run either one, or both side-by-side so you
can directly compare which fits your data better -- that comparison is
itself worth reporting.

Everything from the previous version is preserved:
  - Trend features (rate-of-change over 1h/6h/24h per tower) so gradual
    ramps like capacity_creep are visible to the model, not just snapshots.
  - Proper 3-way time-based split (train / validation / test) with the
    decision threshold tuned on validation (maximizing F1), then reported
    honestly on a held-out test set.
  - #6 Multi-horizon: separate model per horizon (15m...24h), plus
    "lift over baseline" (AUC-PR / positive_rate) for fair cross-horizon
    comparison, since the positive rate itself changes ~100x across horizons.
  - #5 Confidence: now backend-agnostic. Trains a small BAGGED ENSEMBLE
    (default 3 models with different random seeds + bootstrap resampling)
    per horizon. Probability = ensemble mean. Confidence = how much the
    ensemble members agree (low spread = high confidence). Set --bagging 1
    to skip this and use a simple margin-based confidence instead (faster,
    weaker signal).
  - #4 Reasoning: unchanged -- structured evidence across the four
    conceptual agents, fed to an LLM (or the offline rule-based fallback).

WHERE TO SAVE THIS FILE:
    Overwrite the existing file at:
    <your project folder>\\telecom_sim_v2\\agents\\decision_agent.py

USAGE (run from the agents/ folder, after simulate.py has produced data):

    pip install lightgbm xgboost scikit-learn joblib --break-system-packages

    python decision_agent.py --mode train --model both
    python decision_agent.py --mode train --model lightgbm
    python decision_agent.py --mode train --model xgboost
    python decision_agent.py --mode explain --tower T0000 --model lightgbm

--model choices: lightgbm, xgboost, both, random_forest (random_forest kept
available if you want a classical-ML baseline to compare against in your
report, but it's no longer the default).

--bagging N (default 3): how many bootstrap-resampled models per horizon
per backend, used only for the confidence estimate. Set to 1 for fastest
training (single model per horizon, margin-based confidence instead of
ensemble-agreement confidence).
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PREDICTION_HORIZONS  # noqa: E402
from llm_providers import get_provider  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "../data/telecom_timeseries.parquet"
MODEL_DIR = "models"

BASE_FEATURE_COLUMNS = [
    "grid_voltage", "battery_level_pct", "rsrp_dbm",                       # Tower Health Agent's domain
    "utilization_pct", "throughput_mbps", "latency_ms", "packet_loss_pct",  # Traffic Agent's domain
    "storm_flag", "rainfall_mm", "wind_kmh",                                # Weather Agent's domain
    "load_multiplier",                                                      # User Behavior Agent's domain
]

# Trend columns added on top of the instantaneous ones above.
# (column, lookback_hours) -- lookback converted to 15-min steps internally.
TREND_SPECS = [
    ("battery_level_pct", 1), ("battery_level_pct", 6),
    ("grid_voltage", 1), ("grid_voltage", 6),
    ("rsrp_dbm", 6),
    ("utilization_pct", 1), ("utilization_pct", 6), ("utilization_pct", 24),
    ("packet_loss_pct", 1), ("packet_loss_pct", 6),
    ("throughput_mbps", 6),
    ("latency_ms", 1),
]


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rate-of-change features per tower, e.g. 'battery_level_pct_delta_6h'
    = current value minus the value 6 hours ago for that same tower.
    Captures RAMPS, not just instantaneous readings -- this is what exposes
    slow failures like capacity_creep, which look normal in a single
    snapshot but clearly trending in a window.
    """
    print("Engineering trend features (touches every row once, may take a bit)...")
    df = df.sort_values(["tower_id", "timestamp"]).reset_index(drop=True)
    steps_per_hour = 4  # 15-min intervals

    new_cols = {}
    grouped = df.groupby("tower_id", sort=False)
    for col, lookback_h in TREND_SPECS:
        periods = lookback_h * steps_per_hour
        colname = f"{col}_delta_{lookback_h}h"
        new_cols[colname] = grouped[col].diff(periods=periods).fillna(0)

    # Rolling window statistics -- much stronger signal than a single delta,
    # since they capture the shape of recent behavior (spread, extremes),
    # not just the net change between two points.
    unique_cols = sorted(set(col for col, _ in TREND_SPECS))
    for col in unique_cols:
        rolling = grouped[col].rolling(window=24, min_periods=1)

        new_cols[f"{col}_mean_6h"] = (
            rolling.mean()
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_std_6h"] = (
            rolling.std()
            .fillna(0)
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_max_6h"] = (
            rolling.max()
            .reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_min_6h"] = (
            rolling.min()
            .reset_index(level=0, drop=True)
        )

    trend_df = pd.DataFrame(new_cols)
    df = pd.concat([df, trend_df], axis=1)
    return df


def get_feature_columns():
    trend_cols = [f"{col}_delta_{h}h" for col, h in TREND_SPECS]
    unique_cols = sorted(set(col for col, _ in TREND_SPECS))
    rolling_cols = []
    for col in unique_cols:
        rolling_cols += [f"{col}_mean_6h", f"{col}_std_6h", f"{col}_max_6h", f"{col}_min_6h"]
    return BASE_FEATURE_COLUMNS + trend_cols + rolling_cols


# ---------------------------------------------------------------------------
# Model backends -- this is the swappable part
# ---------------------------------------------------------------------------
def build_model(backend: str, seed: int, scale_pos_weight: float):
    """Returns a fresh, unfit classifier for the requested backend."""
    if backend == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
    objective="binary",
    metric="average_precision",

    n_estimators=800,
    learning_rate=0.03,

    num_leaves=63,
    max_depth=-1,

    subsample=0.8,
    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    random_state=seed,
    n_jobs=-1,
    verbosity=-1
	)
    elif backend == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
    elif backend == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: lightgbm, xgboost, random_forest")


def train_bagged_ensemble(backend: str, X_train, y_train, X_val, y_val, n_bagging: int):
    """
    Trains n_bagging models on bootstrap-resampled versions of the training
    set, each with a different random seed. Used for BOTH a (slightly more
    robust, averaged) probability estimate AND the confidence score --
    members that agree with each other -> high confidence.

    n_bagging=1 skips resampling entirely (trains once, on the full set) --
    fastest option, but confidence then falls back to a simpler
    margin-based estimate (see predict_with_confidence).

    Training set is first rebalanced: ALL positives are kept, plus a capped
    sample of negatives (at most 10x the positive count). With outages this
    rare, training on the raw ~100:1 (or worse) imbalance buries the signal;
    this alone often improves precision dramatically. Bagging then resamples
    on top of this already-balanced set.
    """
    positive_idx = y_train[y_train == 1].index
    negative_idx = y_train[y_train == 0].sample(
        n=min(len(positive_idx) * 10, (y_train == 0).sum()),
        random_state=42,
    ).index
    selected = positive_idx.union(negative_idx)

    X_train_bal = X_train.loc[selected]
    y_train_bal = y_train.loc[selected]

    pos_count = y_train_bal.sum()
    neg_count = len(y_train_bal) - pos_count
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    models = []
    for i in range(n_bagging):
        seed = 42 + i
        if n_bagging == 1:
            Xs, ys = X_train_bal, y_train_bal
        else:
            rng = np.random.RandomState(seed)
            sample_idx = rng.choice(len(X_train_bal), size=len(X_train_bal), replace=True)
            Xs, ys = X_train_bal.iloc[sample_idx], y_train_bal.iloc[sample_idx]

        model = build_model(backend, seed, scale_pos_weight)

        if backend == "lightgbm":
            import lightgbm as lgb
            model.fit(
                Xs,
                ys,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0),
                ],
            )
        else:
            model.fit(Xs, ys)

        models.append(model)
    return models


def ensemble_predict_proba(models: list, X) -> np.ndarray:
    """Average probability across all bagged ensemble members."""
    preds = np.array([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# Multi-horizon training + evaluation (per backend)
# ---------------------------------------------------------------------------
def train_all_horizons(df: pd.DataFrame, backends: list, n_bagging: int = 3):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = add_trend_features(df)
    feature_columns = get_feature_columns()

    # 3-way time-based split: train (fit) / validation (tune threshold) /
    # test (final honest report). Time-ordered, never random -- this is a
    # forecasting task, and random splits would leak future rows into training.
    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    train_end = df_sorted["timestamp"].iloc[int(n * 0.70)]
    val_end = df_sorted["timestamp"].iloc[int(n * 0.85)]

    train_df = df[df["timestamp"] < train_end]
    val_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)]
    test_df = df[df["timestamp"] >= val_end]
    print(f"Train: {len(train_df):,} rows (< {train_end})")
    print(f"Val:   {len(val_df):,} rows ({train_end} to {val_end}) -- used only to pick the threshold")
    print(f"Test:  {len(test_df):,} rows (>= {val_end}) -- final numbers reported on this, untouched until now")
    print(f"Bagging: {n_bagging} model(s) per horizon per backend\n")

    results = []
    for backend in backends:
        print(f"\n{'='*60}\nBACKEND: {backend}\n{'='*60}")
        for label, hours in PREDICTION_HORIZONS:
            target_col = f"outage_in_next_{label}"
            if target_col not in df.columns:
                print(f"Skipping {label}: column {target_col} not found "
                      f"(regenerate data with the updated simulate.py)")
                continue

            X_train, y_train = train_df[feature_columns], train_df[target_col]
            X_val, y_val = val_df[feature_columns], val_df[target_col]
            X_test, y_test = test_df[feature_columns], test_df[target_col]

            if y_train.sum() == 0:
                print(f"Skipping {label}: no positive examples in training split.")
                continue

            print()
            print("Training positives :", y_train.sum())
            print("Training negatives :", len(y_train) - y_train.sum())
            print("Validation positives :", y_val.sum())
            print("Test positives :", y_test.sum())
            print()

            models = train_bagged_ensemble(backend, X_train, y_train, X_val, y_val, n_bagging)
            joblib.dump(models, os.path.join(MODEL_DIR, f"decision_model_{backend}_{label}.joblib"))

            importance = pd.Series(
                models[0].feature_importances_,
                index=feature_columns
            ).sort_values(ascending=False)

            print()
            print(importance.head(20))
            print()

            # ---- pick the threshold that maximizes F1 on VALIDATION ----
            val_proba = ensemble_predict_proba(models, X_val)

            print()
            print("Probability summary")
            print(val_proba.min())
            print(np.percentile(val_proba, 50))
            print(np.percentile(val_proba, 90))
            print(np.percentile(val_proba, 95))
            print(np.percentile(val_proba, 99))
            print(val_proba.max())
            print()

            if y_val.sum() > 0:
                prec_v, rec_v, thresholds_v = precision_recall_curve(y_val, val_proba)
                if len(thresholds_v) > 0:
                    f1_v = np.where((prec_v + rec_v) > 0, 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12), 0)
                    mask = prec_v[:-1] >= 0.5
                    if mask.any():
                        idx = np.argmax(rec_v[:-1][mask])
                        best_threshold = thresholds_v[mask][idx]
                    else:
                        best_threshold = thresholds_v[np.argmax(f1_v[:-1])]
                else:
                    best_threshold = 0.5
            else:
                best_threshold = 0.5  # no positives in validation, can't tune -- fall back

            # ---- report FINAL numbers on TEST, at the tuned threshold ----
            test_proba = ensemble_predict_proba(models, X_test)
            test_preds = (test_proba >= best_threshold).astype(int)

            precision_vals, recall_vals, _ = precision_recall_curve(y_test, test_proba)
            auc_pr = auc(recall_vals, precision_vals)
            precision = precision_score(y_test, test_preds, zero_division=0)
            recall = recall_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            positive_rate = y_test.mean()
            lift = auc_pr / positive_rate if positive_rate > 0 else float("nan")

            results.append({
                "backend": backend, "horizon": label, "hours": hours,
                "auc_pr": round(auc_pr, 4),
                "tuned_threshold": round(float(best_threshold), 4),
                "precision_tuned": round(precision, 4),
                "recall_tuned": round(recall, 4),
                "f1_tuned": round(f1, 4),
                "positive_rate_test": round(positive_rate, 6),
                "lift_over_baseline": round(lift, 1),
            })
            print(f"[{backend:>13} | {label:>4}] AUC-PR={auc_pr:.4f}  threshold={best_threshold:.3f}  "
                  f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}  "
                  f"lift={lift:.0f}x  positive_rate={positive_rate*100:.4f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, "horizon_comparison.csv"), index=False)
    print(f"\nSaved comparison table to {MODEL_DIR}/horizon_comparison.csv")

    if len(backends) > 1:
        print("\n--- Side-by-side backend comparison (lift_over_baseline, higher = better) ---")
        pivot = results_df.pivot(index="horizon", columns="backend", values="lift_over_baseline")
        # reorder rows by the natural horizon order, not alphabetically
        horizon_order = [label for label, _ in PREDICTION_HORIZONS]
        pivot = pivot.reindex([h for h in horizon_order if h in pivot.index])
        print(pivot.to_string())

    print("\nCompare 'lift_over_baseline' across horizons (fairer than raw AUC-PR,")
    print("since positive_rate itself changes ~100x between the shortest and longest horizon).")
    return results_df


# ---------------------------------------------------------------------------
# Confidence estimation (backend-agnostic, works for the bagged ensemble)
# ---------------------------------------------------------------------------
def predict_with_confidence(models: list, row: pd.Series):
    """
    Probability = mean across the bagged ensemble members.
    Confidence   = how much the ensemble members AGREE with each other
    (low spread = high confidence), UNLESS n_bagging was 1, in which case
    we fall back to a simple margin-based confidence (distance from 0.5).
    """
    X = row[get_feature_columns()].values.reshape(1, -1)
    member_preds = np.array([m.predict_proba(X)[0, 1] for m in models])
    probability = float(member_preds.mean())

    if len(models) > 1:
        agreement = 1 - (member_preds.std() * 2)  # std maxes out around 0.5 for probabilities
        confidence = float(np.clip(agreement, 0, 1))
    else:
        confidence = float(np.clip(abs(probability - 0.5) * 2, 0, 1))

    return probability, confidence


# ---------------------------------------------------------------------------
# Reasoning over conflicting evidence (unchanged from before)
# ---------------------------------------------------------------------------
def build_evidence(row: pd.Series) -> dict:
    """Translate raw numbers into the kind of qualitative statements each
    conceptual monitoring agent would actually report. Includes trend deltas
    where available, since a ramp is often more informative than a snapshot."""
    evidence = {}

    def trend_note(col, hours):
        key = f"{col}_delta_{hours}h"
        if key in row.index and pd.notna(row[key]) and abs(row[key]) > 0.01:
            direction = "down" if row[key] < 0 else "up"
            return f" (trending {direction} {abs(row[key]):.1f} over last {hours}h)"
        return ""

    if row["battery_level_pct"] < 50 or row["grid_voltage"] < 210:
        evidence["tower_health"] = (
            f"Battery at {row['battery_level_pct']:.0f}%{trend_note('battery_level_pct', 6)}, "
            f"grid voltage {row['grid_voltage']:.0f}V{trend_note('grid_voltage', 6)} -- power supply degrading."
        )
    else:
        evidence["tower_health"] = f"Battery {row['battery_level_pct']:.0f}%, voltage {row['grid_voltage']:.0f}V -- normal."

    if row["rsrp_dbm"] < -95:
        evidence["signal"] = f"RSRP {row['rsrp_dbm']:.0f} dBm{trend_note('rsrp_dbm', 6)} -- rapid signal attenuation."
    else:
        evidence["signal"] = f"RSRP {row['rsrp_dbm']:.0f} dBm -- signal stable."

    if row["utilization_pct"] > 80 or row["packet_loss_pct"] > 5:
        evidence["traffic"] = (
            f"Utilization {row['utilization_pct']:.0f}%{trend_note('utilization_pct', 24)}, "
            f"packet loss {row['packet_loss_pct']:.1f}% -- congested."
        )
    else:
        evidence["traffic"] = f"Utilization {row['utilization_pct']:.0f}%, packet loss {row['packet_loss_pct']:.1f}% -- normal."

    if row["storm_flag"] == 1 or row["rainfall_mm"] > 5:
        evidence["weather"] = f"Storm active, rainfall {row['rainfall_mm']:.1f}mm -- storm probability high."
    else:
        evidence["weather"] = "No storm activity -- clear conditions."

    if row["load_multiplier"] > 1.5:
        evidence["user_behavior"] = f"Load multiplier {row['load_multiplier']:.2f} -- unusual demand spike."
    else:
        evidence["user_behavior"] = f"Load multiplier {row['load_multiplier']:.2f} -- normal demand pattern."

    return evidence


def reason_about_tower(row: pd.Series, horizon_predictions: dict, provider_name: str = "rule_based") -> str:
    evidence = build_evidence(row)
    provider = get_provider(provider_name)

    evidence_lines = "\n".join(f"- {v}" for v in evidence.values())
    prediction_lines = "\n".join(
        f"- {h}: {p['probability']*100:.0f}% probability (confidence {p['confidence']*100:.0f}%)"
        for h, p in horizon_predictions.items()
    )

    prompt = f"""You are analyzing tower {row['tower_id']} at {row['timestamp']}.

Evidence from monitoring agents:
{evidence_lines}

Model predictions across horizons:
{prediction_lines}

In 2-3 sentences, explain which failure mode (hardware/power, congestion,
weather-induced, or slow capacity overload) is most likely given this specific
combination of evidence, and why -- reference which signals align and which
don't, the way a network engineer would reason through it."""

    return provider.generate(prompt)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "explain"], default="train")
    parser.add_argument("--model", default="both",
                         choices=["lightgbm", "xgboost", "both", "random_forest"],
                         help="Which backend(s) to use. 'both' = lightgbm + xgboost side by side.")
    parser.add_argument("--bagging", type=int, default=3,
                         help="Models per horizon per backend, for confidence estimation. "
                              "Use 1 for fastest training (margin-based confidence instead).")
    parser.add_argument("--tower", default=None, help="tower_id for --mode explain")
    parser.add_argument("--provider", default="rule_based",
                         choices=["rule_based", "anthropic", "openai", "ollama"])
    args = parser.parse_args()

    backends = ["lightgbm", "xgboost"] if args.model == "both" else [args.model]

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows.\n")

    if args.mode == "train":
        train_all_horizons(df, backends, n_bagging=args.bagging)

    elif args.mode == "explain":
        if len(backends) > 1:
            print(f"--mode explain needs exactly one backend; using '{backends[0]}' "
                  f"(pass --model lightgbm or --model xgboost to choose explicitly).")
        backend = backends[0]

        tower_id = args.tower or df["tower_id"].iloc[0]
        tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
        tdf = add_trend_features(tdf)  # same trend features the models were trained on

        # Prefer a row that's actually degraded, so the demo is interesting;
        # fall back to the latest row if this tower never degrades.
        degraded_rows = tdf[tdf["is_degraded"] == 1]
        row = degraded_rows.iloc[len(degraded_rows) // 2] if len(degraded_rows) else tdf.iloc[-1]

        horizon_predictions = {}
        for label, _ in PREDICTION_HORIZONS:
            model_path = os.path.join(MODEL_DIR, f"decision_model_{backend}_{label}.joblib")
            if not os.path.exists(model_path):
                print(f"No trained model for {backend}/{label} yet -- run --mode train --model {backend} first.")
                continue
            models = joblib.load(model_path)
            prob, conf = predict_with_confidence(models, row)
            horizon_predictions[label] = {"probability": prob, "confidence": conf}

        print(f"Backend: {backend}")
        print(f"Tower: {tower_id}  Timestamp: {row['timestamp']}  True root cause: {row['root_cause']}\n")
        print("Predictions by horizon:")
        for h, p in horizon_predictions.items():
            print(f"  {h:>4}: probability={p['probability']*100:5.1f}%  confidence={p['confidence']*100:5.1f}%")

        print("\nEvidence:")
        for line in build_evidence(row).values():
            print(f"  - {line}")

        print("\nReasoning:")
        print(reason_about_tower(row, horizon_predictions, provider_name=args.provider))


if __name__ == "__main__":
    main()