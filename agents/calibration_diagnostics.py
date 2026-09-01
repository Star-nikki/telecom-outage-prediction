"""
Priority 4: probability-calibration diagnostics.

READ-ONLY / DIAGNOSTIC ONLY. Does not modify decision_agent.py, does not
touch anything under models/, does not change any probability. It imports
and reuses the real add_trend_features / get_feature_columns / build_model
from decision_agent.py so results reflect the project's actual pipeline,
not a reimplementation.

WHERE TO SAVE:
    Put this file in the SAME folder as decision_agent.py (agents/), since
    it does `import decision_agent as da`.

HOW TO RUN (either works -- the script figures out the data path itself):
    From the project root:      python -m agents.calibration_diagnostics_final
    From inside agents/:        python calibration_diagnostics_final.py

WHAT IT DOES:
    For each horizon in PREDICTION_HORIZONS, trains one LightGBM model on
    the same time-ordered 70/15/15 split decision_agent.py's
    train_all_horizons() uses, then reports on the held-out test set:
      - raw, unrounded probability values (sample + full distribution)
      - Brier score
      - reliability/calibration table (10 bins: predicted vs actual)
      - ECE (expected calibration error)
      - positive/negative counts
    Also runs two split-safety checks: whether train/test share towers, and
    (for the 3h horizon only, to keep runtime reasonable) how performance
    looks on towers held out entirely from training.

If you have >~8GB RAM, MEMORY_SAFE_MODE=False below will run on the full
dataset directly (matches decision_agent.py's own approach exactly). If you
hit a MemoryError, flip it to True -- it computes trend features per-tower
(full history preserved, so trailing windows are unaffected) and then
subsamples, trading some statistical precision for a much smaller peak
memory footprint.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

# Make this runnable both as `python -m agents.calibration_diagnostics_final`
# (CWD = project root) and as `python calibration_diagnostics_final.py` run
# from inside agents/ (CWD = agents/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
from config import PREDICTION_HORIZONS

MEMORY_SAFE_MODE = True   # flip to True if you hit a MemoryError
SUBSAMPLE_FRAC = 0.25      # only used when MEMORY_SAFE_MODE=True

pd.set_option("display.width", 160)


def _resolve_data_path():
    """Tries the same relative path decision_agent.py itself uses, then a
    couple of common alternates, so this works regardless of whether you
    run it from the project root or from inside agents/."""
    candidates = [
        da.DATA_PATH,                                   # decision_agent.py's own path, as-is
        os.path.join("data", "telecom_timeseries.parquet"),
        os.path.join("..", "data", "telecom_timeseries.parquet"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find telecom_timeseries.parquet. Tried: {candidates}. "
        f"Edit _resolve_data_path() or pass the correct path directly."
    )


def load_features():
    data_path = _resolve_data_path()
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows, {df['tower_id'].nunique()} towers.\n")

    if not MEMORY_SAFE_MODE:
        print("Engineering trend features on the full dataset "
              "(same as decision_agent.py's own path)...")
        return da.add_trend_features(df)

    print(f"MEMORY_SAFE_MODE=True: computing trend features per tower "
          f"(full history), then subsampling to ~{SUBSAMPLE_FRAC*100:.0f}%...")
    rng = np.random.RandomState(42)
    pieces = []
    for tid in sorted(df["tower_id"].unique()):
        tdf = da.add_trend_features(df[df["tower_id"] == tid].copy())
        for c in tdf.select_dtypes(include=["float64"]).columns:
            tdf[c] = tdf[c].astype("float32")
        n_keep = int(len(tdf) * SUBSAMPLE_FRAC)
        idx = np.sort(rng.choice(len(tdf), size=n_keep, replace=False))
        pieces.append(tdf.iloc[idx])
        print(f"  {tid}: {len(tdf):,} -> kept {n_keep:,}")
    return pd.concat(pieces, ignore_index=True)


def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_curve_manual(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_range": f"[{bins[b]:.1f}, {bins[b+1]:.1f})",
            "n": int(mask.sum()),
            "mean_predicted": float(y_prob[mask].mean()),
            "actual_positive_rate": float(y_true[mask].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(cal_df, n_total):
    if len(cal_df) == 0:
        return float("nan")
    return float(sum(
        (row["n"] / n_total) * abs(row["mean_predicted"] - row["actual_positive_rate"])
        for _, row in cal_df.iterrows()
    ))


def time_split(df):
    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    train_end = df_sorted["timestamp"].iloc[int(n * 0.70)]
    val_end = df_sorted["timestamp"].iloc[int(n * 0.85)]
    train_df = df[df["timestamp"] < train_end]
    val_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)]
    test_df = df[df["timestamp"] >= val_end]
    return train_df, val_df, test_df, train_end, val_end


def train_one_horizon(train_df, val_df, test_df, feature_columns, target_col):
    X_train, y_train = train_df[feature_columns], train_df[target_col]
    X_val, y_val = val_df[feature_columns], val_df[target_col]
    X_test, y_test = test_df[feature_columns], test_df[target_col]

    if y_train.sum() == 0:
        return None

    pos_idx = y_train[y_train == 1].index
    neg_idx = y_train[y_train == 0].sample(
        n=min(len(pos_idx) * 10, (y_train == 0).sum()), random_state=42).index
    sel = pos_idx.union(neg_idx)
    Xb, yb = X_train.loc[sel], y_train.loc[sel]
    scale_pos_weight = (len(yb) - yb.sum()) / yb.sum() if yb.sum() > 0 else 1.0

    model = da.build_model("lightgbm", seed=42, scale_pos_weight=scale_pos_weight)
    model.fit(Xb, yb, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    test_proba = model.predict_proba(X_test)[:, 1]
    return model, test_proba, y_test.values


def run():
    df = load_features()
    feature_columns = da.get_feature_columns()
    train_df, val_df, test_df, train_end, val_end = time_split(df)

    print(f"Train: {len(train_df):,} rows (< {train_end})")
    print(f"Val:   {len(val_df):,} rows ({train_end} to {val_end})")
    print(f"Test:  {len(test_df):,} rows (>= {val_end})\n")

    train_towers = set(train_df["tower_id"].unique())
    test_towers = set(test_df["tower_id"].unique())
    print(f"Towers in train: {len(train_towers)} | Towers in test: {len(test_towers)} | "
          f"Overlap: {len(train_towers & test_towers)} / {len(train_towers | test_towers)}")
    if train_towers == test_towers:
        print("  -> every tower appears in both train and test. This is a time-based split,\n"
              "     safe against future-row leakage, but it does NOT test generalization to a\n"
              "     tower the model has never seen at all. See the held-out-tower check below.\n")

    summary_rows = []
    for label, hours in PREDICTION_HORIZONS:
        target_col = f"outage_in_next_{label}"
        if target_col not in df.columns:
            print(f"Skipping {label}: no {target_col} column found")
            continue

        t0 = time.time()
        result = train_one_horizon(train_df, val_df, test_df, feature_columns, target_col)
        if result is None:
            print(f"Skipping {label}: no positives in training split")
            continue
        model, test_proba, y_test = result
        print(f"\n{'='*70}\nHORIZON {label} ({hours}h) -- trained in {time.time()-t0:.1f}s\n{'='*70}")
        print(f"Test set: {len(y_test):,} rows | positives: {int(y_test.sum())} | "
              f"positive_rate: {y_test.mean()*100:.4f}%")

        print(f"\nRaw probability distribution (unrounded):")
        print(f"  min={test_proba.min():.10f}  max={test_proba.max():.10f}  mean={test_proba.mean():.10f}")
        for p in [1, 5, 25, 50, 75, 90, 95, 99, 99.9]:
            print(f"  p{p:>5}: {np.percentile(test_proba, p):.10f}")
        n_sat_high = (test_proba > 0.999).sum()
        print(f"  # predictions > 0.999: {n_sat_high} ({n_sat_high/len(test_proba)*100:.2f}% of test set)")
        print(f"\nFirst 10 raw probabilities (unrounded):")
        print("  " + ", ".join(f"{v:.8f}" for v in test_proba[:10]))

        bs = brier_score(y_test, test_proba)
        cal_df = calibration_curve_manual(y_test, test_proba, n_bins=10)
        ece = expected_calibration_error(cal_df, len(y_test))
        print(f"\nBrier score: {bs:.6f}   ECE (10 bins): {ece:.6f}")
        print("\nReliability table (predicted vs actual, by bin):")
        print(cal_df.to_string(index=False))

        summary_rows.append({
            "horizon": label, "hours": hours, "n_test": len(y_test),
            "n_pos_test": int(y_test.sum()), "positive_rate_test": y_test.mean(),
            "brier": bs, "ece": ece,
            "prob_min": test_proba.min(), "prob_max": test_proba.max(),
            "prob_mean": test_proba.mean(), "prob_p50": np.percentile(test_proba, 50),
            "prob_p99": np.percentile(test_proba, 99),
            "pct_over_0999": n_sat_high / len(test_proba) * 100,
        })

    print(f"\n\n{'='*70}\nSUMMARY ACROSS ALL HORIZONS\n{'='*70}")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("calibration_summary.csv", index=False)
    print("\nSaved calibration_summary.csv")

    # --- held-out-tower check (3h only, to keep runtime reasonable) ---
    print(f"\n\n{'='*70}\nHELD-OUT-TOWER CHECK (3h horizon)\n{'='*70}")
    towers = sorted(df["tower_id"].unique())
    rng = np.random.RandomState(7)
    n_holdout = max(1, len(towers) // 5)
    held_out = set(rng.choice(towers, size=n_holdout, replace=False))
    train_towers2 = [t for t in towers if t not in held_out]
    tr = df[df["tower_id"].isin(train_towers2)]
    te = df[df["tower_id"].isin(held_out)]
    print(f"Train towers: {len(train_towers2)} | Held-out (never seen) towers: {len(held_out)}")

    result2 = train_one_horizon(tr, tr, te, feature_columns, "outage_in_next_3h")
    if result2:
        _, proba2, y2 = result2
        auc2 = roc_auc_score(y2, proba2)
        brier2 = brier_score_loss(y2, proba2)
        n_over2 = (proba2 > 0.999).sum()
        print(f"AUC={auc2:.4f}  Brier={brier2:.6f}  max_proba={proba2.max():.6f}  "
              f"# >0.999: {n_over2} ({n_over2/len(proba2)*100:.2f}%)")
        print("If this looks similar to the time-based-split 3h result above, the saturation\n"
              "is not being driven by the model having seen these specific towers before.")


if __name__ == "__main__":
    run()
