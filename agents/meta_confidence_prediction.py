"""
Priority 5 production integration -- opt-in meta-confidence wrapper.

Wraps predict_with_confidence_calibrated() (Priority 4) with an ADDITIONAL,
optional meta-confidence score for horizons where meta_confidence_diagnostics.py
found the alert-only stacked model genuinely beats existing ensemble-spread
confidence -- as of the last diagnostic run, that's 3h only. decision_agent.py
and calibrated_prediction.py are NOT modified; this is a separate layer on
top, so nothing changes unless you explicitly call this wrapper instead.

WHAT IT RETURNS: (probability, confidence, calibration_method, meta_confidence)
    meta_confidence is None whenever:
      - meta_confidence_recommendation.json doesn't exist yet (script not run)
      - this horizon wasn't recommended (use_meta: false)
      - the row's own probability doesn't cross this horizon's decision
        threshold -- the meta-model was trained ONLY on alerts (rows that
        cross threshold), so applying it outside that population is outside
        what it was validated for. Returning None here rather than a
        number computed out-of-distribution is deliberate.
    When meta_confidence IS returned, it's a genuine "how trustworthy is
    this specific alert" score (0-1, higher = more likely to be a correct
    alert), fit and evaluated specifically on alerts, not the general
    population.

WHY THE FEATURES ARE RECOMPUTED HERE, EXPENSIVE AS LEAF SUPPORT IS: the
saved leaf lookup tables (meta_confidence_leaftables_*.joblib) make this
tractable -- they're built once by meta_confidence_diagnostics.py from a
training reference sample and reused here via lookup, not rebuilt per
prediction. Still slower than plain calibrated_prediction.py because of the
per-tree leaf lookups; only computed when the row's own probability crosses
threshold (i.e., only for actual alerts), which is a small fraction of
predictions in practice.

WHERE TO SAVE:
    Same folder as decision_agent.py, calibrated_prediction.py,
    confidence_diagnostics.py, leaf_support_diagnostics.py (agents/).

HOW TO USE:
    import decision_agent as da
    from meta_confidence_prediction import predict_with_meta_confidence
    import joblib

    models = joblib.load("models/decision_model_lightgbm_3h.joblib")
    prob, conf, method, meta_conf = predict_with_meta_confidence(models, row, "lightgbm", "3h")
    # meta_conf is None unless this horizon has a validated model AND this
    # row's probability crossed its decision threshold.

TO WIRE INTO orchestrator_v3.py's decision_agent_node() (a deliberate,
separate step -- not done automatically here):

    # after:
    prob, conf, method = predict_with_confidence_calibrated(models, row, backend, label, model_dir=MODEL_DIR)
    # add:
    _, _, _, meta_conf = predict_with_meta_confidence(models, row, backend, label, model_dir=MODEL_DIR)
    predictions[label]["meta_confidence"] = meta_conf
"""
import os
import json

import numpy as np
import pandas as pd
import joblib

import decision_agent as da
from confidence_diagnostics import compute_row_level_stats
from leaf_support_diagnostics import score_leaf_support
from calibrated_prediction import predict_with_confidence_calibrated, _resolve_model_dir

_recommendation_cache = None
_artifact_cache = {}


def _load_recommendation(model_dir):
    global _recommendation_cache
    if _recommendation_cache is None:
        path = os.path.join(model_dir, "meta_confidence_recommendation.json")
        if os.path.exists(path):
            with open(path) as f:
                _recommendation_cache = json.load(f)
        else:
            _recommendation_cache = {}
    return _recommendation_cache


def _load_artifacts(model_dir, backend, label):
    key = (model_dir, backend, label)
    if key not in _artifact_cache:
        model_path = os.path.join(model_dir, f"meta_confidence_model_{backend}_{label}.joblib")
        scaler_path = os.path.join(model_dir, f"meta_confidence_scaler_{backend}_{label}.joblib")
        leaf_path = os.path.join(model_dir, f"meta_confidence_leaftables_{backend}_{label}.joblib")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(leaf_path)):
            _artifact_cache[key] = None
        else:
            _artifact_cache[key] = {
                "model": joblib.load(model_path),
                "scaler": joblib.load(scaler_path),
                "leaf_tables_per_model": joblib.load(leaf_path),
            }
    return _artifact_cache[key]


def predict_with_meta_confidence(models, row, backend: str, label: str, model_dir: str = None):
    """
    Returns (probability, confidence, calibration_method, meta_confidence).
    meta_confidence is None unless this horizon has a validated alert-only
    model AND this row's own probability crosses the decision threshold it
    was trained/validated on.
    """
    if model_dir is None:
        model_dir = _resolve_model_dir()

    prob, conf, method = predict_with_confidence_calibrated(models, row, backend, label, model_dir=model_dir)

    recommendation = _load_recommendation(model_dir)
    horizon_rec = recommendation.get(label)
    if not horizon_rec or not horizon_rec.get("use_meta"):
        return prob, conf, method, None

    threshold_decision = horizon_rec["threshold_decision"]
    if prob < threshold_decision:
        # Outside the population the meta-model was trained/validated on
        # (alerts only) -- don't compute or return a score here.
        return prob, conf, method, None

    artifacts = _load_artifacts(model_dir, backend, label)
    if artifacts is None:
        return prob, conf, method, None

    feature_columns = da.get_feature_columns()
    # row (built from state["row"], a dict that also carries a string
    # timestamp) has dtype=object as a WHOLE Series -- any subset/rebuild
    # inherits that object dtype per column. decision_agent.predict_with_confidence()
    # never hits this because it hands LightGBM a raw numpy array (no dtype
    # validation on that path); compute_row_level_stats() below explicitly
    # builds a pandas DataFrame, and LightGBM validates DataFrame dtypes
    # strictly. Cast to float BEFORE rebuilding, or this crashes with
    # "pandas dtypes must be int, float or bool" the moment a real pipeline
    # row (not a clean-dtype test fixture) reaches this function.
    X_row = row[feature_columns].astype(float).values.reshape(1, -1)

    # existing_confidence, raw_ensemble_std -- same wrapper every other
    # diagnostic in this project uses, applied to a single row.
    X_single = pd.DataFrame(X_row, columns=feature_columns)
    _, raw_std, existing_conf = compute_row_level_stats(models, X_single)

    # leaf support, via the SAVED lookup tables (no rebuild at inference time)
    per_model_support, per_model_pos = [], []
    for m, (total_tables, pos_tables) in zip(models, artifacts["leaf_tables_per_model"]):
        support, pos_support = score_leaf_support(m, X_single, total_tables, pos_tables)
        per_model_support.append(support.mean(axis=1))
        per_model_pos.append(pos_support.mean(axis=1))
    mean_support = np.mean(per_model_support, axis=0)
    mean_pos_support = np.mean(per_model_pos, axis=0)

    feature_vec = np.column_stack([
        np.array([prob]), existing_conf, raw_std,
        np.log1p(mean_support), np.log1p(mean_pos_support),
    ])
    feature_vec_s = artifacts["scaler"].transform(feature_vec)
    meta_confidence = float(artifacts["model"].predict_proba(feature_vec_s)[0, 1])

    return prob, conf, method, meta_confidence


if __name__ == "__main__":
    model_dir = _resolve_model_dir()
    rec = _load_recommendation(model_dir)
    if not rec:
        print(f"No meta_confidence_recommendation.json found in {model_dir} yet -- "
              f"run meta_confidence_diagnostics.py first.")
    else:
        print(f"Loaded meta-confidence recommendation for {len(rec)} horizon(s) from {model_dir}:")
        for label, r in rec.items():
            if r.get("use_meta"):
                print(f"  {label}: ACTIVE -- {r['reason']}")
            else:
                print(f"  {label}: not used -- {r['reason']}")
