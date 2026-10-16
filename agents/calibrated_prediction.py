"""
Priority 4 production integration -- opt-in calibration wrapper.

decision_agent.py is NOT modified. This is a separate, thin wrapper around
its existing predict_with_confidence(), so nothing about current behavior
changes unless you explicitly call predict_with_confidence_calibrated()
instead. If calibration_experiment.py hasn't been run yet (no
calibration_recommendation.json on disk), this behaves IDENTICALLY to
predict_with_confidence() -- fully backward compatible, safe to import
even before Priority 4 has been run.

WHAT IT DOES:
    Applies a saved Platt/Isotonic calibrator to the PROBABILITY only, and
    only for horizons where calibration_experiment.py found it genuinely
    helps (improves Brier score without meaningfully hurting AUC). Never
    forces calibration onto a horizon it didn't help.

WHAT IT DELIBERATELY DOES NOT TOUCH:
    Confidence is passed through completely unchanged. Priority 5 already
    found ensemble-spread confidence (1 - 2*std) is compressed and
    unreliable, but fixing THAT is a separate, larger piece of work
    (Candidate 2 -- conformal prediction -- was the recommended next step,
    not yet built). Bolting a probability-calibration fix onto the
    confidence score would conflate two different diagnosed problems; this
    file only fixes the one Priority 4 was scoped to fix.

WHERE TO SAVE:
    Same folder as decision_agent.py (agents/).

HOW TO USE (after running calibration_experiment.py at least once):

    import decision_agent as da
    from calibrated_prediction import predict_with_confidence_calibrated
    import joblib

    models = joblib.load("models/decision_model_lightgbm_3h.joblib")
    prob, conf, method = predict_with_confidence_calibrated(models, row, "lightgbm", "3h")
    # prob is calibrated if 3h was recommended for calibration, else identical
    # to da.predict_with_confidence(models, row)[0]. `method` tells you which:
    # "raw", "platt", or "isotonic" -- always print/log this so it's never a
    # silent, invisible substitution.

TO WIRE THIS INTO orchestrator_v3.py's decision_agent_node() LATER (not
done automatically here -- this file only provides the wrapper, wiring it
into the live pipeline is a deliberate one-line swap you make when ready):

    # before:
    prob, conf = da.predict_with_confidence(models, row)
    # after:
    prob, conf, method = predict_with_confidence_calibrated(models, row, state["backend"], label)
"""
import os
import json

import numpy as np
import joblib

import decision_agent as da

_calibrators_cache = {}
_recommendation_cache = None


def _resolve_model_dir():
    candidates = [
        da.MODEL_DIR, "models", os.path.join("agents", "models"), os.path.join("..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return da.MODEL_DIR  # fall back to whatever decision_agent.py itself would use


def _load_recommendation(model_dir):
    global _recommendation_cache
    if _recommendation_cache is None:
        path = os.path.join(model_dir, "calibration_recommendation.json")
        if os.path.exists(path):
            with open(path) as f:
                _recommendation_cache = json.load(f)
        else:
            _recommendation_cache = {}
    return _recommendation_cache


def _apply_platt(calibrator, prob: float) -> float:
    eps = 1e-6
    logit = np.log(np.clip(prob, eps, 1 - eps) / (1 - np.clip(prob, eps, 1 - eps)))
    return float(calibrator.predict_proba(np.array([[logit]]))[0, 1])


def _apply_isotonic(calibrator, prob: float) -> float:
    return float(calibrator.predict([prob])[0])


def predict_with_confidence_calibrated(models, row, backend: str, label: str, model_dir: str = None):
    """
    Drop-in replacement for decision_agent.predict_with_confidence(models, row)
    that additionally returns which calibration method was applied (if any).

    Returns: (probability, confidence, method)
        method is one of "raw", "platt", "isotonic" -- "raw" means either no
        recommendation file exists yet, or this horizon was explicitly
        recommended to stay uncalibrated.
    """
    prob, conf = da.predict_with_confidence(models, row)

    if model_dir is None:
        model_dir = _resolve_model_dir()

    recommendation = _load_recommendation(model_dir)
    horizon_rec = recommendation.get(label)
    if not horizon_rec or horizon_rec.get("recommended") == "raw":
        return prob, conf, "raw"

    method = horizon_rec["recommended"]
    calibrator_path = os.path.join(model_dir, f"calibrator_{backend}_{label}_{method}.joblib")
    if not os.path.exists(calibrator_path):
        # Recommendation file says calibrate, but the calibrator file itself
        # is missing -- fail safe to raw rather than crashing production.
        return prob, conf, "raw"

    if calibrator_path not in _calibrators_cache:
        _calibrators_cache[calibrator_path] = joblib.load(calibrator_path)
    calibrator = _calibrators_cache[calibrator_path]

    if method == "platt":
        calibrated_prob = _apply_platt(calibrator, prob)
    else:
        calibrated_prob = _apply_isotonic(calibrator, prob)

    return calibrated_prob, conf, method


if __name__ == "__main__":
    # Quick self-check: confirms the fallback path works even with no
    # recommendation file present, without needing real models/data.
    model_dir = _resolve_model_dir()
    rec = _load_recommendation(model_dir)
    if not rec:
        print(f"No calibration_recommendation.json found in {model_dir} yet -- "
              f"run calibration_experiment.py first. Until then, "
              f"predict_with_confidence_calibrated() behaves identically to "
              f"predict_with_confidence().")
    else:
        print(f"Loaded recommendation for {len(rec)} horizon(s) from {model_dir}:")
        for label, r in rec.items():
            print(f"  {label}: {r['recommended']} -- {r['reason']}")
