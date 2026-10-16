"""
LLM-as-judge automated evaluation harness.

PURPOSE (as scoped in the handoff notes): automate DETECTION of issues
across the full dataset -- not fixing them, and not a substitute for the
diagnostic scripts already built for Priority 4/5 (calibration_diagnostics.py,
confidence_diagnostics.py, leaf_support_diagnostics.py cover the ML/statistics
side; this harness covers the LLM-output-quality side those scripts don't
touch: is the root-cause reasoning coherent, is the recommendation actually
grounded and non-contradictory, is the final report internally consistent).

WHAT IT DOES:
    For a sample of tower/snapshot combinations, runs the REAL
    orchestrator_llm.py pipeline (same build_graph()/AgentState machinery
    the CLI and dashboard use -- no reimplementation), then asks a judge
    LLM to score the resulting output against a fixed rubric:

      1. GROUNDING    -- does every recommended action trace back to a
                          retrieved knowledge-base SOP or the historical
                          context, with no invented facts/numbers?
      2. CONSISTENCY   -- does the root cause reasoning agree with the
                          evidence section (no contradictions like "sudden
                          onset" reasoning paired with a "gradual creep"
                          recommendation -- the exact bug pattern found and
                          fixed earlier in advanced_rag.py)?
      3. COMPLETENESS  -- does the report actually address the predicted
                          failure category (not a generic answer that
                          would fit any category)?
      4. CLARITY       -- would a telecom operations engineer be able to
                          act on this without needing to ask a follow-up
                          question?

    Each dimension is scored 1-5 by the judge with a required one-sentence
    justification (forces the judge to commit to specific evidence rather
    than a bare number). Results are aggregated into a summary CSV/report
    flagging the worst-scoring cases for manual review -- this harness
    finds problems, it doesn't fix them.

WHY THIS DOESN'T REPLACE THE EXISTING VALIDATION LOGIC:
    orchestrator_llm.py already has hardened, deterministic, zero-tolerance
    grounding checks (_recommendation_is_grounded, negation-aware phrase
    blocking, etc.) that REJECT bad LLM output before it ever reaches a
    report -- that's a hard gate, always on, no API cost. This harness is a
    SEPARATE, softer, sampling-based check that runs AFTER that gate, to
    catch subtler quality issues (like the earlier capacity_creep/
    congestion_collapse contradiction) that pass the hard grounding checks
    because every individual claim IS traceable to a retrieved SOP, but the
    combination is still internally inconsistent. Different failure mode,
    different tool.

WHERE TO SAVE:
    Same folder as orchestrator_llm.py (agents/).

RUN:
    python eval_harness.py --n-samples 20 --judge-provider anthropic
    python eval_harness.py --n-samples 20 --judge-provider ollama --judge-model llama3.1
    python eval_harness.py --towers T0000 T0001 T0002   # specific towers instead of random sample

COST NOTE: each sample = 1 orchestrator run (which itself makes 1-2 LLM
calls for root cause + recommendation) + 1 judge call. With --judge-provider
rule_based (default), the judge is a crude keyword-based stand-in with no
API cost, so you can test the harness end-to-end for free before spending
on real judge calls -- see RuleBasedJudge below for exactly what it checks
and does NOT check.
"""
import argparse
import os
import sys
import json
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
import orchestrator_v3 as o3
import orchestrator_llm as ollm
from llm_providers import get_provider

RUBRIC_DIMENSIONS = ["grounding", "consistency", "completeness", "clarity"]

JUDGE_PROMPT_TEMPLATE = """You are auditing an AI-generated telecom network incident report for
quality issues. Score it on four dimensions, 1 (severe problem) to 5 (no issue).

REPORT TO AUDIT:
{report}

DIMENSIONS:
1. GROUNDING: Does every recommended action trace back to the retrieved
   knowledge base references or historical context shown in the report? Flag
   any invented facts, numbers, or claims not supported by what's shown.
2. CONSISTENCY: Does the root cause reasoning agree with the evidence and the
   recommended actions? Flag any contradiction (e.g. reasoning says "sudden
   onset" but a recommendation is written for a "gradual" problem).
3. COMPLETENESS: Do the recommended actions actually address the specific
   predicted root cause category, rather than being generic enough to fit
   any failure type?
4. CLARITY: Could a telecom operations engineer act on this without needing
   to ask a follow-up question?

Respond ONLY with valid JSON, no other text, in exactly this shape:
{{
  "grounding": {{"score": <1-5>, "justification": "<one sentence>"}},
  "consistency": {{"score": <1-5>, "justification": "<one sentence>"}},
  "completeness": {{"score": <1-5>, "justification": "<one sentence>"}},
  "clarity": {{"score": <1-5>, "justification": "<one sentence>"}}
}}"""


class RuleBasedJudge:
    """
    No API key, no cost -- a crude keyword-based stand-in so the harness
    can be tested end-to-end for free. It can genuinely catch the exact
    contradiction pattern found earlier (capacity-creep language paired
    with an acute/sudden-onset root cause), but it CANNOT judge grounding,
    completeness, or clarity in any real sense -- those dimensions get a
    flat neutral score of 3 with a justification saying so, honestly,
    rather than faking a judgment a keyword matcher can't make. Use a real
    LLM judge before trusting scores on those three dimensions.
    """

    CONTRADICTION_PAIRS = [
        (["sudden onset", "acute", "not a gradual"], ["capacity planning review",
                                                        "organic demand growth"]),
        (["hardware", "power supply"], ["no maintenance action", "self-resolving"]),
    ]

    def judge(self, report: str) -> dict:
        text = report.lower()
        consistency_score = 5
        consistency_note = "No known contradiction pattern detected (keyword check only)."
        for acute_terms, gradual_terms in self.CONTRADICTION_PAIRS:
            if any(t in text for t in acute_terms) and any(t in text for t in gradual_terms):
                consistency_score = 1
                consistency_note = (
                    f"Found both an acute-onset phrase and a contradictory gradual/other-category "
                    f"phrase in the same report -- likely the mismatched-SOP pattern found earlier.")
                break

        return {
            "grounding": {"score": 3, "justification": "RuleBasedJudge cannot verify grounding -- "
                                                         "use a real LLM judge for this dimension."},
            "consistency": {"score": consistency_score, "justification": consistency_note},
            "completeness": {"score": 3, "justification": "RuleBasedJudge cannot verify completeness -- "
                                                            "use a real LLM judge for this dimension."},
            "clarity": {"score": 3, "justification": "RuleBasedJudge cannot verify clarity -- "
                                                       "use a real LLM judge for this dimension."},
        }


class LLMJudge:
    def __init__(self, provider_name, **kwargs):
        self.provider = get_provider(provider_name, **kwargs)

    def judge(self, report: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(report=report)
        raw = self.provider.generate(prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        try:
            parsed = json.loads(cleaned)
            for dim in RUBRIC_DIMENSIONS:
                if dim not in parsed or "score" not in parsed[dim]:
                    raise ValueError(f"Missing dimension '{dim}' in judge response")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            return {dim: {"score": None, "justification": f"JUDGE PARSE FAILURE: {e}. Raw: {raw[:200]}"}
                    for dim in RUBRIC_DIMENSIONS}


def run_one_sample(df, tower_id, backend, provider, judge, hours_before=3.0):
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    if len(tdf) == 0:
        return None
    tdf = da.add_trend_features(tdf)
    row_series = o3.pick_demo_row(tdf, tower_id, hours_before_outage=hours_before)
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])

    initial_state = {
        "tower_id": tower_id, "row": row, "evidence": {}, "predictions": {},
        "primary_probability": 0.0, "primary_confidence": 0.0, "threshold": 0.0,
        "root_cause_category": None, "root_cause_reasoning": None,
        "knowledge_context": None, "historical_context": None, "recommendation": None,
        "final_report": None, "provider_name": provider, "backend": backend,
        "primary_horizon": "3h",
    }
    app = ollm.build_graph()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    if final_state.get("root_cause_category") is None:
        return {"tower_id": tower_id, "skipped": True,
                "reason": "below alert threshold, no report generated"}

    report = final_state.get("final_report", "")
    judgment = judge.judge(report)

    row_result = {"tower_id": tower_id, "skipped": False,
                  "root_cause_category": final_state["root_cause_category"]}
    for dim in RUBRIC_DIMENSIONS:
        row_result[f"{dim}_score"] = judgment[dim]["score"]
        row_result[f"{dim}_justification"] = judgment[dim]["justification"]
    row_result["report_text"] = report
    return row_result


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10,
                         help="How many random tower/snapshot combos to evaluate.")
    parser.add_argument("--towers", nargs="*", default=None,
                         help="Specific tower IDs instead of a random sample.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--llm-provider", default="rule_based",
                         help="Provider for the orchestrator's own root-cause/recommendation LLM calls.")
    parser.add_argument("--judge-provider", default="rule_based",
                         choices=["rule_based", "anthropic", "openai", "ollama", "phi4_local"],
                         help="Provider for the JUDGE. 'rule_based' here uses RuleBasedJudge "
                              "(free, limited -- see its docstring) rather than llm_providers' "
                              "RuleBasedProvider.")
    parser.add_argument("--judge-model", default=None,
                         help="Model name override for the judge provider (e.g. an Ollama model tag).")
    parser.add_argument("--hours-before", type=float, default=3.0)
    args = parser.parse_args()

    print(f"Loading {o3.DATA_PATH} ...")
    df = pd.read_parquet(o3.DATA_PATH)

    if args.towers:
        sample_towers = args.towers
    else:
        all_towers = sorted(df["tower_id"].unique())
        import random
        random.seed(42)
        sample_towers = random.sample(all_towers, min(args.n_samples, len(all_towers)))

    if args.judge_provider == "rule_based":
        judge = RuleBasedJudge()
        print("Judge: RuleBasedJudge (free, keyword-based -- grounding/completeness/clarity "
              "are NOT really evaluated, see docstring). Use --judge-provider anthropic/ollama/etc. "
              "for a real judgment.")
    else:
        kwargs = {"model": args.judge_model} if args.judge_model else {}
        judge = LLMJudge(args.judge_provider, **kwargs)
        print(f"Judge: LLMJudge via provider '{args.judge_provider}'.")

    print(f"Evaluating {len(sample_towers)} tower/snapshot combinations "
          f"(llm_provider={args.llm_provider}, backend={args.backend})...\n")

    results = []
    for i, tower_id in enumerate(sample_towers):
        print(f"[{i+1}/{len(sample_towers)}] {tower_id} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = run_one_sample(df, tower_id, args.backend, args.llm_provider, judge,
                                     args.hours_before)
        except Exception as e:
            result = {"tower_id": tower_id, "skipped": True, "reason": f"ERROR: {e}"}
        if result is None:
            result = {"tower_id": tower_id, "skipped": True, "reason": "no data for this tower"}
        results.append(result)
        elapsed = time.time() - t0
        if result.get("skipped"):
            print(f"skipped ({result['reason']}) [{elapsed:.1f}s]")
        else:
            scores = [result[f"{d}_score"] for d in RUBRIC_DIMENSIONS if result[f"{d}_score"] is not None]
            avg = sum(scores) / len(scores) if scores else float("nan")
            print(f"avg score {avg:.1f}/5 [{elapsed:.1f}s]")

    results_df = pd.DataFrame(results)
    results_df.to_csv("eval_harness_results.csv", index=False)
    print(f"\nSaved eval_harness_results.csv ({len(results_df)} rows)")

    evaluated = results_df[~results_df.get("skipped", pd.Series([False] * len(results_df)))]
    if len(evaluated) == 0:
        print("No samples were actually evaluated (all skipped) -- check the skip reasons above.")
        return

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Evaluated: {len(evaluated)} / {len(results_df)} (rest skipped, see CSV for reasons)")
    for dim in RUBRIC_DIMENSIONS:
        col = f"{dim}_score"
        valid = evaluated[col].dropna()
        if len(valid):
            print(f"  {dim:14s}: mean={valid.mean():.2f}  min={valid.min()}  "
                  f"# scored <=2 (flagged): {(valid <= 2).sum()}")

    flagged = evaluated[
        (evaluated[[f"{d}_score" for d in RUBRIC_DIMENSIONS]].min(axis=1) <= 2)
    ]
    if len(flagged):
        print(f"\n{len(flagged)} report(s) flagged for manual review (any dimension scored <=2):")
        for _, r in flagged.iterrows():
            worst_dim = min(RUBRIC_DIMENSIONS, key=lambda d: (r[f"{d}_score"] if pd.notna(r[f"{d}_score"]) else 99))
            print(f"  {r['tower_id']}: worst dimension = {worst_dim} "
                  f"(score {r[f'{worst_dim}_score']}) -- {r[f'{worst_dim}_justification']}")
        flagged.to_csv("eval_harness_flagged.csv", index=False)
        print("\nSaved eval_harness_flagged.csv (full text for manual review)")
    else:
        print("\nNo reports flagged for manual review in this sample.")


if __name__ == "__main__":
    run()
