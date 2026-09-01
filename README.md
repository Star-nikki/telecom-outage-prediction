# 5G Network Outage Prediction — v2 (Improvements #6 → #1)

This version adds all six reviewed improvements on top of the original
simulator. Everything is code — **nothing has been run for you.** Run each
step yourself and share the actual output back for discussion.

## What changed, file by file

| File | Status | Improvement |
|---|---|---|
| `config.py` | modified | #6 multi-horizon list, #3 mobility event types, #2 real weather path, #1 streaming speed |
| `behavior.py` | rewritten | #3 named probabilistic mobility events (festival/cricket_match/concert/emergency) |
| `weather.py` | modified | #2 loader that prefers real weather if available, else synthetic |
| `fetch_historical_weather.py` | **new** | #2 pulls real historical weather from Open-Meteo |
| `simulate.py` | modified | #6 multi-horizon labels, #2/#3 wired to updated weather/behavior |
| `streaming_simulator.py` | **new** | #1 replays the dataset tick-by-tick like a live feed |
| `agents/llm_providers.py` | **new** | LLM-agnostic interface (Claude/GPT/Llama/offline rule-based) |
| `agents/decision_agent.py` | **new** | #6 per-horizon models + comparison table, #5 confidence, #4 reasoning |

`towers.py` and `failures.py` are unchanged.

---

## Step-by-step: how to run everything, in order

### 0. Setup (once)

```bash
pip install pandas numpy pyarrow scikit-learn joblib requests --break-system-packages
```

### 1. (Optional but recommended) Fetch real historical weather — Improvement #2

```bash
python fetch_historical_weather.py
```

**What this does:** calls the free Open-Meteo Archive API for 6 real cities
(edit `REGION_COORDINATES` in the script to match your actual regions) and
writes `data/real_weather.csv`. Any date beyond "today" (impossible to have
real weather for the future) is automatically backfilled with the same
synthetic generator used before — the script prints exactly which date
range was backfilled, so you can report this limitation honestly.

**If you skip this step:** `simulate.py` will automatically fall back to
fully synthetic weather, same as before. Nothing breaks either way.

### 2. Generate the dataset — now with multi-horizon labels + named events

```bash
python simulate.py
```

**What to expect:** same ~60-tower, 3-year run as before, but now:
- 7 label columns instead of 1: `outage_in_next_15m`, `_30m`, `_1h`, `_3h`,
  `_6h`, `_12h`, `_24h`
- A new `data/mobility_events.csv` ground-truth log of named events
  (festival/cricket_match/concert/emergency) — when, where, how big
- If step 1 was run, weather columns reflect real historical data instead
  of synthetic

Runtime will be similar to before (a few minutes on modest hardware) — the
extra columns don't meaningfully change compute cost.

### 3. Train the Decision Agent across all horizons — Improvements #6, #5, #4

```bash
cd agents
python decision_agent.py --mode train
```

**What this does:** trains a separate Random Forest per horizon (15m
through 24h) using a **time-based split** (train on earlier dates, test on
later ones — not random, since this is a forecasting task and random splits
would leak future information into training). Prints a table like:

```
[ 15m] AUC-PR=...  precision=...  recall=...  positive_rate=...
[ 30m] AUC-PR=...  ...
...
[ 24h] AUC-PR=...  ...
```

and saves it to `agents/models/horizon_comparison.csv`. **This table is the
Improvement #6 deliverable** — you should see AUC-PR highest at short
horizons and decay toward longer ones. That decay curve is your evidence
that short-horizon prediction is closer to detection and long-horizon is
genuinely harder forecasting — a legitimate result to discuss in your
capstone, whatever shape it actually takes on your machine.

### 4. See the Decision Agent reason about a specific tower — Improvement #4 + #5

```bash
python decision_agent.py --mode explain --tower T0000
```

(Replace `T0000` with any tower ID from your `data/towers.csv`. Check
`data/failure_events.csv` for tower IDs that actually experienced a failure
— reasoning is far more interesting on those than on a tower that never
degraded.)

**What this does:** picks a moment when that tower was degraded, builds a
structured evidence summary across all four conceptual agents (Tower
Health, Traffic, Weather, User Behavior), shows probability + **confidence**
for every horizon, then generates a natural-language explanation of the
likely cause.

**By default this uses the offline rule-based provider** (no API key, no
internet needed) so it works immediately. To use a real LLM instead:
```bash
export ANTHROPIC_API_KEY=your_key_here
pip install anthropic --break-system-packages
python decision_agent.py --mode explain --tower T0000 --provider anthropic
```
Swap `--provider openai` or `--provider ollama` the same way (see
`agents/llm_providers.py` for what each needs installed/configured).

### 5. (Optional demo) Run the streaming simulator — Improvement #1

```bash
cd ..   # back to scripts/ folder
python streaming_simulator.py
```

**What this does:** replays the dataset you already generated tick-by-tick
(one tick = 15 simulated minutes), printing an alert the instant any tower
enters a degraded or at-risk state — simulating what a live monitoring
system watching the feed in real time would see. By default it fast-forwards
(see `STREAMING_SPEED_MULTIPLIER` in `config.py`); set it to `1` for true
real-time pacing, or `None` for no delay at all (fastest possible replay).
Press Ctrl+C to stop early — it's 3 years of data, you don't need to watch
all of it.

---

## Known limitations to state honestly in your report

- **Real weather is capped at "today."** Any future date in your simulation
  window necessarily uses synthetic weather — this is unavoidable and the
  script tells you exactly which rows were backfilled.
- **The rule-based reasoning provider is a template, not real reasoning** —
  it's there so the pipeline runs without any API key, but the actual
  "agentic reasoning" quality only shows up once you plug in a real LLM
  provider.
- **Confidence here is tree-agreement, not calibrated uncertainty** — it's a
  reasonable proxy but isn't the same as a formally calibrated confidence
  interval. Worth stating plainly rather than overclaiming.
- **Horizon comparison results will vary by run** — since failure events are
  randomly scheduled, your exact AUC-PR numbers will differ from any
  example numbers mentioned above. What should stay consistent is the
  *shape*: short horizons outperforming long ones.
