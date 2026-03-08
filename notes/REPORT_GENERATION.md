# Post-Training Report Generation

After every training run completes, Claude Code reads the logs and writes a report
that tells the story of what the agents learned. This is for judges and demos.

## Output location
Reports are self-contained HTML files (NOT PNGs) at:
  /home/hacka/hackathons/nm_game/public/auditron/report_YYYYMMDD_HHMMSS.html

Public URL:
  https://aibizbrain.com/nm_game/static/auditron/report_YYYYMMDD_HHMMSS.html

Never overwrite — always use unique timestamped filenames.
Charts are embedded as base64 PNG inside the HTML (fully self-contained, no external deps).
Static files are served by Express under /static/ from the public/ dir on port 3005.

## Log files to read (all timestamped)
- `reasoning_*.jsonl`  — per-step completions: agent type, reason text, scores
- `episodes_*.jsonl`   — per-episode eval: per-round auditor/buyer/supplier actions
- `eval_*.json`        — episode-level summary metrics
- `train.log`          — TRL step-by-step reward/loss/entropy numbers

## Agent reasoning policy
- **Suppliers**: NO reason field. JSON is {"bid_price": X, "actual_strength": Y} only.
  Reasoning reward = 0 always. Log bid/actual/cheat/profit data only.
- **Auditor**: reason + flags fields. THIS IS THE STORY. Score reasoning reward here only.
  Log every reason text, word count, keyword hits, pick, flags.
- **Buyer**: Keep reason field (readable, interesting) but DO NOT score reasoning reward.
  Log reason text for display in reports only.

## Valid JSON logging — IMPORTANT for accurate charting
The interesting valid-JSON chart is: per training step, what % of completions returned valid JSON?
Each completion is logged with valid_json: true/false (added in quick6+).
- Supplier successes: logged in economic_reward with valid_json=True
- Auditor/buyer successes: logged in format_reward with valid_json=True
- ALL failures (any agent): logged in format_reward with valid_json=False, parse_error=True

Chart: group by step, compute valid_json=True / total per step. Smooth and plot.

DO NOT use `format_reward/mean` from train.log — per-batch average, misleading (supplier batches
always 2.0, auditor batches always -1.0 regardless of actual JSON validity).

## Per-round eval logging — required for sections 6, 7, 8
evaluate_model() must log a "round_detail" entry for EVERY round of EVERY eval episode.
These enable: per-supplier profit/bids-won/cheating charts, part failure rate, buyer-follows-auditor.

Required fields per round_detail entry (append to episodes_*.jsonl):
```json
{
  "type": "round_detail",
  "eval_step": 25,          // which periodic eval checkpoint
  "episode": 1,
  "round": 14,
  "auditor_pick": "supplier_2",   // auditor's recommended supplier
  "auditor_flags": ["supplier_3"],
  "buyer_pick": "supplier_2",     // who buyer actually chose
  "buyer_followed_auditor": true,
  "winner": "supplier_2",         // same as buyer_pick
  "winner_cheating": false,       // did winner submit actual_strength < required?
  "winner_actual_strength": 75.0,
  "required_strength": 70.0,
  "winner_bid_price": 87.5,
  "winner_profit": 15.2,
  "part_failed": false,
  "per_supplier": {               // all 5 suppliers this round
    "supplier_1": {"bid_price": 91.2, "won": false, "cheating": false},
    "supplier_2": {"bid_price": 87.5, "won": true,  "cheating": false},
    ...
  }
}
```
Also keep the existing episode-summary entries (type="eval_summary") for sections 5 and 11.

## What periodic evals must log (every EVAL_EVERY=25 steps)
Run 1 quick episode every 25 training steps and append to episodes_*.jsonl:
- step number
- auditor TPR and FPR (the key accuracy metric)
- per-supplier: profit, bids_won, parts_failed, personality
- buyer: total_spend, total_penalties, reward
- auditor: reward
This is what enables the "over time" graphs.

## Eval episodes
EVAL_EPISODES=3 for quick runs — just a sanity check.
Average all episodes together for reporting — DO NOT report them separately.
Show one combined row per run: avg failures, avg buyer_spend, avg auditor TPR/FPR.

## Report rules
- NEVER reference previous runs. Each report stands alone. No "fixes applied vs run X".
- Lead with economic outcomes (profit, spend, failures) not reward values — reward is internal.
- Valid JSON % is a small subplot, not a hero callout. Expected to hit ~100% fast.
- TPR = True Positive Rate: fraction of actual cheaters the auditor correctly flagged. Explain it.
- Failures = # of 50 parts that structurally failed (dishonest supplier shipped weak part → failure → 10x penalty).
- Report profit, not reward, wherever possible.

## Report sections (in order)

### 1. Header callouts (hero numbers — economic outcomes)
- Buyer total spend (vs what they'd spend buying randomly)
- Part failures out of 50
- Auditor fraud detection rate (TPR — explain: "caught X of Y cheaters")
- Total supplier profit (who made money?)
- Runtime / model / steps (small, bottom)

### 2. Valid JSON % (SMALL plot — not important, tucked beside reward curve)
- Source: reasoning_*.jsonl — use the `parse_error` field per entry (NOT format_reward from train.log)
  format_reward from train.log is per-batch not per-completion; supplier batches are always 2.0
  regardless of actual validity, so it looks like a useless on/off signal.
- Count: entries where parse_error=False (or "parse_error" key absent) / total entries
- Expected to shoot to ~100% quickly and stay there
- Keep it small — embed as a subplot or a narrow panel next to the reward curve, not full-width

### 3. Total Reward Curve (graph)
- Smoothed total reward, ±1 std band
- Stacked component breakdown (format / economic / reasoning)

### 4. The Auditor Learns to Reason (MAIN STORY — quotes + graphs)
Pull auditor reason text from reasoning_*.jsonl.
Select quotes at ~step 1, 25%, 50%, 75%, 100%.
Show evolution: empty → naming suppliers → citing failure history → full analysis.

TWO GRAPHS (these are the STAR charts — users love them):
  a) Auditor reason word count over steps (scatter + smoothed line)
  b) Evidence quality: % of reasons mentioning supplier names / failures / prices / rounds over steps

### 5. Auditor Accuracy Over Time (graph) ← requires periodic evals
- TPR (caught real cheaters) and FPR (wrongly flagged honest suppliers) over training steps
- Explain in plain English: "By step N, the auditor correctly identified X% of cheaters"
- Source: episodes_*.jsonl entries with type="periodic_eval"

### 6. Per-Personality Charts (5 separate line charts, one per personality type) ← requires periodic evals
IMPORTANT: supplier_1...supplier_5 are randomly reassigned a personality each episode.
NEVER label charts by supplier ID. Always label by personality name (Honest, Shrewd, Mostly Honest,
Dishonest, Opportunistic). Aggregate all suppliers that had that personality across episodes.

Each chart is a LINE CHART with x = training step (checkpoint steps: 0, 40, 80 for quick runs).
Each chart shows 3 lines over checkpoint steps for that personality type:
  a) Avg profit per episode — primary axis
  b) % of bids won — secondary axis
  c) % of rounds where they cheated

Source: episodes_*.jsonl entries with type="periodic_eval" — logged at each checkpoint.
  Field: avg_personality_profits — keyed by personality name (e.g. "Honest", "Dishonest")
Also use round_detail entries for bids_won and cheat % per personality per checkpoint.

NOTE: Supplier profit during the 80 gradient-update training steps is NOT tracked —
those steps train agents on individual completions, not full games. Profit only exists
in the checkpoint eval episodes. This is why the x-axis is checkpoint steps, not training steps.
CHECKPOINT_EVERY=40 for quick runs (2 checkpoints: step 40, step 80 + final eval).
For real runs (500+ steps), set CHECKPOINT_EVERY=50 for smoother curves.

### 7. Part Failure Rate Over Time (graph) ← requires periodic evals
- Per eval episode: % of the 50 rounds where the winning part structurally failed
- Expected: starts high (cheaters win often early), drops as auditor gets better
- Source: episodes_*.jsonl round_detail entries — count rounds where part_failed=True

### 8. Buyer Follows Auditor % Over Time (graph) ← requires periodic evals
- Per eval episode: % of rounds where buyer.pick == auditor.pick (buyer followed the recommendation)
- Expected: should trend toward 100% as auditor gets a reliable track record
- Source: episodes_*.jsonl round_detail entries — compare auditor_pick vs buyer_pick per round
- Also interesting: of the times buyer DIDN'T follow auditor, how often did they get burned?

### 9. Auditor Reasoning Quotes — THE STORY
- 5-6 quotes spanning training, each with step, word count, evidence badge pills
- These are the money shot for judges

### 10. Buyer Reasoning Quotes (logged, not scored — for narrative interest only)
- 3 quotes: early / mid / late
- Did the buyer learn to reference auditor advice?

### 11. Eval Episode Results Table
- Averaged across all eval episodes (not listed separately)
- Columns: eval_step, avg_failures, avg_buyer_spend, avg_auditor_TPR, avg_auditor_FPR,
    per-personality avg_profit
- NEVER include a "Supplier Slot" column (supplier_1...supplier_5). Personalities are randomly
  reshuffled each episode so the slot is meaningless. Only personality name matters.
- "Cheated" column must be a PERCENTAGE (% of rounds they submitted actual_strength < required),
  not a boolean Yes/No. Compute from round_detail entries: cheating rounds / total rounds played.

## How to generate the report
```python
# After training completes:
import glob, json
reasoning_file = sorted(glob.glob("reasoning_*.jsonl"))[-1]
episode_file   = sorted(glob.glob("episodes_*.jsonl"))[-1]

# Auditor entries (agent == "auditor")
entries = [json.loads(l) for l in open(reasoning_file)]
auditor = [e for e in entries if e.get("agent") == "auditor"]
# Sample quotes at 4-5 points
n = len(auditor)
samples = [auditor[i] for i in [0, n//4, n//2, 3*n//4, n-1]]

# Periodic eval data for over-time graphs
evals = [json.loads(l) for l in open(episode_file) if '"periodic"' in l]
```

## Output format
Self-contained HTML with all charts as base64-embedded PNGs.
Path: /home/hacka/hackathons/nm_game/public/auditron/report_YYYYMMDD_HHMMSS.html
URL:  https://aibizbrain.com/nm_game/static/auditron/report_YYYYMMDD_HHMMSS.html

## How to generate the report

```python
# After training completes:
# 1. Find the latest log files
import glob, os
reasoning_file = sorted(glob.glob("reasoning_*.jsonl"))[-1]
episode_file   = sorted(glob.glob("episodes_*.jsonl"))[-1]
eval_file      = sorted(glob.glob("eval_*.json"))[-1]

# 2. Pull quote samples from reasoning log
import json
entries = [json.loads(l) for l in open(reasoning_file)]
auditor_entries = [e for e in entries if e.get("agent") == "auditor" and "reason" in e]
# Sample at 4 points across training
n = len(auditor_entries)
samples = [auditor_entries[i] for i in [0, n//4, n//2, -1]]

# 3. Generate graphs (matplotlib, dark theme, save to /home/hacka/.../auditron/)
# 4. Write a markdown summary with embedded image links
```

## Tone for judge-facing report
- Lead with the STORY not the numbers: "The auditor started by picking randomly.
  By step 250, it was citing failure histories by name."
- Highlight the emergent behavior: Opportunistic supplier being honest early
- Connect to OpenEnv sub-themes: Scalable Oversight, Multi-Actor Environments
- Keep it punchy — judges read dozens of submissions

## File to write after each run
Save as: /home/openenv/notes/run_reports/report_YYYYMMDD_HHMMSS.md
