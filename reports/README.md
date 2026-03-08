# Auditron Report Generation

This directory is self-contained. A new agent on a fresh server can generate reports from training logs using the files here.

## Files

- `gen_report_template.py` — the report generator script
- `REPORT_GENERATION.md` — full spec: what sections to include, design decisions, data format notes, run-by-run changelog
- `README.md` — this file

## What a report is

A self-contained HTML file with embedded base64 charts (no external deps). Each report tells the story of one training run — primarily the auditor's reasoning arc from random guessing to precise fraud detection.

Output goes to:
```
/home/hacka/hackathons/nm_game/public/auditron/report_RUN##.html
```
Public URL:
```
https://aibizbrain.com/nm_game/static/auditron/report_RUN##.html
```

## How to generate a report

```bash
RUN_NUMBER=31 LOGS_DIR=/tmp/run31_logs python3 reports/gen_report_template.py
```

The script auto-discovers log files inside `LOGS_DIR`:
- `train.log` — TRL training steps, reward/loss/entropy
- `reasoning.jsonl` — per-step completions (auditor quotes live here)
- `episodes.jsonl` — per-episode eval data (TPR/FPR, per-personality stats)
- `eval.json` — optional episode-level summary (absent for run 23+)

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `RUN_NUMBER` | `17` | Run number — used in output filename |
| `LOGS_DIR` | `/tmp/runN_logs` | Directory containing log files |
| `TRAIN_LOG` | `$LOGS_DIR/train.log` | Override train log path |
| `REASONING` | `$LOGS_DIR/reasoning.jsonl` | Override reasoning log path |
| `EPISODES` | `$LOGS_DIR/episodes.jsonl` | Override episodes log path |
| `EVAL_JSON` | `$LOGS_DIR/eval.json` | Override eval json path |

## Where log files come from

Training runs on a Northflank H100 service (`hackathon` project, `jupyter-pytorch` service). After a run completes, download logs using the NF CLI:

```bash
NF=/home/cc/.local/bin/northflank

# Download all logs to /tmp/runN_logs/
mkdir -p /tmp/run32_logs
$NF download service file --projectId hackathon --serviceId jupyter-pytorch \
    --remotePath /home/jovyan/OpenEnvNorthflankEnv/train.log \
    --localPath /tmp/run32_logs/train.log
$NF download service file --projectId hackathon --serviceId jupyter-pytorch \
    --remotePath /home/jovyan/OpenEnvNorthflankEnv/reasoning_*.jsonl \
    --localPath /tmp/run32_logs/reasoning.jsonl
$NF download service file --projectId hackathon --serviceId jupyter-pytorch \
    --remotePath /home/jovyan/OpenEnvNorthflankEnv/episodes_*.jsonl \
    --localPath /tmp/run32_logs/episodes.jsonl
```

If you don't know the exact filenames, exec a `ls` on the H100 first:
```bash
$NF exec service --projectId hackathon --serviceId jupyter-pytorch \
    --cmd 'ls /home/jovyan/OpenEnvNorthflankEnv/*.jsonl *.log' --shell-cmd 'bash -c'
```

## Run number tracking

Last known runs and their reports:

| Run | Report | Notes |
|-----|--------|-------|
| 18–22 | report_18.html … report_22.html | Run 22: switched to 4 sellers |
| 23 | report_23.html | No eval.json; hero stats from episodes.jsonl |
| 25, 26 | report_25.html, report_26.html | Weak models, many zeros |
| 28 | report_28.html | |
| 30 | report_30.html | First run with `final_round` entry type |
| 31 | report_31.html | Latest as of 2026-03-08 |

Next run will be **32+**. Check for new log files before generating.

## Design decisions (read REPORT_GENERATION.md for full detail)

- **Amber color scheme**: `#f59e0b` accent, `#0d0a00` background — set in script constants
- **The story is the auditor quotes**: judges spend most time in "Auditor Reasoning Milestones" — make sure quotes show a clear arc from vague → specific
- **Auditor only knows seller IDs**: never substitute personality names into quote text
- **Profit line is always green** (`#4ade80`) across all personality charts, shared y-axis (0–$10,300)
- **No "same supplier both recommended and flagged"**: lowest-cost unflagged supplier = recommended
- **Sections removed**: Part Failure Rate, Buyer Trust in Auditor, Buyer Reasoning, Auditor Accuracy Over Time, What is TPR box, Runtime line

## Design sandbox

`/home/hacka/hackathons/nm_game/public/auditron/reports_lab/report_amber.html` is the hand-crafted ideal demo report. Use it as a visual reference when evaluating generated reports.
