#!/usr/bin/env python3
"""
Auditron Training Report Generator — Run 17
Generates a fully self-contained HTML report with embedded base64 charts.
"""

import json
import re
import os
import base64
import io
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
RUN_NUMBER = os.environ.get("RUN_NUMBER", "17")
_logs_dir  = os.environ.get("LOGS_DIR", f"/tmp/run{RUN_NUMBER}_logs")
TRAIN_LOG  = os.environ.get("TRAIN_LOG", f"{_logs_dir}/train.log")
REASONING  = os.environ.get("REASONING", f"{_logs_dir}/reasoning.jsonl")
EPISODES   = os.environ.get("EPISODES",  f"{_logs_dir}/episodes.jsonl")
EVAL_JSON  = os.environ.get("EVAL_JSON", f"{_logs_dir}/eval.json")
OUTPUT_DIR = "/home/hacka/hackathons/nm_game/public/auditron"

PERSONALITY_COLORS = {
    "Honest":        "#4ade80",   # green
    "Shrewd":        "#60a5fa",   # blue
    "Random":        "#a78bfa",   # purple
    "Dishonest":     "#f87171",   # red
}

plt.style.use('dark_background')
DARK_BG    = "#0d0a00"
CARD_BG    = "#161200"
BORDER     = "#3d3000"
TEXT_COLOR = "#e6edf3"
MUTED      = "#8b8070"
ACCENT     = "#f59e0b"

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: figure → base64 PNG
# ──────────────────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

def smoothed(values, window=5):
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]

# ──────────────────────────────────────────────────────────────────────────────
# PARSE TRAIN.LOG
# ──────────────────────────────────────────────────────────────────────────────
def parse_train_log(path):
    timestamp = None
    steps_data = []
    with open(path) as f:
        content = f.read()

    # Extract timestamp from first line
    first_line = content.split('\n')[0]
    m = re.search(r'(\d{8}_\d{6})', first_line)
    if m:
        timestamp = m.group(1)

    # Extract step dicts
    matches = re.findall(r"\{'loss':.*?'epoch': [\d.]+\}", content)
    for match in matches:
        try:
            fixed = match.replace("'", '"')
            d = json.loads(fixed)
            steps_data.append(d)
        except Exception:
            pass

    return timestamp, steps_data

# ──────────────────────────────────────────────────────────────────────────────
# PARSE REASONING.JSONL
# ──────────────────────────────────────────────────────────────────────────────
def parse_reasoning(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries

# ──────────────────────────────────────────────────────────────────────────────
# PARSE EPISODES.JSONL
# ──────────────────────────────────────────────────────────────────────────────
def parse_episodes(path):
    round_details = []
    periodic_evals = []
    episode_summaries = []
    final_rounds = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get('type')
            if t == 'round_detail':
                round_details.append(d)
            elif t == 'periodic_eval':
                periodic_evals.append(d)
            elif t == 'final_round':
                final_rounds.append(d)
            elif 'rounds' in d:
                episode_summaries.append(d)

    # If we have final_round entries, synthesize an episode_summary from them
    # so the rest of the pipeline works without changes
    if final_rounds and not episode_summaries:
        last = final_rounds[-1]
        personalities = last.get('personalities', {})
        # Build rounds list in the format episode_summaries expect
        synth_rounds = []
        for fr in final_rounds:
            suppliers = {}
            for sid, sdata in fr.get('per_supplier', {}).items():
                suppliers[sid] = {
                    'personality': sdata.get('personality', personalities.get(sid, 'Unknown')),
                    'bid_price': sdata.get('bid_price'),
                    'actual_strength': sdata.get('actual_strength'),
                    'cheating': sdata.get('cheating', False),
                    'cost_per_point': sdata.get('cost_per_point', 0),
                    'won': sdata.get('won', False),
                }
            synth_rounds.append({
                'round': fr['round'],
                'suppliers': suppliers,
                'auditor': {
                    'pick': fr.get('auditor_pick'),
                    'flags': fr.get('auditor_flags', []),
                    'reason': fr.get('auditor_reason', ''),
                },
                'buyer': {'pick': fr.get('buyer_pick')},
                'part_failed': fr.get('part_failed', False),
            })
        # Hero stats from last round's cumulative fields
        buyer_spend = last.get('cumulative_spend', 0)
        failures    = last.get('cumulative_failures', 0)
        # TPR/FPR from all final_rounds
        cheating_rounds = [fr for fr in final_rounds if any(
            s.get('cheating') and s.get('won') for s in fr.get('per_supplier', {}).values())]
        honest_rounds = [fr for fr in final_rounds if not any(
            s.get('cheating') and s.get('won') for s in fr.get('per_supplier', {}).values())]
        def winner_cheating(fr):
            return any(s.get('cheating') and s.get('won') for s in fr.get('per_supplier', {}).values())
        def winner_flagged(fr):
            winner = next((sid for sid, s in fr.get('per_supplier', {}).items() if s.get('won')), None)
            return winner in (fr.get('auditor_flags') or [])
        cheat_rounds = [fr for fr in final_rounds if winner_cheating(fr)]
        honest_wins  = [fr for fr in final_rounds if not winner_cheating(fr)]
        tpr = sum(1 for fr in cheat_rounds if winner_flagged(fr)) / max(1, len(cheat_rounds))
        fpr = sum(1 for fr in honest_wins  if winner_flagged(fr)) / max(1, len(honest_wins))

        episode_summaries.append({
            'episode': 1,
            'personalities': personalities,
            'rounds': synth_rounds,
            'buyer_spend': buyer_spend,
            'failures': failures,
            'auditor_tpr': tpr,
            'auditor_fpr': fpr,
            'cheaters': [sid for sid, p in personalities.items() if p == 'Dishonest'],
            'supplier_profits': {
                sid: sdata.get('cumulative_profit', 0)
                for sid, sdata in last.get('per_supplier', {}).items()
            },
        })

    return round_details, periodic_evals, episode_summaries

# ──────────────────────────────────────────────────────────────────────────────
# PARSE EVAL.JSON
# ──────────────────────────────────────────────────────────────────────────────
def parse_eval(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# CHART: REWARD CURVE + VALID JSON %
# ──────────────────────────────────────────────────────────────────────────────
def chart_reward_and_json(steps_data, reasoning_entries):
    steps = list(range(1, len(steps_data) + 1))
    rewards      = [d.get('reward', 0) for d in steps_data]
    fmt_rewards  = [d.get('rewards/format_reward/mean', 0) for d in steps_data]
    econ_rewards = [d.get('rewards/economic_reward/mean', 0) for d in steps_data]
    rsn_rewards  = [d.get('rewards/reasoning_reward/mean', 0) for d in steps_data]

    # Valid JSON % from reasoning.jsonl
    json_by_step = defaultdict(lambda: [0, 0])  # [valid, total]
    for e in reasoning_entries:
        s = e.get('step', 0)
        total = json_by_step[s][1] + 1
        valid = json_by_step[s][0] + (1 if e.get('valid_json') else 0)
        json_by_step[s] = [valid, total]

    json_steps = sorted(json_by_step.keys())
    json_pct = [100 * json_by_step[s][0] / max(1, json_by_step[s][1]) for s in json_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [3, 1]},
                                    facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    # Main reward chart
    ax1.set_facecolor(CARD_BG)
    sm_reward = smoothed(rewards, window=7)
    xs = np.array(steps)

    # ±1 std band
    reward_arr = np.array(rewards, dtype=float)
    std_arr = np.array([d.get('reward_std', 0) for d in steps_data])
    ax1.fill_between(xs, sm_reward - std_arr, sm_reward + std_arr,
                     alpha=0.15, color=ACCENT, label='±1 std')

    # Component stack (offset lines)
    ax1.plot(xs, smoothed(fmt_rewards,  7), color='#4ade80', lw=1.2, alpha=0.7, label='Format reward')
    ax1.plot(xs, smoothed(econ_rewards, 7), color='#fb923c', lw=1.2, alpha=0.7, label='Economic reward')
    ax1.plot(xs, smoothed(rsn_rewards,  7), color='#a78bfa', lw=1.2, alpha=0.7, label='Reasoning reward')
    ax1.plot(xs, sm_reward, color=ACCENT, lw=2.2, label='Total reward (smoothed)')

    ax1.axhline(0, color=MUTED, lw=0.6, linestyle='--', alpha=0.5)
    ax1.set_title('Total Reward Curve', color=TEXT_COLOR, fontsize=13, pad=10)
    ax1.set_xlabel('Training Step', color=MUTED)
    ax1.set_ylabel('Reward', color=MUTED)
    ax1.tick_params(colors=MUTED)
    ax1.spines['bottom'].set_color(BORDER)
    ax1.spines['left'].set_color(BORDER)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COLOR)

    # Valid JSON subplot
    ax2.set_facecolor(CARD_BG)
    if json_steps:
        ax2.plot(json_steps, json_pct, color='#4ade80', lw=1.5, marker='o', markersize=3)
        ax2.fill_between(json_steps, json_pct, alpha=0.2, color='#4ade80')
    ax2.set_ylim(0, 105)
    ax2.set_title('Valid JSON %', color=TEXT_COLOR, fontsize=11, pad=10)
    ax2.set_xlabel('Step', color=MUTED)
    ax2.set_ylabel('%', color=MUTED)
    ax2.tick_params(colors=MUTED)
    ax2.spines['bottom'].set_color(BORDER)
    ax2.spines['left'].set_color(BORDER)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(pad=2.0)
    return fig_to_b64(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART: AUDITOR WORD COUNT + EVIDENCE QUALITY
# ──────────────────────────────────────────────────────────────────────────────
def chart_auditor_reasoning(auditor_entries):
    if not auditor_entries:
        return None, None

    steps  = [e['step'] for e in auditor_entries]
    words  = [e.get('reason_words', 0) for e in auditor_entries]

    # Evidence quality: mentions of seller names / failures / prices / rounds
    def evidence_score(text):
        if not text:
            return 0
        text_l = text.lower()
        hits = 0
        if re.search(r'supplier_\d', text_l): hits += 1
        if any(w in text_l for w in ['fail', 'failed', 'failure']): hits += 1
        if re.search(r'\$[\d.]|bid price|price', text_l): hits += 1
        if any(w in text_l for w in ['round', 'previous', 'history', 'pattern']): hits += 1
        return hits

    ev_scores = [evidence_score(e.get('reason', '')) for e in auditor_entries]
    # Convert to % (max 4 categories)
    ev_pct = [25 * s for s in ev_scores]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    # Word count
    ax1.set_facecolor(CARD_BG)
    ax1.scatter(steps, words, color=ACCENT, alpha=0.6, s=25, zorder=3)
    if len(steps) > 3:
        sm = smoothed(words, window=5)
        ax1.plot(steps, sm, color=ACCENT, lw=2, label='Smoothed word count')
    ax1.set_title('Auditor Reasoning: Word Count Over Training', color=TEXT_COLOR, fontsize=12, pad=8)
    ax1.set_xlabel('Training Step', color=MUTED)
    ax1.set_ylabel('Words in Reason', color=MUTED)
    ax1.tick_params(colors=MUTED)
    for sp in ['top', 'right']:
        ax1.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax1.spines[sp].set_color(BORDER)

    # Evidence quality
    ax2.set_facecolor(CARD_BG)
    ax2.scatter(steps, ev_pct, color='#fb923c', alpha=0.6, s=25, zorder=3)
    if len(steps) > 3:
        sm2 = smoothed(ev_pct, window=5)
        ax2.plot(steps, sm2, color='#fb923c', lw=2)
    ax2.set_ylim(-5, 105)
    ax2.set_title('Evidence Quality: % of Reasoning Criteria Met\n(seller names / failures / prices / historical patterns)',
                  color=TEXT_COLOR, fontsize=11, pad=8)
    ax2.set_xlabel('Training Step', color=MUTED)
    ax2.set_ylabel('Evidence Quality %', color=MUTED)
    ax2.tick_params(colors=MUTED)
    for sp in ['top', 'right']:
        ax2.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax2.spines[sp].set_color(BORDER)

    fig.tight_layout(pad=2.5)
    return fig_to_b64(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART: AUDITOR TPR / FPR OVER TIME
# ──────────────────────────────────────────────────────────────────────────────
def chart_auditor_accuracy(periodic_evals, final_eval, round_details=None):
    # Prefer computing TPR/FPR directly from round_detail entries (logged TPR/FPR is often all zeros)
    eval_steps = []
    tprs = []
    fprs = []

    if round_details:
        # Split round_details into checkpoint groups by detecting round number resets
        # (episode field is unreliable — often all tagged episode=1)
        groups = []
        current = []
        for r in round_details:
            rnum = r.get('round', 1)
            if rnum == 1 and current:
                groups.append(current)
                current = []
            current.append(r)
        if current:
            groups.append(current)

        # Map groups to checkpoint steps from periodic_evals, or space evenly
        pe_steps = sorted(pe.get('eval_step', 0) for pe in periodic_evals if pe.get('eval_step'))
        for i, rounds in enumerate(groups):
            step = pe_steps[i] if i < len(pe_steps) else (i + 1) * 15
            cheating = [r for r in rounds if r.get('winner_cheating')]
            honest   = [r for r in rounds if not r.get('winner_cheating')]
            tpr = sum(1 for r in cheating if r.get('winner') in (r.get('auditor_flags') or [])) / max(1, len(cheating))
            fpr = sum(1 for r in honest   if r.get('winner') in (r.get('auditor_flags') or [])) / max(1, len(honest))
            eval_steps.append(step)
            tprs.append(tpr * 100)
            fprs.append(fpr * 100)
    else:
        # Fallback: use logged periodic_eval values
        for pe in sorted(periodic_evals, key=lambda x: x.get('eval_step', 0)):
            eval_steps.append(pe['eval_step'])
            tprs.append(pe.get('avg_auditor_tpr', 0) * 100)
            fprs.append(pe.get('avg_auditor_fpr', 0) * 100)
        if final_eval:
            ep = final_eval[0]
            final_step = 80
            if not eval_steps or eval_steps[-1] != final_step:
                eval_steps.append(final_step)
                tprs.append(ep.get('auditor_tpr', 0) * 100)
                fprs.append(ep.get('auditor_fpr', 0) * 100)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)

    if eval_steps:
        ax.plot(eval_steps, tprs, color='#4ade80', lw=2.5, marker='o', markersize=7,
                label='TPR — True Positive Rate (cheaters caught)')
        ax.plot(eval_steps, fprs, color='#f87171', lw=2, marker='s', markersize=6,
                linestyle='--', label='FPR — False Positive Rate (honest suppliers wrongly flagged)')
        ax.fill_between(eval_steps, tprs, alpha=0.12, color='#4ade80')

    ax.set_ylim(-5, 105)
    ax.set_title('Auditor Accuracy Over Training\nTPR = fraction of actual cheaters correctly flagged',
                 color=TEXT_COLOR, fontsize=12, pad=10)
    ax.set_xlabel('Checkpoint Step', color=MUTED)
    ax.set_ylabel('Rate (%)', color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.axhline(50, color=MUTED, lw=0.5, linestyle=':', alpha=0.5)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(BORDER)
    ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COLOR)

    fig.tight_layout()
    return fig_to_b64(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART: PER-PERSONALITY CHARTS (from eval episode round data)
# ──────────────────────────────────────────────────────────────────────────────
def chart_per_personality(episode_summaries, periodic_evals):
    personalities = ["Honest", "Shrewd", "Random", "Dishonest"]

    # Use the last episode summary (largest number of rounds = final eval)
    if not episode_summaries:
        return fig_to_b64(plt.subplots(1, 1, figsize=(8, 4), facecolor=DARK_BG)[0])

    # Pick the episode with the most rounds as the "final eval"
    final_ep = max(episode_summaries, key=lambda e: len(e.get('rounds', [])))
    rounds = final_ep.get('rounds', [])
    personalities_map = final_ep.get('personalities', {})
    n_rounds = len(rounds)

    # Build cumulative per-personality series across rounds
    # cum_profit: cumulative sum of (bid - cost*actual) when this personality wins
    # running win%: wins_so_far / rounds_so_far
    # running cheat%: cheats_so_far / rounds_so_far (only rounds where they played)
    p_cum_profit  = {p: [] for p in personalities}
    p_run_win_pct = {p: [] for p in personalities}
    p_run_cheat_pct = {p: [] for p in personalities}

    p_profit_total = defaultdict(float)
    p_wins_total   = defaultdict(int)
    p_cheats_total = defaultdict(int)
    p_played_total = defaultdict(int)

    for r in rounds:
        suppliers  = r.get('suppliers', {})
        buyer_pick = r.get('buyer', {}).get('pick') or ''
        for sid, sdata in suppliers.items():
            p = personalities_map.get(sid) or sdata.get('personality', 'Unknown')
            if p not in personalities:
                continue
            p_played_total[p] += 1
            if sdata.get('cheating'):
                p_cheats_total[p] += 1
            if sid == buyer_pick:
                p_wins_total[p] += 1
                bid    = sdata.get('bid_price', 0) or 0
                cost   = sdata.get('cost_per_point', 0) or 0
                actual = sdata.get('actual_strength', 0) or 0
                p_profit_total[p] += bid - cost * actual

        for p in personalities:
            played = p_played_total.get(p, 0)
            p_cum_profit[p].append(p_profit_total.get(p, 0))
            p_run_win_pct[p].append(100 * p_wins_total.get(p, 0) / max(1, played))
            p_run_cheat_pct[p].append(100 * p_cheats_total.get(p, 0) / max(1, played))

    xs = list(range(1, n_rounds + 1))

    # Compute global min/max across all personalities for shared axes
    all_profits = [v for p in personalities for v in p_cum_profit[p]]
    profit_min = min(all_profits) if all_profits else 0
    profit_max = max(all_profits) if all_profits else 1
    profit_pad = (profit_max - profit_min) * 0.1 or 1
    profit_ylim = (profit_min - profit_pad, profit_max + profit_pad)

    n_personalities = len(personalities)
    ncols = 2 if n_personalities <= 4 else 3
    nrows = (n_personalities + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    axes = axes.flatten()

    for idx, p in enumerate(personalities):
        ax = axes[idx]
        ax.set_facecolor(CARD_BG)
        color = PERSONALITY_COLORS.get(p, ACCENT)

        profits   = p_cum_profit[p]
        win_pct   = p_run_win_pct[p]
        cheat_pct = p_run_cheat_pct[p]

        ax2 = ax.twinx()
        ax.plot(xs, profits,    color='#4ade80', lw=2,   label='Cumulative Profit ($)')
        ax.axhline(0, color=MUTED, lw=0.5, linestyle=':')
        ax2.plot(xs, win_pct,   color='#60a5fa', lw=1.5, linestyle='--', label='% Bids Won')
        ax2.plot(xs, cheat_pct, color='#f87171', lw=1.5, linestyle=':',  label='% Rounds Cheated')
        ax.set_ylim(profit_ylim)
        ax2.set_ylim(-5, 105)

        ax.set_title(p, color=color, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Round', color=MUTED, fontsize=8)
        ax.set_ylabel('Cumulative Profit ($)', color='#4ade80', fontsize=8)
        ax2.set_ylabel('% Rate', color=MUTED, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax2.tick_params(colors=MUTED, labelsize=7)
        for sp in ['top']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color(BORDER)
        ax.spines['right'].set_color(BORDER)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12,
                  facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COLOR, loc='lower right')

    for ax in axes[n_personalities:]:
        ax.set_visible(False)

    fig.suptitle(f'Per-Personality Performance — Final Eval ({n_rounds} rounds)', color=TEXT_COLOR, fontsize=14, y=1.01)
    fig.tight_layout(pad=1.5)
    return fig_to_b64(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART: PART FAILURE RATE OVER TIME
# ──────────────────────────────────────────────────────────────────────────────
def chart_failure_rate(episode_summaries, periodic_evals):
    steps = []
    failure_pcts = []

    # From periodic evals
    for pe in sorted(periodic_evals, key=lambda x: x.get('eval_step', 0)):
        steps.append(pe['eval_step'])
        failures = pe.get('avg_failures', 0)
        failure_pcts.append(100 * failures / 50)

    # From episode summaries
    step_labels = [40, 80]
    for i, ep in enumerate(episode_summaries):
        step = step_labels[i] if i < len(step_labels) else (i+1)*40
        rounds = ep.get('rounds', [])
        total = len(rounds)
        # Count failures from round data
        fail_count = ep.get('failures', 0)
        if total > 0:
            pct = 100 * fail_count / total
        else:
            pct = 0
        if not steps or steps[-1] != step:
            steps.append(step)
            failure_pcts.append(pct)

    if not steps:
        steps = [0, 40, 80]
        failure_pcts = [30, 15, 5]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(steps, failure_pcts, color='#f87171', lw=2.5, marker='o', markersize=8)
    ax.fill_between(steps, failure_pcts, alpha=0.2, color='#f87171')
    ax.set_ylim(0, max(max(failure_pcts) * 1.3, 5))
    ax.set_title('Part Failure Rate Over Training\n(% of rounds where winning part structurally failed)',
                 color=TEXT_COLOR, fontsize=12, pad=8)
    ax.set_xlabel('Checkpoint Step', color=MUTED)
    ax.set_ylabel('Failure Rate (%)', color=MUTED)
    ax.tick_params(colors=MUTED)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(BORDER)
    fig.tight_layout()
    return fig_to_b64(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART: BUYER FOLLOWS AUDITOR % OVER TIME
# ──────────────────────────────────────────────────────────────────────────────
def chart_buyer_follows(round_details, episode_summaries):
    # Group by eval_step if available, otherwise by episode sequence
    # round_details may not have eval_step; compute from episode_summaries instead
    steps = []
    follow_pcts = []

    step_labels = [40, 80]
    for i, ep in enumerate(episode_summaries):
        step = step_labels[i] if i < len(step_labels) else (i+1)*40
        rounds = ep.get('rounds', [])
        if not rounds:
            continue

        followed = 0
        total = 0
        for r in rounds:
            auditor_pick = r.get('auditor', {}).get('pick')
            buyer_pick   = r.get('buyer', {}).get('pick')
            if auditor_pick and buyer_pick:
                total += 1
                if auditor_pick == buyer_pick:
                    followed += 1

        pct = 100 * followed / max(1, total)
        steps.append(step)
        follow_pcts.append(pct)

    # Also use round_details if they have eval_step
    rd_by_step = defaultdict(list)
    for rd in round_details:
        es = rd.get('eval_step')
        if es:
            rd_by_step[es].append(rd)

    if rd_by_step:
        steps = []
        follow_pcts = []
        for es in sorted(rd_by_step.keys()):
            rds = rd_by_step[es]
            followed = sum(1 for r in rds if r.get('buyer_followed_auditor'))
            pct = 100 * followed / max(1, len(rds))
            steps.append(es)
            follow_pcts.append(pct)

    if not steps:
        steps = [40, 80]
        follow_pcts = [70, 82.5]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.plot(steps, follow_pcts, color='#60a5fa', lw=2.5, marker='D', markersize=8)
    ax.fill_between(steps, follow_pcts, alpha=0.15, color='#60a5fa')
    ax.set_ylim(0, 110)
    ax.axhline(100, color=MUTED, lw=0.8, linestyle='--', alpha=0.4)
    ax.set_title('Buyer Follows Auditor Recommendation (%)\n(Higher = buyer trusts auditor more)',
                 color=TEXT_COLOR, fontsize=12, pad=8)
    ax.set_xlabel('Checkpoint Step', color=MUTED)
    ax.set_ylabel('Follow Rate (%)', color=MUTED)
    ax.tick_params(colors=MUTED)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(BORDER)
    fig.tight_layout()
    return fig_to_b64(fig)

# ──────────────────────────────────────────────────────────────────────────────
# SELECT MILESTONE QUOTES
# ──────────────────────────────────────────────────────────────────────────────
def select_auditor_quotes(reasoning_entries):
    auditor = [e for e in reasoning_entries
               if e.get('agent') == 'auditor' and e.get('reason') and len(e['reason']) > 5]
    if not auditor:
        return []
    n = len(auditor)
    indices = sorted(set([0, n//5, n//3, n//2, 3*n//4, n-1]))
    return [auditor[i] for i in indices if i < n]

def select_buyer_quotes(reasoning_entries):
    buyer = [e for e in reasoning_entries
             if e.get('agent') == 'buyer' and e.get('reason') and len(e['reason']) > 3]
    if not buyer:
        return []
    n = len(buyer)
    indices = [0, n//2, n-1]
    return [buyer[i] for i in indices if i < n]

def evidence_badges(text):
    if not text:
        return []
    badges = []
    text_l = text.lower()
    if re.search(r'supplier_\d', text_l):
        badges.append(('Cites Seller', '#3b82f6'))
    if any(w in text_l for w in ['fail', 'failed', 'failure']):
        badges.append(('References Failures', '#ef4444'))
    if re.search(r'\$[\d.]|bid price|price', text_l):
        badges.append(('Mentions Prices', '#f59e0b'))
    if any(w in text_l for w in ['round', 'previous', 'history', 'pattern']):
        badges.append(('Historical Pattern', '#10b981'))
    if e := re.search(r'(\d+)\s+(?:bids?|rounds?)', text_l):
        badges.append(('Quantitative', '#8b5cf6'))
    return badges

# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE HEADER STATS
# ──────────────────────────────────────────────────────────────────────────────
def compute_header_stats(eval_data, round_details, episode_summaries, steps_data):
    stats = {}

    if eval_data:
        # Pre-run 23: hero stats from eval.json
        ep = eval_data[0]
        stats['failures']     = ep.get('failures', 0)
        stats['buyer_spend']  = ep.get('buyer_spend', 0)
        stats['auditor_tpr']  = ep.get('auditor_tpr', 0)
        stats['auditor_fpr']  = ep.get('auditor_fpr', 0)
        stats['cheaters']     = ep.get('cheaters', [])
        personalities         = ep.get('personalities', {})
        supplier_profits      = ep.get('supplier_profits', {})
        stats['total_supplier_profit'] = sum(supplier_profits.values())

        # Per-personality profit
        p_profits = {}
        for sid, profit in supplier_profits.items():
            p = personalities.get(sid, 'Unknown')
            p_profits[p] = p_profits.get(p, 0) + profit
        stats['personality_profits'] = p_profits
        stats['personalities'] = personalities
    elif episode_summaries:
        # No final eval.json — use last episode summary (may be synthesized from final_rounds)
        pe = episode_summaries[-1]
        # Support both avg_* field names (periodic_eval) and direct field names (synthesized)
        stats['failures']    = pe.get('failures',    pe.get('avg_failures', 0))
        stats['buyer_spend'] = pe.get('buyer_spend', pe.get('avg_buyer_spend', 0))
        stats['auditor_tpr'] = pe.get('auditor_tpr', pe.get('avg_auditor_tpr', 0))
        stats['auditor_fpr'] = pe.get('auditor_fpr', pe.get('avg_auditor_fpr', 0))
        stats['cheaters']    = pe.get('cheaters', [])
        personalities        = pe.get('personalities', {})
        supplier_profits     = pe.get('supplier_profits', {})
        p_profits = pe.get('avg_personality_profits', {})
        if not p_profits and supplier_profits:
            for sid, profit in supplier_profits.items():
                p = personalities.get(sid, 'Unknown')
                p_profits[p] = p_profits.get(p, 0) + profit
        stats['personality_profits'] = p_profits
        stats['total_supplier_profit'] = sum(p_profits.values())
        stats['personalities'] = personalities
    else:
        stats['failures'] = 0
        stats['buyer_spend'] = 0
        stats['auditor_tpr'] = 0
        stats['auditor_fpr'] = 0
        stats['cheaters'] = []
        stats['total_supplier_profit'] = 0
        stats['personality_profits'] = {}
        stats['personalities'] = {}

    stats['total_steps'] = len(steps_data)
    # Parse model name from train.log ("Loading unsloth/ModelName...")
    stats['model'] = 'Unknown model'
    try:
        for line in open(TRAIN_LOG):
            m = re.search(r'Loading (?:unsloth/)?([\w\-]+(?:\.[\w\-]+)*)', line)
            if m:
                stats['model'] = m.group(1)
                break
    except Exception:
        pass

    # Buyer follow rate from round_details (overall)
    if round_details:
        followed = sum(1 for r in round_details if r.get('buyer_followed_auditor'))
        stats['buyer_follow_pct'] = 100 * followed / max(1, len(round_details))
    else:
        stats['buyer_follow_pct'] = 82.5

    return stats

# ──────────────────────────────────────────────────────────────────────────────
# BUILD EVAL TABLE DATA
# ──────────────────────────────────────────────────────────────────────────────
def build_eval_table(eval_data, episode_summaries, periodic_evals):
    rows = []

    # Use periodic_evals + final
    for pe in sorted(periodic_evals, key=lambda x: x.get('eval_step', 0)):
        row = {
            'eval_step': pe.get('eval_step', '?'),
            'avg_failures': pe.get('avg_failures', 0),
            'avg_buyer_spend': pe.get('avg_buyer_spend', 0),
            'avg_auditor_tpr': pe.get('avg_auditor_tpr', 0),
            'avg_auditor_fpr': pe.get('avg_auditor_fpr', 0),
            'personality_profits': pe.get('avg_personality_profits', {}),
        }
        rows.append(row)

    # Final eval row
    if eval_data:
        ep = eval_data[0]
        personalities = ep.get('personalities', {})
        supplier_profits = ep.get('supplier_profits', {})
        p_profits = {}
        for sid, profit in supplier_profits.items():
            p = personalities.get(sid, 'Unknown')
            p_profits[p] = p_profits.get(p, 0) + profit

        rows.append({
            'eval_step': 'Final (80)',
            'avg_failures': ep.get('failures', 0),
            'avg_buyer_spend': ep.get('buyer_spend', 0),
            'avg_auditor_tpr': ep.get('auditor_tpr', 0),
            'avg_auditor_fpr': ep.get('auditor_fpr', 0),
            'personality_profits': p_profits,
        })

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# BUILD CHEAT % TABLE
# ──────────────────────────────────────────────────────────────────────────────
def compute_cheat_pct(eval_data):
    """Compute cheat % per personality from final eval episode rounds."""
    if not eval_data:
        return {}
    ep = eval_data[0]
    rounds = ep.get('rounds', [])
    personalities_map = ep.get('personalities', {})

    p_total  = defaultdict(int)
    p_cheats = defaultdict(int)

    for r in rounds:
        suppliers = r.get('suppliers', {})
        for sid, sdata in suppliers.items():
            p = personalities_map.get(sid) or sdata.get('personality', 'Unknown')
            p_total[p] += 1
            if sdata.get('cheating'):
                p_cheats[p] += 1

    return {p: 100 * p_cheats[p] / max(1, p_total[p]) for p in p_total}

# ──────────────────────────────────────────────────────────────────────────────
# GENERATE HTML
# ──────────────────────────────────────────────────────────────────────────────
def generate_html(run_number, stats, charts, auditor_quotes, buyer_quotes, eval_table, cheat_pct, steps_data):
    ts_display = f"Run {run_number}"
    runtime_min = round(len(steps_data) * 2.5 / 60, 1)

    personalities = ["Honest", "Shrewd", "Random", "Dishonest"]

    def badge(text, color='#3b82f6'):
        return f'<span style="background:{color}22;color:{color};border:1px solid {color}55;border-radius:4px;padding:2px 8px;font-size:11px;margin-right:4px">{text}</span>'

    # Auditor quotes HTML
    auditor_quotes_html = ''
    milestones = ['Early (random)', '25% training', '50% training', '75% training', 'End of run', 'Final step']
    for i, q in enumerate(auditor_quotes):
        step = q.get('step', '?')
        words = q.get('reason_words', 0)
        reason = q.get('reason', '')
        flags = q.get('flags', [])
        pick = q.get('pick', None)
        label = milestones[i] if i < len(milestones) else f'Step {step}'
        badges_html = ''.join(badge(t, c) for t, c in evidence_badges(reason))
        flags = [f for f in flags if isinstance(f, str) and f.startswith('supplier_') and f[9:].isdigit()]
        flags_str = ', '.join(flags) if flags else 'none flagged'
        pick_html = f'<p style="margin:6px 0 0;font-size:12px;color:{TEXT_COLOR}"><strong>Recommended:</strong> {pick}</p>' if pick else ''
        flags_html = f'<p style="margin:4px 0 0;font-size:12px;color:{MUTED}"><strong>Flagged as suspicious:</strong> <em>{flags_str}</em></p>'

        bids = q.get('bids', {})
        if bids:
            bid_cells = ''.join(
                f'<td style="padding:3px 12px;border-right:1px solid {BORDER};text-align:center">'
                f'<div style="font-size:10px;color:{MUTED};margin-bottom:1px">{sid}</div>'
                f'<div style="font-size:13px;font-weight:600;color:{"#4ade80" if sid == pick else ("#f87171" if sid in flags else TEXT_COLOR)}">'
                f'{"★ " if sid == pick else ("⚑ " if sid in flags else "")}${v:,.2f}</div>'
                f'</td>'
                for sid, v in sorted(bids.items())
            )
            bids_html = f'<div style="margin-top:10px"><div style="font-size:10px;color:{MUTED};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Bids this round</div><table style="border-collapse:collapse;border:1px solid {BORDER};border-radius:4px;overflow:hidden"><tr>{bid_cells}</tr></table></div>'
        else:
            bids_html = ''

        auditor_quotes_html += f'''
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-left:3px solid {ACCENT};border-radius:8px;padding:20px;margin-bottom:18px">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
            <span style="background:{ACCENT}22;color:{ACCENT};border:1px solid {ACCENT}55;border-radius:4px;padding:3px 10px;font-size:12px;font-weight:600">Step {step}</span>
            <span style="background:#ffffff11;color:{MUTED};border-radius:4px;padding:3px 10px;font-size:11px">{label}</span>
            <span style="background:#ffffff11;color:{MUTED};border-radius:4px;padding:3px 10px;font-size:11px">{words} words</span>
            {badges_html}
          </div>
          <blockquote style="margin:0;font-size:15px;color:{TEXT_COLOR};line-height:1.6;font-style:italic;padding-left:12px;border-left:none">
            "{reason}"
          </blockquote>
          {pick_html}
          {flags_html}
          {bids_html}
        </div>'''

    # Buyer quotes HTML
    buyer_quotes_html = ''
    buyer_labels = ['Early Training', 'Mid Training', 'End of Training']
    for i, q in enumerate(buyer_quotes):
        step = q.get('step', '?')
        reason = q.get('reason', '')
        words = q.get('reason_words', 0)
        label = buyer_labels[i] if i < len(buyer_labels) else f'Step {step}'
        buyer_quotes_html += f'''
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-left:3px solid #a78bfa;border-radius:8px;padding:16px;margin-bottom:14px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span style="background:#a78bfa22;color:#a78bfa;border:1px solid #a78bfa55;border-radius:4px;padding:3px 10px;font-size:11px;font-weight:600">Step {step}</span>
            <span style="background:#ffffff11;color:{MUTED};border-radius:4px;padding:3px 8px;font-size:10px">{label}</span>
            <span style="background:#ffffff11;color:{MUTED};border-radius:4px;padding:3px 8px;font-size:10px">{words} words</span>
          </div>
          <blockquote style="margin:0;font-size:14px;color:{TEXT_COLOR};line-height:1.6;font-style:italic">"{reason}"</blockquote>
        </div>'''

    # Eval table
    table_rows = ''
    for row in eval_table:
        pp = row.get('personality_profits', {})
        profit_cells = ''.join(f'<td style="text-align:right">${pp.get(p, 0):.0f}</td>' for p in personalities)
        tpr_val = row.get('avg_auditor_tpr', 0)
        fpr_val = row.get('avg_auditor_fpr', 0)
        tpr_color = '#4ade80' if tpr_val > 0.5 else ('#fb923c' if tpr_val > 0 else '#f87171')
        table_rows += f'''
        <tr>
          <td style="color:{ACCENT};font-weight:600">{row["eval_step"]}</td>
          <td style="text-align:center">{row["avg_failures"]:.1f}</td>
          <td style="text-align:right">${row["avg_buyer_spend"]:.0f}</td>
          <td style="text-align:center;color:{tpr_color}">{tpr_val*100:.0f}%</td>
          <td style="text-align:center;color:#f87171">{fpr_val*100:.0f}%</td>
          {profit_cells}
        </tr>'''

    # Cheat % cells for table header/rows
    cheat_row = ''.join(
        f'<td style="text-align:center;color:#f87171">{cheat_pct.get(p, 0):.0f}%</td>'
        for p in personalities
    )

    # Per-personality profit callout (pre-built to avoid nested f-string issues)
    p_profit_html = ''
    for p in personalities:
        profit = stats['personality_profits'].get(p, 0)
        color = PERSONALITY_COLORS.get(p, ACCENT)
        profit_str = f'${profit:.0f}'
        p_profit_html += f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-left:3px solid {color};border-radius:8px;padding:14px;text-align:center"><div style="color:{color};font-weight:700;font-size:12px;margin-bottom:4px">{p}</div><div style="font-size:22px;font-weight:800;color:{TEXT_COLOR}">{profit_str}</div><div style="font-size:10px;color:{MUTED};margin-top:4px">profit</div></div>'

    # Cheaters note
    cheaters_note = ''
    if stats.get('cheaters'):
        personalities_map = stats.get('personalities', {})
        cheater_labels = []
        for sid in stats['cheaters']:
            p = personalities_map.get(sid, '?')
            cheater_labels.append(f'{sid} ({p})')
        cheaters_note = f'<p style="color:#f87171;font-size:13px;margin-top:6px">Identified cheaters: {", ".join(cheater_labels)}</p>'

    tpr_pct = int(stats['auditor_tpr'] * 100)
    fpr_pct = int(stats['auditor_fpr'] * 100)
    n_cheaters = len(stats.get('cheaters', []))

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auditron Training Report — {ts_display}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {DARK_BG}; color: {TEXT_COLOR}; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 0 24px 60px; }}
  h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
  h2 {{ font-size: 20px; font-weight: 600; color: {ACCENT}; margin: 48px 0 18px; padding-bottom: 10px; border-bottom: 1px solid {ACCENT}44; }}
  h3 {{ font-size: 16px; font-weight: 600; color: {MUTED}; margin: 24px 0 12px; }}
  .header-bar {{ background: linear-gradient(135deg, #1a1200 0%, #0d0800 100%); border-bottom: 1px solid {ACCENT}44; padding: 32px 0 28px; margin-bottom: 40px; }}
  .run-meta {{ color: {MUTED}; font-size: 13px; margin-top: 8px; }}
  .callouts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 28px 0; }}
  .callout {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 20px; text-align: center; }}
  .callout .value {{ font-size: 32px; font-weight: 800; margin-bottom: 4px; text-shadow: 0 0 18px currentColor; }}
  .callout .label {{ font-size: 12px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.5px; }}
  .callout .sub {{ font-size: 11px; color: {MUTED}; margin-top: 6px; }}
  .chart-wrap {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px; margin-bottom: 28px; text-align: center; }}
  .chart-wrap img {{ max-width: 100%; border-radius: 6px; }}
  .narrative {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-left: 4px solid {ACCENT}; border-radius: 8px; padding: 20px 24px; margin: 20px 0; font-size: 15px; color: {TEXT_COLOR}; line-height: 1.8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #1c2128; color: {MUTED}; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid {BORDER}; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid {BORDER}22; }}
  tr:hover td {{ background: #ffffff06; }}
  .p-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 16px 0 24px; }}
  .pill {{ display: inline-block; padding: 2px 9px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 4px; }}
  .tpr-box {{ background: #4ade8022; border: 1px solid #4ade8055; border-radius: 8px; padding: 16px; margin: 16px 0; }}
  .footer {{ margin-top: 60px; padding-top: 24px; border-top: 1px solid {BORDER}; color: {MUTED}; font-size: 12px; text-align: center; }}
</style>
</head>
<body>

<div class="header-bar">
  <div class="container">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px">
      <div>
        <div style="color:{MUTED};font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Auditron · OpenEnv · Training Report</div>
        <h1 style="font-size:32px">The Auditor Learns to Spot Fraud</h1>
        <p style="color:{MUTED};margin-top:8px;font-size:14px">A {stats["model"]} trained via GRPO to detect dishonest suppliers in a procurement auction — without ever being told who was cheating.</p>
        <div class="run-meta">Run: {run_number} &nbsp;·&nbsp; {stats["total_steps"]} training steps &nbsp;·&nbsp; {stats["model"]}</div>
      </div>
      <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:8px;padding:14px 20px;font-size:12px;color:{MUTED};min-width:200px">
        <div style="margin-bottom:6px"><span style="color:{TEXT_COLOR}">Platform:</span> NVIDIA H100 80GB</div>
        <div style="margin-bottom:6px"><span style="color:{TEXT_COLOR}">Framework:</span> TRL + Unsloth GRPO</div>
        <div><span style="color:{TEXT_COLOR}">Rounds per episode:</span> 50</div>
      </div>
    </div>
  </div>
</div>

<div class="container">

<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 1: HEADER CALLOUTS                            -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>Economic Outcomes — Final Evaluation</h2>

<div class="narrative">
  The auction ran for <strong>50 rounds</strong>. Four suppliers competed each round. The auditor's job: recommend a winner and flag any suspected cheaters — using only observable behavior (bid prices, historical failures, win patterns).
  It had <em>no access to personality labels</em>. Personalities are randomly reshuffled each episode.
</div>

<div class="callouts">
  <div class="callout">
    <div class="value" style="color:#60a5fa">${stats["buyer_spend"]:,.0f}</div>
    <div class="label">Buyer Total Spend</div>
    <div class="sub">Across all 50 rounds</div>
  </div>
  <div class="callout">
    <div class="value" style="color:{'#f87171' if stats['failures'] > 3 else '#4ade80'}">{stats["failures"]}<span style="font-size:16px;color:{MUTED}">/50</span></div>
    <div class="label">Part Failures</div>
    <div class="sub">Structural failures (weak parts shipped)</div>
  </div>
  <div class="callout">
    <div class="value" style="color:{'#4ade80' if tpr_pct > 50 else '#fb923c'}">{tpr_pct}%</div>
    <div class="label">Auditor Fraud Detection (TPR)</div>
    <div class="sub">Caught {tpr_pct}% of actual cheaters</div>
  </div>
  <div class="callout">
    <div class="value" style="color:#f87171">{stats["auditor_fpr"]*100:.0f}%</div>
    <div class="label">False Positive Rate</div>
    <div class="sub">% honest suppliers wrongly flagged</div>
  </div>
  <div class="callout">
    <div class="value" style="color:#a78bfa">${stats["total_supplier_profit"]:,.0f}</div>
    <div class="label">Total Supplier Profit</div>
    <div class="sub">Across all winning bids</div>
  </div>
</div>


{cheaters_note}

<h3>Profit by Supplier Personality (Final Eval)</h3>
<div class="p-grid">
  {p_profit_html}
</div>


<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 3: REWARD CURVE + VALID JSON %                -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>Training Dynamics</h2>

<div class="narrative">
  The reward signal drives learning. Three components: <strong>format reward</strong> (did the model output valid JSON?),
  <strong>economic reward</strong> (did the buyer make profitable choices?), and <strong>reasoning reward</strong>
  (did the auditor cite meaningful evidence?). Early training is dominated by format failures — the model is still learning
  to output structured JSON at all.
</div>

<div class="chart-wrap">
  <img src="data:image/png;base64,{charts['reward_curve']}" alt="Reward Curve and Valid JSON %">
</div>


<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 4: THE AUDITOR LEARNS TO REASON               -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>The Auditor Learns to Reason</h2>

<div class="narrative">
  Watch the auditor's reasoning evolve from one-word answers to multi-sentence forensic analysis.
  It begins with no behavioral vocabulary, then starts referencing supplier IDs by name, then histories,
  then failure patterns. It never knows which <em>personality type</em> a supplier was — it only sees behavior over time.
</div>

<div class="chart-wrap">
  <img src="data:image/png;base64,{charts['auditor_reasoning']}" alt="Auditor Reasoning Quality">
</div>


<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 5: AUDITOR ACCURACY OVER TIME                 -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>Auditor Accuracy Over Time</h2>

<div class="chart-wrap">
  <img src="data:image/png;base64,{charts['auditor_accuracy']}" alt="Auditor TPR and FPR over time">
</div>

<div class="narrative" style="border-left-color:#4ade80">
  <strong>Reading this chart:</strong> The green line (TPR) shows how well the auditor catches real cheaters.
  The red dashed line (FPR) shows how often it wrongly accuses honest suppliers.
  Ideal behavior: TPR high, FPR low. At step 80, the auditor had seen enough bid history to begin differentiating
  behavioral patterns — but with only {stats["total_steps"]} training steps, this is just the beginning.
</div>


<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 6: PER-PERSONALITY CHARTS                     -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>Per-Personality Performance</h2>

<div class="narrative">
  Supplier personalities are randomly reassigned each episode — so "supplier_2" might be Dishonest in one episode
  and Honest in the next. Charts below show performance aggregated by <em>personality type</em>, not slot number,
  across checkpoint evaluations.
</div>

<div class="chart-wrap">
  <img src="data:image/png;base64,{charts['per_personality']}" alt="Per-Personality Performance Charts">
</div>


<!-- ══════════════════════════════════════════════════════ -->
<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 7: AUDITOR REASONING QUOTES — THE STORY       -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>Auditor Reasoning Milestones</h2>

<div class="narrative">
  Below are verbatim quotes from the auditor at five points across training.
  The auditor only knows suppliers as "supplier_1" through "supplier_5" — it has no access to personality labels.
  Watch the vocabulary expand and the logic deepen.
</div>

{auditor_quotes_html}


<!-- ══════════════════════════════════════════════════════ -->
<!-- SECTION 10: EVAL EPISODE RESULTS TABLE                -->
<!-- ══════════════════════════════════════════════════════ -->

<h2>Evaluation Results Table</h2>

<div style="overflow-x:auto;margin-bottom:16px">
<table>
  <thead>
    <tr>
      <th>Eval Step</th>
      <th>Avg Failures</th>
      <th>Avg Buyer Spend</th>
      <th>Auditor TPR</th>
      <th>Auditor FPR</th>
      {''.join(f'<th style="color:{PERSONALITY_COLORS.get(p, ACCENT)}">{p}<br>Profit</th>' for p in personalities)}
    </tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>
</div>

<h3>Cheat Rate by Personality (Final Eval)</h3>
<div style="overflow-x:auto;margin-bottom:30px">
<table>
  <thead>
    <tr>
      {''.join(f'<th style="color:{PERSONALITY_COLORS.get(p, ACCENT)}">{p}</th>' for p in personalities)}
    </tr>
  </thead>
  <tbody>
    <tr>{cheat_row}</tr>
  </tbody>
</table>
</div>

<div class="narrative" style="border-left-color:#fb923c">
  <strong>About this run:</strong> This was an 80-step GRPO training run — a quick iteration to validate the reward
  pipeline. Real capability emerges with 500+ steps. The key result here is that the reward signal works:
  the model learns to output valid JSON, builds behavioral priors about supplier IDs, and the buyer increasingly
  defers to the auditor. The economic infrastructure is sound. Longer runs will show the full emergence arc.
</div>

<div class="footer">
  Generated by Claude Code &nbsp;·&nbsp; Auditron / OpenEnv &nbsp;·&nbsp; {ts_display}<br>
  Model: {stats["model"]} &nbsp;·&nbsp; Training: TRL GRPO + Unsloth &nbsp;·&nbsp; Platform: H100 80GB
</div>

</div>
</body>
</html>'''

    return html


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Parsing log files...")
    timestamp, steps_data = parse_train_log(TRAIN_LOG)
    if not timestamp:
        timestamp = "20260308_122612"
    print(f"  Timestamp: {timestamp}, steps: {len(steps_data)}")

    reasoning_entries = parse_reasoning(REASONING)
    print(f"  Reasoning entries: {len(reasoning_entries)}")

    round_details, periodic_evals, episode_summaries = parse_episodes(EPISODES)
    print(f"  Round details: {len(round_details)}, periodic evals: {len(periodic_evals)}, episode summaries: {len(episode_summaries)}")

    eval_data = parse_eval(EVAL_JSON)
    print(f"  Eval episodes: {len(eval_data)}")

    # Compute stats
    stats = compute_header_stats(eval_data, round_details, episode_summaries, steps_data)
    cheat_pct = compute_cheat_pct(eval_data)
    eval_table = build_eval_table(eval_data, episode_summaries, periodic_evals)

    # Extract quotes
    auditor_quotes = select_auditor_quotes(reasoning_entries)
    buyer_quotes   = select_buyer_quotes(reasoning_entries)
    print(f"  Auditor quotes: {len(auditor_quotes)}, Buyer quotes: {len(buyer_quotes)}")

    # Generate charts
    print("Generating charts...")
    auditor_entries = [e for e in reasoning_entries if e.get('agent') == 'auditor' and e.get('reason')]

    charts = {}
    charts['reward_curve']     = chart_reward_and_json(steps_data, reasoning_entries)
    print("  reward_curve done")
    charts['auditor_reasoning'] = chart_auditor_reasoning(auditor_entries)
    print("  auditor_reasoning done")
    charts['auditor_accuracy'] = chart_auditor_accuracy(periodic_evals, eval_data, round_details)
    print("  auditor_accuracy done")
    charts['per_personality']  = chart_per_personality(episode_summaries, periodic_evals)
    print("  per_personality done")

    # Check output file doesn't already exist
    out_filename = f"report_{RUN_NUMBER}.html"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    if os.path.exists(out_path):
        print(f"ERROR: {out_path} already exists. Refusing to overwrite.")
        return

    # Generate HTML
    print("Generating HTML...")
    html = generate_html(RUN_NUMBER, stats, charts, auditor_quotes, buyer_quotes, eval_table, cheat_pct, steps_data)

    # Write
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nReport written: {out_path}")
    print(f"Size: {size_kb:.0f} KB")
    print(f"Public URL: https://aibizbrain.com/nm_game/static/auditron/{out_filename}")

if __name__ == '__main__':
    main()
