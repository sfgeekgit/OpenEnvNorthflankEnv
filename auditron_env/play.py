"""
Auditron — Interactive Play Mode

Play as all agents manually to test env logic.
Run: python3 play.py
"""

import json
from server import AuditronAction, AuditronEnv, SUPPLIER_IDS


def main():
    env = AuditronEnv()
    obs = env.reset(seed=42)
    print(f"\n{'='*60}")
    print(f"AUDITRON — Interactive Play Mode")
    print(f"{'='*60}")
    print(f"{obs.message}\n")

    while not obs.done:
        s = env.state
        rnd = s.current_round
        phase = s.phase

        if phase == 0:
            # Supplier phase
            remaining = [sid for sid in SUPPLIER_IDS if sid not in s.suppliers_submitted]
            agent_id = remaining[0]
            obs_data = env.get_supplier_obs(agent_id)
            print(f"--- Round {rnd} | {agent_id} ({obs_data['personality']}) ---")
            print(f"  Required strength: {obs_data['required_strength']}")
            print(f"  Your cost/point:   {obs_data['your_cost_per_point']:.2f}")
            print(f"  Format: {{\"bid_price\": N, \"actual_strength\": N}}")

            raw = input(f"  {agent_id} action > ").strip()
            if not raw:
                # Auto: honest bid at 20% markup
                strength = obs_data["required_strength"]
                cost = strength * obs_data["your_cost_per_point"]
                raw = json.dumps({"bid_price": round(cost * 1.2, 1), "actual_strength": strength})
                print(f"  (auto) {raw}")

            obs = env.step(AuditronAction(agent_id=agent_id, content=raw))
            print(f"  -> {obs.message}")
            if obs.reward != 0:
                print(f"     reward={obs.reward:.3f}")
            print()

        elif phase == 1:
            # Auditor phase
            obs_data = env.get_auditor_obs()
            print(f"--- Round {rnd} | AUDITOR ---")
            print(f"  Required strength: {obs_data['required_strength']}")
            print(f"  Bids: {json.dumps(obs_data['bids'], indent=2)}")
            print(f"  Format: {{\"pick\": \"supplier_N\", \"reason\": \"...\", \"flags\": [...]}}")

            raw = input("  auditor action > ").strip()
            if not raw:
                # Auto: pick cheapest
                cheapest = min(obs_data["bids"], key=obs_data["bids"].get)
                raw = json.dumps({
                    "pick": cheapest,
                    "reason": f"Cheapest bid from {cheapest}",
                    "flags": [],
                })
                print(f"  (auto) {raw}")

            obs = env.step(AuditronAction(agent_id="auditor", content=raw))
            print(f"  -> {obs.message}")
            if obs.reward != 0:
                print(f"     reward={obs.reward:.3f}")
            print()

        elif phase == 2:
            # Buyer phase
            obs_data = env.get_buyer_obs()
            print(f"--- Round {rnd} | BUYER ---")
            print(f"  Bids: {json.dumps(obs_data['bids'], indent=2)}")
            print(f"  Auditor says: pick={obs_data['auditor_recommendation'].get('pick')}")
            print(f"  Spend so far: {obs_data['total_spend_so_far']:.1f}")
            print(f"  Format: {{\"pick\": \"supplier_N\", \"reason\": \"...\"}}")

            raw = input("  buyer action > ").strip()
            if not raw:
                # Auto: follow auditor
                auditor_pick = obs_data["auditor_recommendation"].get("pick", SUPPLIER_IDS[0])
                raw = json.dumps({
                    "pick": auditor_pick,
                    "reason": "Following auditor recommendation",
                })
                print(f"  (auto) {raw}")

            obs = env.step(AuditronAction(agent_id="buyer", content=raw))
            print(f"  -> {obs.message}")
            if obs.reward != 0:
                print(f"     reward={obs.reward:.3f}")
            print()

    # Episode done — show summary
    print(f"\n{'='*60}")
    print("EPISODE COMPLETE")
    print(f"{'='*60}")
    s = env.state
    print(f"Buyer total spend:     {s.buyer_total_spend:.1f}")
    print(f"Buyer total penalties:  {s.buyer_total_penalties:.1f}")
    failures = sum(1 for p in s.bridge_parts if p["failed"])
    print(f"Parts failed:           {failures}/{len(s.bridge_parts)}")
    print(f"\nFinal rewards:")
    for sid in SUPPLIER_IDS:
        print(f"  {sid}: {s.supplier_rewards[sid]:.3f}")
    print(f"  auditor:    {s.auditor_reward:.3f}")
    print(f"  buyer:      {s.buyer_reward:.3f}")


if __name__ == "__main__":
    main()
