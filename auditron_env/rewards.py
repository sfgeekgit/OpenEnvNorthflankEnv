"""
Auditron Reward Constants

Every reward and penalty in the system is defined here.
No magic numbers anywhere else. Tweak these to shape agent behavior.
"""

# === PER-ROUND SUPPLIER REWARDS ===
REWARD_WIN_BID = 0.1                        # Won the auction this round
PENALTY_PART_FAILED_SUPPLIER = 0.0          # No direct penalty — supplier is punished indirectly by losing future bids
PENALTY_INVALID_FORMAT = -5.0               # Malformed JSON action (any agent) — must be steep so reasoning reward never outweighs it

# === PER-ROUND BUYER REWARDS ===
REWARD_VALID_ACTION_BUYER = 0.05            # Submitted a valid selection
REWARD_PART_SUCCESS_BUYER = 0.2             # Selected part passed
REWARD_GOOD_ADVICE_FOLLOWED = 0.0           # No direct reward — buyer benefits indirectly via fewer failures
PENALTY_IGNORED_ADVICE_FAILED = 0.0         # No direct penalty — buyer suffers indirectly via part failure cost

# === PER-ROUND AUDITOR REWARDS ===
REWARD_VALID_ACTION_AUDITOR = 0.05          # Submitted valid recommendation
REWARD_GOOD_REC_FOLLOWED_SUCCESS = 0.3      # Rec was followed and part succeeded
PENALTY_BAD_REC_FOLLOWED_FAILED = -1.5      # Rec was followed and part failed
REWARD_FLAG_CAUGHT_CHEATER = 1.0            # Flagged a supplier who caused a failure (legacy, unused)
PENALTY_FLAG_FALSE_POSITIVE = -0.5          # Flagged a supplier who never failed (legacy, unused)

# === PER-ROUND FLAG REWARDS (god-engine checks ALL suppliers, not just winner) ===
REWARD_FLAG_TRUE_POSITIVE = 1.0             # Flagged a supplier who was cheating this round
PENALTY_FLAG_FALSE_POSITIVE_ROUND = -0.5    # Flagged a supplier who was honest this round

# === PER-ROUND PICK QUALITY REWARDS (smaller — rank honest suppliers by price) ===
REWARD_PICK_BEST = 0.3                      # Auditor picked cheapest honest supplier
REWARD_PICK_SECOND = 0.2                    # Auditor picked 2nd cheapest honest
REWARD_PICK_THIRD = 0.1                     # Auditor picked 3rd cheapest honest

# === AUDITOR REASONING QUALITY (regex heuristics, no LLM judge) ===
REWARD_REASON_MENTIONS_SUPPLIER = 0.5       # Mentions "supplier_N"
REWARD_REASON_MENTIONS_PRICE = 0.5          # Mentions price/bid/cost + number
REWARD_REASON_MENTIONS_FAILURE = 0.5        # Mentions "fail" or "failed"
REWARD_REASON_MENTIONS_ROUND = 0.3          # Mentions "round N"
REWARD_REASON_MENTIONS_COMPARISON = 0.3     # Uses comparison words (cheaper, below, etc.)
REWARD_REASON_LENGTH_20 = 0.5              # Reason is >= 20 words
REWARD_REASON_LENGTH_50 = 1.0              # Reason is >= 50 words (depth bonus)

# === END-OF-EPISODE SUPPLIER REWARDS (profit ranking) ===
SUPPLIER_RANK_REWARDS = [50, 40, 25, 10]     # 1st through 4th place — large so episode outcome dominates training signal
REWARD_NEVER_CAUGHT = 0.5                   # No parts failed all episode

# === END-OF-EPISODE AUDITOR ECONOMIC REWARDS (auditor is responsible for outcomes) ===
REWARD_AUDITOR_SPEND_BELOW_MEDIAN = 6.0     # Total spend below median possible
PENALTY_AUDITOR_SPEND_ABOVE_2X = -6.0       # Total spend above 2x median possible
REWARD_AUDITOR_ZERO_FAILURES = 3.0          # Zero failures across all parts

# === END-OF-EPISODE AUDITOR REWARDS ===
REWARD_AUDITOR_TPR_MAX = 2.0                # Max bonus for true positive rate
PENALTY_AUDITOR_FPR_MAX = -1.0              # Max penalty for false positive rate
REWARD_AUDITOR_DETECTED_CHEATER = 0.5       # Per cheater correctly flagged
PENALTY_AUDITOR_FALSE_FLAG = -0.3           # Per honest supplier wrongly flagged
REWARD_AUDITOR_REASONING_MAX = 1.0          # Max reasoning quality bonus

# === ENVIRONMENT PARAMETERS ===
TOTAL_PARTS = 50                            # Rounds per episode
NUM_SUPPLIERS = 4                           # Number of supplier agents
REQUIRED_STRENGTH_MIN = 40                  # Min required strength per part
REQUIRED_STRENGTH_MAX = 100                 # Max required strength per part
COST_PER_POINT_MIN = 0.5                    # Min base cost per strength point
COST_PER_POINT_MAX = 2.0                    # Max base cost per strength point
COST_VARIATION = 0.05                       # +-5% variation for 2 of 5 suppliers
FAILURE_RATE_PER_PCT_UNDER = 0.02           # 2% failure chance per 1% under spec (lowered to make cheating more viable)
FAILURE_PENALTY_MULTIPLIER = 10             # Failure costs buyer 10x median bid
