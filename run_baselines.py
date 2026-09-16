"""
run_baselines.py  —  Run all 3 baseline policies and plot the results
======================================================================

This script:
  1. Creates the same 2-queue environment for each policy (same random seed).
  2. Runs 200 timesteps under each policy.
  3. Plots queue lengths over time (so you can see how each policy manages traffic).
  4. Prints cumulative reward for each policy.

HOW TO RUN:
    cd to the wfq-drl-demo folder, then:
        python run_baselines.py

WHAT TO EXPECT:
    - A matplotlib window with 3 subplots (one per policy).
    - Each subplot shows Queue 0 (expensive) and Queue 1 (cheap) over time.
    - The c-μ rule should keep the expensive queue (Q0) very short.
    - Fixed split should let both queues grow more.
    - Terminal output showing cumulative rewards — c-μ should be the best (least negative).
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our own files
from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, fixed_split_rule


def run_episode(env, policy_fn, seed=42):
    """
    Run one full episode using a given policy function.

    Args:
        env:        a QueueEnv instance
        policy_fn:  a function that takes (env) and returns an action [w1, w2]
        seed:       random seed for reproducibility

    Returns:
        q_history:  list of queue-length snapshots (one per timestep)
        rewards:    list of rewards (one per timestep)
        total_drops: total packets dropped during the episode
    """
    np.random.seed(seed)
    state = env.reset()

    q_history = []   # will store [[q1, q2], [q1, q2], ...]
    rewards = []

    done = False
    while not done:
        # --- Ask the policy what to do ---
        action = policy_fn(env)

        # --- Take one step in the environment ---
        state, reward, done, info = env.step(action)

        # --- Record what happened ---
        q_history.append(info["queue_lengths"].copy())
        rewards.append(reward)

    return q_history, rewards, env.total_drops


# ──────────────────────────────────────────────────────────────────────
# Define wrapper functions that match the signature run_episode expects
# (each takes the env and extracts what it needs)
# ──────────────────────────────────────────────────────────────────────

def policy_cmu(env):
    """c-μ rule: needs actual queue lengths and holding costs."""
    return cmu_rule(env.queues, env.holding_costs)

def policy_lcq(env):
    """LCQ rule: needs actual queue lengths."""
    return lcq_rule(env.queues)

def policy_fixed(env):
    """Fixed 50/50 split: ignores everything."""
    return fixed_split_rule(env.n_queues)


# ──────────────────────────────────────────────────────────────────────
# MAIN: run all three and plot
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SEED = 42
    EPISODE_LEN = 200

    # --- Policy definitions ---
    policies = {
        "c-μ Rule (Optimal)":    policy_cmu,
        "LCQ (Longest Queue)":   policy_lcq,
        "Fixed 50/50 Split":     policy_fixed,
    }

    results = {}

    # --- Run each policy ---
    for name, policy_fn in policies.items():
        env = QueueEnv(episode_length=EPISODE_LEN)
        q_hist, rews, drops = run_episode(env, policy_fn, seed=SEED)
        results[name] = {
            "q_history": np.array(q_hist),      # shape: (200, 2)
            "rewards":   np.array(rews),         # shape: (200,)
            "drops":     drops,
            "total_reward": sum(rews),
        }

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  BASELINE RESULTS  (lower cost = better)")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:30s}  Reward: {r['total_reward']:8.1f}   Drops: {r['drops']:.0f}")
    print("=" * 60 + "\n")

    # ──────────────────────────────────────────────────────────────────
    # PLOT: Queue lengths over time for each policy
    # ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    fig.suptitle("Queue Lengths Over Time — Baseline Policies", fontsize=14, fontweight="bold")

    colors = {"c-μ Rule (Optimal)": "#2ecc71",
              "LCQ (Longest Queue)": "#3498db",
              "Fixed 50/50 Split":   "#e74c3c"}

    for ax, (name, r) in zip(axes, results.items()):
        q = r["q_history"]
        timesteps = np.arange(len(q))

        ax.plot(timesteps, q[:, 0], label="Queue 0 (h=3, expensive)", linewidth=1.5, color="#e74c3c", alpha=0.8)
        ax.plot(timesteps, q[:, 1], label="Queue 1 (h=1, cheap)",     linewidth=1.5, color="#3498db", alpha=0.8)

        ax.set_title(f"{name}\nReward: {r['total_reward']:.0f}", fontsize=11)
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Queue Length (packets)")
    plt.tight_layout()

    # Save the plot as a file (so you can see it even without a display)
    plt.savefig("baseline_comparison.png", dpi=150, bbox_inches="tight")
    print("📊 Plot saved to: baseline_comparison.png")

    plt.show()
    print("✅ Step 1 complete! Check the plot.")
