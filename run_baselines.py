import numpy as np
import matplotlib.pyplot as plt

from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, fixed_split_rule, wfq_static_rule, priority_queueing_rule

QUEUE_LABELS = [
    "Q0 (Net Control, h=10)",
    "Q1 (Voice EF, h=8)",
    "Q2 (Video AF4, h=6)",
    "Q3 (Stream AF3, h=4)",
    "Q4 (Crit Data AF2, h=3)",
    "Q5 (Std Data AF1, h=2)",
    "Q6 (Best Effort, h=1)",
    "Q7 (Background, h=0.5)"
]

QUEUE_COLORS = [
    "#e74c3c",  # Q0: Red
    "#e67e22",  # Q1: Orange
    "#f1c40f",  # Q2: Yellow
    "#2ecc71",  # Q3: Green
    "#1abc9c",  # Q4: Teal
    "#3498db",  # Q5: Blue
    "#9b59b6",  # Q6: Purple
    "#7f8c8d",  # Q7: Gray
]

def run_episode(env, policy_fn, seed=42):
    np.random.seed(seed)
    state = env.reset()
    q_history = []
    rewards = []

    done = False
    while not done:
        action = policy_fn(env)
        state, reward, done, info = env.step(action)
        q_history.append(info["queue_lengths"].copy())
        rewards.append(reward)

    return q_history, rewards, env.total_drops


def policy_cmu(env):
    return cmu_rule(env.queues, env.holding_costs)

def policy_pq(env):
    return priority_queueing_rule(env.queues)

def policy_wfq(env):
    return wfq_static_rule(env.n_queues)

def policy_lcq(env):
    return lcq_rule(env.queues)

def policy_fixed(env):
    return fixed_split_rule(env.n_queues)


if __name__ == "__main__":
    SEED = 42
    EPISODE_LEN = 200

    policies = {
        "c-μ Rule": policy_cmu,
        "Priority Queueing (PQ)": policy_pq,
        "Static WFQ": policy_wfq,
        "LCQ (Longest Queue)": policy_lcq,
    }

    results = {}

    for name, policy_fn in policies.items():
        env = QueueEnv(n_queues=8, episode_length=EPISODE_LEN)
        q_hist, rews, drops = run_episode(env, policy_fn, seed=SEED)
        results[name] = {
            "q_history": np.array(q_hist),
            "rewards": np.array(rews),
            "drops": drops,
            "total_reward": sum(rews),
        }

    print("\n" + "=" * 65)
    print("  8-QUEUE MULTI-ROUTER BASELINE BENCHMARKS")
    print("=" * 65)
    for name, r in results.items():
        print(f"  {name:25s} | Reward: {r['total_reward']:9.1f} | Packet Drops: {r['drops']:.0f}")
    print("=" * 65 + "\n")

    # Plot queue trajectories for each baseline
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    fig.suptitle("Queue Trajectories (8 Queues) in Multi-Router Network", fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        q = r["q_history"]
        timesteps = np.arange(len(q))

        for qi in range(8):
            ax.plot(timesteps, q[:, qi], label=QUEUE_LABELS[qi], linewidth=1.4, color=QUEUE_COLORS[qi], alpha=0.85)

        ax.set_title(f"{name} — Total Reward: {r['total_reward']:.0f} (Drops: {r['drops']:.0f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Simulation Timestep Slot", fontsize=10)
        ax.set_ylabel("Queue Length (packets)", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to: baseline_comparison.png")
    print("Baseline evaluation complete.")
