

import numpy as np
import matplotlib.pyplot as plt

from queue_env  import QueueEnv
from baselines  import cmu_rule, lcq_rule, fixed_split_rule
from ddpg_agent import DDPGAgent
from sac_agent  import SACAgent

EPISODE_LENGTH = 200    # timesteps per episode
NUM_EPISODES   = 250    # training episodes (fast but enough to converge)
EVAL_SEEDS     = 5      # seeds to average over during evaluation
SMOOTH_WINDOW  = 20     # rolling average window for the plot


def rolling_avg(data, w=SMOOTH_WINDOW):
    return [np.mean(data[max(0, i - w):i + 1]) for i in range(len(data))]
def run_baseline_episode(policy_name, seed=0):
    """Returns (total_reward, avg_queue_lengths[2])"""
    np.random.seed(seed)
    env = QueueEnv(episode_length=EPISODE_LENGTH)
    env.reset()
    total_reward = 0.0
    total_q      = np.zeros(2)
    done         = False
    steps        = 0

    while not done:
        if policy_name == "cmu":
            action = cmu_rule(env.queues, env.holding_costs)
        else:
            action = lcq_rule(env.queues)

        _, reward, done, info = env.step(action)
        total_reward += reward
        total_q      += info["queue_lengths"]
        steps        += 1

    return total_reward, total_q / max(steps, 1)

def train_ddpg(agent):
    env         = QueueEnv(episode_length=EPISODE_LENGTH)
    rewards     = []
    noise_scale = 1.0
    NOISE_DECAY = 0.997
    MIN_NOISE   = 0.05

    for ep in range(NUM_EPISODES):
        state = env.reset()
        agent.noise.reset()
        ep_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            action                   = agent.select_action(state, add_noise=True,
                                                           noise_scale=noise_scale)
            next_state, r, done, _   = env.step(action)
            agent.store(state, action, r, next_state, float(done))
            agent.learn()
            state, ep_reward         = next_state, ep_reward + r
            if done:
                break

        rewards.append(ep_reward)
        noise_scale = max(MIN_NOISE, noise_scale * NOISE_DECAY)

        if (ep + 1) % 50 == 0:
            print(f"    [DDPG] Ep {ep+1:3d}/{NUM_EPISODES} | "
                  f"Last-{SMOOTH_WINDOW} avg: {np.mean(rewards[-SMOOTH_WINDOW:]):.1f} | "
                  f"Noise: {noise_scale:.3f}")

    return rewards


def train_sac(agent):
    # SAC is naturally stochastic — no external noise needed!
    env     = QueueEnv(episode_length=EPISODE_LENGTH)
    rewards = []

    for ep in range(NUM_EPISODES):
        state     = env.reset()
        ep_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            action                 = agent.select_action(state, deterministic=False)
            next_state, r, done, _ = env.step(action)
            agent.store(state, action, r, next_state, float(done))
            agent.learn()
            state, ep_reward       = next_state, ep_reward + r
            if done:
                break

        rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            print(f"    [SAC]  Ep {ep+1:3d}/{NUM_EPISODES} | "
                  f"Last-{SMOOTH_WINDOW} avg: {np.mean(rewards[-SMOOTH_WINDOW:]):.1f} | "
                  f"Alpha: {agent.alpha:.4f}")

    return rewards


def evaluate_agent(agent, agent_type="ddpg"):
    all_rewards = []
    all_q       = []
    env         = QueueEnv(episode_length=EPISODE_LENGTH)

    for seed in range(EVAL_SEEDS):
        np.random.seed(seed)
        state     = env.reset()
        ep_reward = 0.0
        total_q   = np.zeros(2)
        done      = False
        steps     = 0

        while not done:
            if agent_type == "ddpg":
                action = agent.select_action(state, add_noise=False)
            else:
                action = agent.select_action(state, deterministic=True)

            state, r, done, info = env.step(action)
            ep_reward += r
            total_q   += info["queue_lengths"]
            steps     += 1

        all_rewards.append(ep_reward)
        all_q.append(total_q / max(steps, 1))

    return np.mean(all_rewards), np.mean(all_q, axis=0)

print("=" * 60)
print("  PHASE 1: Measuring baselines...")
print("=" * 60)

cmu_results = [run_baseline_episode("cmu", seed=i) for i in range(EVAL_SEEDS)]
lcq_results = [run_baseline_episode("lcq", seed=i) for i in range(EVAL_SEEDS)]

cmu_avg_r = np.mean([x[0] for x in cmu_results])
lcq_avg_r = np.mean([x[0] for x in lcq_results])
cmu_avg_q = np.mean([x[1] for x in cmu_results], axis=0)
lcq_avg_q = np.mean([x[1] for x in lcq_results], axis=0)

print(f"  c-μ Rule → avg reward: {cmu_avg_r:.1f} | "
      f"avg Q0 length: {cmu_avg_q[0]:.2f}")
print(f"  LCQ Rule → avg reward: {lcq_avg_r:.1f} | "
      f"avg Q0 length: {lcq_avg_q[0]:.2f}")
print()
print("=" * 60)
print(f"  PHASE 2: Training DDPG ({NUM_EPISODES} episodes)...")
print("=" * 60)

ddpg          = DDPGAgent()
ddpg_rewards  = train_ddpg(ddpg)
ddpg_eval_r, ddpg_eval_q = evaluate_agent(ddpg, "ddpg")
print(f"  ✓ DDPG final avg reward: {ddpg_eval_r:.1f}")

print()
print("=" * 60)
print(f"  PHASE 3: Training SAC ({NUM_EPISODES} episodes)...")
print("=" * 60)

sac          = SACAgent()
sac_rewards  = train_sac(sac)
sac_eval_r, sac_eval_q = evaluate_agent(sac, "sac")
print(f"  ✓ SAC  final avg reward: {sac_eval_r:.1f}")

print()
print("=" * 60)
print("  PHASE 4: Plotting results...")
print("=" * 60)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("DDPG vs SAC — Queue Bandwidth Scheduling Demo",
             fontsize=14, fontweight="bold")

episodes = np.arange(NUM_EPISODES)

# ── Panel (a): Reward convergence ────────────────────────────
# Raw faint traces
ax_a.plot(ddpg_rewards, color="#aed6f1", linewidth=0.7, alpha=0.4)
ax_a.plot(sac_rewards,  color="#f9c8c8", linewidth=0.7, alpha=0.4)

# Smoothed curves (the trend)
ax_a.plot(rolling_avg(ddpg_rewards), color="#2980b9", linewidth=2.2,
          label=f"DDPG  (final ≈ {ddpg_eval_r:.0f})")
ax_a.plot(rolling_avg(sac_rewards),  color="#c0392b", linewidth=2.2,
          label=f"SAC   (final ≈ {sac_eval_r:.0f})")

# Reference lines
ax_a.axhline(cmu_avg_r, color="#27ae60", linestyle="--", linewidth=1.8,
             label=f"c-μ Rule (optimal): {cmu_avg_r:.0f}")
ax_a.axhline(lcq_avg_r, color="#f39c12", linestyle="--", linewidth=1.3,
             label=f"LCQ Rule: {lcq_avg_r:.0f}")

ax_a.set_title("(a)  Reward Convergence — DDPG vs SAC", fontsize=11,
               fontweight="bold", loc="left")
ax_a.set_xlabel("Episode", fontsize=10)
ax_a.set_ylabel("Total Episode Reward", fontsize=10)
ax_a.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax_a.grid(True, alpha=0.3)


labels     = ["c-μ Rule", "LCQ Rule", "DDPG", "SAC"]
q0_values  = [cmu_avg_q[0], lcq_avg_q[0], ddpg_eval_q[0], sac_eval_q[0]]
bar_colors = ["#27ae60", "#f39c12", "#2980b9", "#c0392b"]

bars = ax_b.bar(labels, q0_values, color=bar_colors, width=0.5,
                edgecolor="white", linewidth=1.5)

# Value labels on top of bars
for bar, val in zip(bars, q0_values):
    ax_b.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + max(q0_values) * 0.02,
              f"{val:.3f}",
              ha="center", va="bottom", fontsize=10, fontweight="bold")

# Highlight winner with a bold border
best_idx = int(np.argmin(q0_values))
bars[best_idx].set_edgecolor("#2c3e50")
bars[best_idx].set_linewidth(2.5)
ax_b.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
          bars[best_idx].get_height() / 2,
          "🏆", ha="center", va="center", fontsize=16)

ax_b.set_title("(b)  Avg Queue 0 Length (Expensive Queue)\n"
               "= Delay Proxy via Little's Law — lower is better",
               fontsize=11, fontweight="bold", loc="left")
ax_b.set_ylabel("Avg Queue 0 Length (packets)", fontsize=10)
ax_b.set_ylim(0, max(q0_values) * 1.30)
ax_b.grid(True, axis="y", alpha=0.3)


# ── Summary printout ──────────────────────────────────────────
print()
print("=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Policy':<18} {'Avg Reward':>12} {'Avg Q0 Length':>15}")
print(f"  {'-'*48}")
print(f"  {'c-μ Rule':<18} {cmu_avg_r:>12.1f} {cmu_avg_q[0]:>15.3f}")
print(f"  {'LCQ Rule':<18} {lcq_avg_r:>12.1f} {lcq_avg_q[0]:>15.3f}")
print(f"  {'DDPG':<18} {ddpg_eval_r:>12.1f} {ddpg_eval_q[0]:>15.3f}")
print(f"  {'SAC':<18} {sac_eval_r:>12.1f} {sac_eval_q[0]:>15.3f}")
print("=" * 60)


plt.tight_layout()
plt.savefig("ddpg_vs_sac.png", dpi=150, bbox_inches="tight")
print("  Saved: ddpg_vs_sac.png")
plt.show()
print("\n✅ Comparison complete!")