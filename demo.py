# demo.py
# ============================================================
# THE COMPLETE SIDE-BY-SIDE DEMO
# Runs all baselines + trains DDPG + produces final 3-panel plot
#
# HOW TO RUN:
#   python demo.py
#
# WHAT TO EXPECT:
#   ~1-2 minutes of training, then a 3-panel figure showing:
#     (a) DDPG learning curve with reference lines
#     (b) Queue 0 (expensive) trajectories: all 4 policies
#     (c) Bar chart of total cost per episode (lower = better)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, fixed_split_rule
from ddpg_agent import DDPGAgent


# ─────────────────────────────────────────────────────────────
# SETTINGS  (tweak these if you want longer training, etc.)
# ─────────────────────────────────────────────────────────────
SEED            = 42
EPISODE_LENGTH  = 200
NUM_EPISODES    = 300    # training episodes
NOISE_DECAY     = 0.997
MIN_NOISE       = 0.05
AVG_SEEDS       = 5      # how many seeds to average baselines over


# ─────────────────────────────────────────────────────────────
# PHASE 1: Run baselines
# ─────────────────────────────────────────────────────────────
def run_one_baseline(policy_name, seed=42):
    """Run one episode with a hand-crafted policy. Returns (q_history, reward, drops)."""
    np.random.seed(seed)
    env = QueueEnv(episode_length=EPISODE_LENGTH)
    env.reset()
    q_history = []
    total_reward = 0
    done = False

    while not done:
        if policy_name == "cmu":
            action = cmu_rule(env.queues, env.holding_costs)
        elif policy_name == "lcq":
            action = lcq_rule(env.queues)
        else:  # fixed
            action = fixed_split_rule(env.n_queues)

        _, reward, done, info = env.step(action)
        q_history.append(info["queue_lengths"].copy())
        total_reward += reward

    return np.array(q_history), total_reward, env.total_drops


print("=" * 55)
print("  PHASE 1: Running baselines...")
print("=" * 55)

# Single seed run → for queue trajectory plots
cmu_q,   cmu_r,   cmu_drops   = run_one_baseline("cmu",   seed=SEED)
lcq_q,   lcq_r,   lcq_drops   = run_one_baseline("lcq",   seed=SEED)
fixed_q, fixed_r, fixed_drops = run_one_baseline("fixed", seed=SEED)

# Multi-seed average → for fair reference lines and bar chart
cmu_avg   = np.mean([run_one_baseline("cmu",   seed=i)[1] for i in range(AVG_SEEDS)])
lcq_avg   = np.mean([run_one_baseline("lcq",   seed=i)[1] for i in range(AVG_SEEDS)])
fixed_avg = np.mean([run_one_baseline("fixed", seed=i)[1] for i in range(AVG_SEEDS)])

print(f"  c-μ Rule avg reward   : {cmu_avg:.1f}")
print(f"  LCQ Rule avg reward   : {lcq_avg:.1f}")
print(f"  Fixed 50/50 avg reward: {fixed_avg:.1f}")


# ─────────────────────────────────────────────────────────────
# PHASE 2: Train DDPG agent
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  PHASE 2: Training DDPG ({NUM_EPISODES} episodes)...")
print("=" * 55)

agent = DDPGAgent()
env   = QueueEnv(episode_length=EPISODE_LENGTH)
episode_rewards = []
noise_scale = 1.0

for ep in range(NUM_EPISODES):
    state = env.reset()
    agent.noise.reset()
    total_reward = 0

    for t in range(EPISODE_LENGTH):
        action     = agent.select_action(state, add_noise=True, noise_scale=noise_scale)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, float(done))
        agent.learn()
        state        = next_state
        total_reward += reward
        if done:
            break

    episode_rewards.append(total_reward)
    noise_scale = max(MIN_NOISE, noise_scale * NOISE_DECAY)

    if (ep + 1) % 50 == 0:
        recent = np.mean(episode_rewards[-20:])
        print(f"  Ep {ep+1:3d}/{NUM_EPISODES}  | "
              f" Last-20 avg: {recent:7.1f}  | "
              f" Noise: {noise_scale:.3f}")


# ─────────────────────────────────────────────────────────────
# PHASE 3: Evaluate trained agent (no noise = pure learned policy)
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  PHASE 3: Evaluating trained agent...")
print("=" * 55)

# Single seed → for trajectory plot
np.random.seed(SEED)
state = env.reset()
ddpg_q = []
ddpg_r = 0
done   = False

while not done:
    action = agent.select_action(state, add_noise=False)  # no noise = exploit
    state, reward, done, info = env.step(action)
    ddpg_q.append(info["queue_lengths"].copy())
    ddpg_r += reward

ddpg_q     = np.array(ddpg_q)
ddpg_drops = env.total_drops

# Multi-seed average → for bar chart
ddpg_avg_rewards = []
for seed in range(AVG_SEEDS):
    np.random.seed(seed)
    state = env.reset()
    ep_r  = 0
    done  = False
    while not done:
        action       = agent.select_action(state, add_noise=False)
        state, r, done, _ = env.step(action)
        ep_r += r
    ddpg_avg_rewards.append(ep_r)

ddpg_avg = np.mean(ddpg_avg_rewards)
gap_pct  = abs((ddpg_avg - cmu_avg) / cmu_avg) * 100

print(f"  Trained DDPG avg reward: {ddpg_avg:.1f}")
print(f"  c-μ target:              {cmu_avg:.1f}")
print(f"  Gap from optimal:        {gap_pct:.1f}%")


# ─────────────────────────────────────────────────────────────
# PHASE 4: Build the 3-panel figure
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  PHASE 4: Generating final comparison plot...")
print("=" * 55)

def rolling_avg(data, w=20):
    """Smooth a noisy list by computing a rolling average."""
    return [np.mean(data[max(0, i - w):i + 1]) for i in range(len(data))]

# Convert rewards to costs (positive = easier to compare in bar chart)
# cost = -reward  →  lower cost = better policy
cmu_cost   = -cmu_avg
lcq_cost   = -lcq_avg
fixed_cost = -fixed_avg
ddpg_cost  = -ddpg_avg

fig = plt.figure(figsize=(17, 10))
fig.suptitle(
    "WFQ-DRL Demo — DDPG Agent vs. Baseline Policies",
    fontsize=15, fontweight="bold", y=0.99
)

# 2-row grid: top row = full-width; bottom row = 2/3 + 1/3
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)


# ── Panel (a): Learning Curve (top, spans all 3 columns) ─────────────
ax_a = fig.add_subplot(gs[0, :])

smoothed = rolling_avg(episode_rewards, w=20)

ax_a.plot(episode_rewards,
          color="#aed6f1", linewidth=0.8, alpha=0.5,
          label="Episode reward (raw)")
ax_a.plot(smoothed,
          color="#2980b9", linewidth=2.3,
          label="Smoothed reward (20-ep avg)")
ax_a.axhline(cmu_avg,   color="#27ae60", linestyle="--", linewidth=1.8,
             label=f"c-μ Rule (optimal): {cmu_avg:.0f}")
ax_a.axhline(lcq_avg,   color="#f39c12", linestyle="--", linewidth=1.3,
             label=f"LCQ Rule: {lcq_avg:.0f}")
ax_a.axhline(fixed_avg, color="#e74c3c", linestyle="--", linewidth=1.3,
             label=f"Fixed 50/50 (floor): {fixed_avg:.0f}")

ax_a.set_title("(a)  DDPG Learning Curve — Reward climbs toward c-μ optimum",
               fontsize=11, fontweight="bold", loc="left")
ax_a.set_xlabel("Episode", fontsize=10)
ax_a.set_ylabel("Total Episode Reward", fontsize=10)
ax_a.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
ax_a.grid(True, alpha=0.3)


# ── Panel (b): Queue 0 trajectories (bottom-left, spans 2 columns) ───
ax_b = fig.add_subplot(gs[1, :2])

timesteps = np.arange(EPISODE_LENGTH)

# Queue 0 = expensive queue (h=3) — the one that matters most for cost
ax_b.plot(timesteps, cmu_q[:, 0],
          color="#27ae60", linewidth=1.6, alpha=0.9,
          label="c-μ  — Q0 expensive (h=3)")
ax_b.plot(timesteps, lcq_q[:, 0],
          color="#f39c12", linewidth=1.6, alpha=0.9,
          label="LCQ  — Q0 expensive (h=3)")
ax_b.plot(timesteps, fixed_q[:, 0],
          color="#e74c3c", linewidth=1.6, alpha=0.9,
          label="Fixed — Q0 expensive (h=3)")
ax_b.plot(timesteps, ddpg_q[:, 0],
          color="#2980b9", linewidth=2.2, alpha=1.0,
          label="DDPG — Q0 expensive (h=3)")

# Dotted lines for Queue 1 (cheap, h=1) — DDPG vs Fixed for contrast
ax_b.plot(timesteps, fixed_q[:, 1],
          color="#e74c3c", linewidth=1.0, linestyle=":",
          alpha=0.5, label="Fixed — Q1 cheap (h=1)")
ax_b.plot(timesteps, ddpg_q[:, 1],
          color="#2980b9", linewidth=1.0, linestyle=":",
          alpha=0.5, label="DDPG — Q1 cheap (h=1)")

ax_b.set_title("(b)  Queue Lengths Over Time — DDPG vs Baselines\n"
               "Solid = expensive Queue 0, Dotted = cheap Queue 1",
               fontsize=11, fontweight="bold", loc="left")
ax_b.set_xlabel("Timestep", fontsize=10)
ax_b.set_ylabel("Queue Length (packets)", fontsize=10)
ax_b.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9)
ax_b.grid(True, alpha=0.3)


# ── Panel (c): Cost Bar Chart (bottom-right) ──────────────────────────
ax_c = fig.add_subplot(gs[1, 2])

policies   = ["c-μ\nRule", "LCQ", "Fixed\n50/50", "DDPG\n(trained)"]
costs      = [cmu_cost, lcq_cost, fixed_cost, ddpg_cost]
bar_colors = ["#27ae60", "#f39c12", "#e74c3c", "#2980b9"]

bars = ax_c.bar(policies, costs, color=bar_colors, width=0.55,
                edgecolor="white", linewidth=1.5)

# Label each bar with its value
for bar, val in zip(bars, costs):
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(costs) * 0.01,
        f"{val:.0f}",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold"
    )

# Highlight the DDPG bar with a border to draw attention
bars[3].set_edgecolor("#1a252f")
bars[3].set_linewidth(2.5)

ax_c.set_title("(c)  Total Episode Cost\n(lower = better policy)",
               fontsize=11, fontweight="bold", loc="left")
ax_c.set_ylabel("Total Holding Cost  (= −reward)", fontsize=9)
ax_c.set_ylim(0, max(costs) * 1.22)
ax_c.grid(True, axis="y", alpha=0.3)

# Annotate how close DDPG got to optimal
ax_c.annotate(
    f"DDPG is only\n{gap_pct:.1f}% from\noptimal!",
    xy=(3, ddpg_cost),
    xytext=(2.2, ddpg_cost + max(costs) * 0.18),
    fontsize=8, color="#2980b9", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.5)
)


# ── Save and show ─────────────────────────────────────────────────────
plt.savefig("final_demo.png", dpi=150, bbox_inches="tight")
print("  Plot saved to: final_demo.png")
plt.show()

print()
print("=" * 55)
print("  ✅  All 3 steps complete!")
print(f"  DDPG final avg reward : {ddpg_avg:.1f}")
print(f"  c-μ Rule (optimal)    : {cmu_avg:.1f}")
print(f"  Gap from optimal      : {gap_pct:.1f}%")
print("=" * 55)