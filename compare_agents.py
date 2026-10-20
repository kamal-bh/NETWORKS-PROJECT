import numpy as np
import matplotlib.pyplot as plt

from queue_env  import QueueEnv
from baselines  import cmu_rule, lcq_rule, fixed_split_rule, wfq_static_rule
from ddpg_agent import DDPGAgent
from sac_agent  import SACAgent

EPISODE_LENGTH = 200    # timesteps per episode
NUM_EPISODES   = 250    # training episodes
EVAL_SEEDS     = 5      # seeds to average over during evaluation
SMOOTH_WINDOW  = 20     # rolling average window for the plot
N_QUEUES       = 8


def rolling_avg(data, w=SMOOTH_WINDOW):
    return [np.mean(data[max(0, i - w):i + 1]) for i in range(len(data))]


def run_baseline_episode(policy_name, seed=0):
    """Returns (total_reward, avg_queue_lengths[8], total_drops)"""
    np.random.seed(seed)
    env = QueueEnv(n_queues=N_QUEUES, episode_length=EPISODE_LENGTH)
    env.reset()
    total_reward = 0.0
    total_q      = np.zeros(N_QUEUES)
    done         = False
    steps        = 0

    while not done:
        if policy_name == "cmu":
            action = cmu_rule(env.queues, env.holding_costs)
        elif policy_name == "lcq":
            action = lcq_rule(env.queues)
        elif policy_name == "wfq":
            action = wfq_static_rule(env.n_queues)
        else:
            action = fixed_split_rule(env.n_queues)

        _, reward, done, info = env.step(action)
        total_reward += reward
        total_q      += info["queue_lengths"]
        steps        += 1

    return total_reward, total_q / max(steps, 1), env.total_drops


def train_ddpg(agent):
    env         = QueueEnv(n_queues=N_QUEUES, episode_length=EPISODE_LENGTH)
    rewards     = []
    noise_scale = 1.0
    NOISE_DECAY = 0.995
    MIN_NOISE   = 0.05

    for ep in range(NUM_EPISODES):
        state = env.reset()
        agent.noise.reset()
        ep_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            action = agent.select_action(state, add_noise=True, noise_scale=noise_scale)
            next_state, r, done, _ = env.step(action)
            agent.store(state, action, r, next_state, float(done))
            agent.learn()
            state, ep_reward = next_state, ep_reward + r
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
    env     = QueueEnv(n_queues=N_QUEUES, episode_length=EPISODE_LENGTH)
    rewards = []

    for ep in range(NUM_EPISODES):
        state     = env.reset()
        ep_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            action = agent.select_action(state, deterministic=False)
            next_state, r, done, _ = env.step(action)
            agent.store(state, action, r, next_state, float(done))
            agent.learn()
            state, ep_reward = next_state, ep_reward + r
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
    all_drops   = []
    env         = QueueEnv(n_queues=N_QUEUES, episode_length=EPISODE_LENGTH)

    for seed in range(EVAL_SEEDS):
        np.random.seed(seed)
        state     = env.reset()
        ep_reward = 0.0
        total_q   = np.zeros(N_QUEUES)
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
        all_drops.append(env.total_drops)

    return np.mean(all_rewards), np.mean(all_q, axis=0), np.mean(all_drops)


print("=" * 65)
print("  PHASE 1: Measuring 8-Queue Multi-Router Baselines...")
print("=" * 65)

cmu_results = [run_baseline_episode("cmu", seed=i) for i in range(EVAL_SEEDS)]
lcq_results = [run_baseline_episode("lcq", seed=i) for i in range(EVAL_SEEDS)]
wfq_results = [run_baseline_episode("wfq", seed=i) for i in range(EVAL_SEEDS)]

cmu_avg_r = np.mean([x[0] for x in cmu_results])
lcq_avg_r = np.mean([x[0] for x in lcq_results])
wfq_avg_r = np.mean([x[0] for x in wfq_results])

cmu_avg_q = np.mean([x[1] for x in cmu_results], axis=0)
lcq_avg_q = np.mean([x[1] for x in lcq_results], axis=0)
wfq_avg_q = np.mean([x[1] for x in wfq_results], axis=0)

cmu_avg_d = np.mean([x[2] for x in cmu_results])
lcq_avg_d = np.mean([x[2] for x in lcq_results])
wfq_avg_d = np.mean([x[2] for x in wfq_results])

print(f"  c-μ Rule   → avg reward: {cmu_avg_r:9.1f} | drops: {cmu_avg_d:.0f} | Q0: {cmu_avg_q[0]:.2f}")
print(f"  LCQ Rule   → avg reward: {lcq_avg_r:9.1f} | drops: {lcq_avg_d:.0f} | Q0: {lcq_avg_q[0]:.2f}")
print(f"  Static WFQ → avg reward: {wfq_avg_r:9.1f} | drops: {wfq_avg_d:.0f} | Q0: {wfq_avg_q[0]:.2f}")
print()

print("=" * 65)
print(f"  PHASE 2: Training DDPG Agent ({NUM_EPISODES} episodes, 8 queues)...")
print("=" * 65)
ddpg = DDPGAgent(state_dim=N_QUEUES, action_dim=N_QUEUES, hidden=128)
ddpg_rewards = train_ddpg(ddpg)
ddpg_eval_r, ddpg_eval_q, ddpg_eval_d = evaluate_agent(ddpg, "ddpg")
print(f"  ✓ DDPG final avg reward: {ddpg_eval_r:.1f} | drops: {ddpg_eval_d:.0f}")
print()

print("=" * 65)
print(f"  PHASE 3: Training SAC Agent ({NUM_EPISODES} episodes, 8 queues)...")
print("=" * 65)
sac = SACAgent(state_dim=N_QUEUES, action_dim=N_QUEUES, hidden=128)
sac_rewards = train_sac(sac)
sac_eval_r, sac_eval_q, sac_eval_d = evaluate_agent(sac, "sac")
print(f"  ✓ SAC final avg reward: {sac_eval_r:.1f} | drops: {sac_eval_d:.0f}")
print()

print("=" * 65)
print("  PHASE 4: Plotting Comparative Evaluation...")
print("=" * 65)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("DDPG vs SAC in 8-Queue Multi-Router Environment", fontsize=14, fontweight="bold")

# Panel (a): Reward convergence
ax_a.plot(ddpg_rewards, color="#aed6f1", linewidth=0.7, alpha=0.35)
ax_a.plot(sac_rewards,  color="#f9c8c8", linewidth=0.7, alpha=0.35)

ax_a.plot(rolling_avg(ddpg_rewards), color="#2980b9", linewidth=2.2,
          label=f"DDPG (eval: {ddpg_eval_r:.0f})")
ax_a.plot(rolling_avg(sac_rewards),  color="#c0392b", linewidth=2.2,
          label=f"SAC  (eval: {sac_eval_r:.0f})")

ax_a.axhline(lcq_avg_r, color="#27ae60", linestyle="--", linewidth=1.6, label=f"LCQ Rule: {lcq_avg_r:.0f}")
ax_a.axhline(wfq_avg_r, color="#f39c12", linestyle="--", linewidth=1.4, label=f"Static WFQ: {wfq_avg_r:.0f}")
ax_a.axhline(cmu_avg_r, color="#7f8c8d", linestyle=":",  linewidth=1.2, label=f"c-μ Rule: {cmu_avg_r:.0f}")

ax_a.set_title("(a) Reward Convergence Across Training Episodes", fontsize=11, fontweight="bold", loc="left")
ax_a.set_xlabel("Episode", fontsize=10)
ax_a.set_ylabel("Total Episode Reward (Negative Cost)", fontsize=10)
ax_a.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax_a.grid(True, alpha=0.3)

# Panel (b): High-priority Q0 Delay Proxy + Average Backlog
policies = ["c-μ Rule", "LCQ Rule", "Static WFQ", "DDPG", "SAC"]
high_priority_q = [cmu_avg_q[0], lcq_avg_q[0], wfq_avg_q[0], ddpg_eval_q[0], sac_eval_q[0]]
total_backlog   = [cmu_avg_q.sum(), lcq_avg_q.sum(), wfq_avg_q.sum(), ddpg_eval_q.sum(), sac_eval_q.sum()]

x = np.arange(len(policies))
width = 0.35

rects1 = ax_b.bar(x - width/2, high_priority_q, width, label='Q0 (Net Control) Avg Queue', color="#e74c3c", alpha=0.85)
rects2 = ax_b.bar(x + width/2, total_backlog, width, label='Total 8-Queue Avg Backlog', color="#3498db", alpha=0.85)

ax_b.set_title("(b) Queue Backlog Comparison (Lower = Lower Latency)", fontsize=11, fontweight="bold", loc="left")
ax_b.set_xticks(x)
ax_b.set_xticklabels(policies, fontsize=10)
ax_b.set_ylabel("Packets in Buffer", fontsize=10)
ax_b.legend(fontsize=9)
ax_b.grid(True, axis="y", alpha=0.3)

# Add text above bars
for r in rects1:
    h = r.get_height()
    ax_b.text(r.get_x() + r.get_width()/2., h + 0.1, f'{h:.2f}', ha='center', va='bottom', fontsize=8)

for r in rects2:
    h = r.get_height()
    ax_b.text(r.get_x() + r.get_width()/2., h + 0.1, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

print()
print("=" * 65)
print("  FINAL BENCHMARK COMPARISON")
print("=" * 65)
print(f"  {'Policy':<14} | {'Avg Reward':>11} | {'Drops':>6} | {'Avg Q0':>8} | {'Total Q':>8}")
print(f"  {'-'*57}")
print(f"  {'c-μ Rule':<14} | {cmu_avg_r:>11.1f} | {cmu_avg_d:>6.0f} | {cmu_avg_q[0]:>8.2f} | {cmu_avg_q.sum():>8.2f}")
print(f"  {'LCQ Rule':<14} | {lcq_avg_r:>11.1f} | {lcq_avg_d:>6.0f} | {lcq_avg_q[0]:>8.2f} | {lcq_avg_q.sum():>8.2f}")
print(f"  {'Static WFQ':<14} | {wfq_avg_r:>11.1f} | {wfq_avg_d:>6.0f} | {wfq_avg_q[0]:>8.2f} | {wfq_avg_q.sum():>8.2f}")
print(f"  {'DDPG':<14} | {ddpg_eval_r:>11.1f} | {ddpg_eval_d:>6.0f} | {ddpg_eval_q[0]:>8.2f} | {ddpg_eval_q.sum():>8.2f}")
print(f"  {'SAC':<14} | {sac_eval_r:>11.1f} | {sac_eval_d:>6.0f} | {sac_eval_q[0]:>8.2f} | {sac_eval_q.sum():>8.2f}")
print("=" * 65)

plt.tight_layout()
plt.savefig("ddpg_vs_sac.png", dpi=150, bbox_inches="tight")
print("  Saved: ddpg_vs_sac.png")
print("\nComparison complete.")