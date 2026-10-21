import numpy as np
import matplotlib.pyplot as plt

from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, fixed_split_rule, wfq_static_rule
from ddpg_agent import DDPGAgent

def run_baseline_episode(policy_name, env, seed=42):
    np.random.seed(seed)
    env.reset()
    total_reward = 0
    done = False

    while not done:
        if policy_name == "cmu":
            action = cmu_rule(env.queues, env.holding_costs)
        elif policy_name == "lcq":
            action = lcq_rule(env.queues)
        elif policy_name == "wfq":
            action = wfq_static_rule(env.n_queues)
        else:
            action = fixed_split_rule(env.n_queues)

        _, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward


# STEP 1: Measure baseline rewards
print("=" * 60)
print("  Measuring 8-Queue Multi-Router baseline rewards...")
print("=" * 60)

N_BASELINE_EPISODES = 5
cmu_rewards   = [run_baseline_episode("cmu",   QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]
lcq_rewards   = [run_baseline_episode("lcq",   QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]
wfq_rewards   = [run_baseline_episode("wfq",   QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]
fixed_rewards = [run_baseline_episode("fixed", QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]

avg_cmu   = np.mean(cmu_rewards)
avg_lcq   = np.mean(lcq_rewards)
avg_wfq   = np.mean(wfq_rewards)
avg_fixed = np.mean(fixed_rewards)

print(f"  c-μ Rule:             {avg_cmu:.1f}")
print(f"  LCQ Rule (benchmark): {avg_lcq:.1f}")
print(f"  Static WFQ:           {avg_wfq:.1f}")
print(f"  Fixed 1/8 split:      {avg_fixed:.1f}")
print()


# STEP 2: Train the DDPG agent
print("=" * 60)
print("  Training DDPG agent (300 episodes, 8 queues)...")
print("=" * 60)

NUM_EPISODES    = 300
EPISODE_LENGTH  = 200
NOISE_DECAY     = 0.995
MIN_NOISE       = 0.05

agent = DDPGAgent(state_dim=8, action_dim=8, hidden=128)
if agent.load("ddpg_checkpoint.pth"):
    print("  ✓ Loaded existing checkpoint from ddpg_checkpoint.pth (continuing training)...")
else:
    print("  ○ Training DDPG from scratch...")
env   = QueueEnv(n_queues=8, episode_length=EPISODE_LENGTH)

episode_rewards = []
noise_scale     = 1.0

for episode in range(NUM_EPISODES):
    state = env.reset()
    agent.noise.reset()
    total_reward = 0

    for t in range(EPISODE_LENGTH):
        action = agent.select_action(state, add_noise=True, noise_scale=noise_scale)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, float(done))
        agent.learn()

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    noise_scale = max(MIN_NOISE, noise_scale * NOISE_DECAY)

    if (episode + 1) % 25 == 0:
        recent_avg = np.mean(episode_rewards[-25:])
        print(f"  Episode {episode+1:3d}/{NUM_EPISODES} | "
              f"Last-25 avg reward: {recent_avg:8.1f} | "
              f"Noise scale: {noise_scale:.3f}")

agent.save("ddpg_checkpoint.pth")
print("  ✓ Saved updated DDPG weights to ddpg_checkpoint.pth")


# STEP 3: Evaluate the trained agent
print()
print("  Evaluating trained DDPG agent (deterministic policy)...")

eval_rewards = []
for seed in range(5):
    np.random.seed(seed)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, add_noise=False)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    eval_rewards.append(total_reward)

avg_eval = np.mean(eval_rewards)
print(f"  Trained DDPG final avg reward: {avg_eval:.1f}")
print(f"  LCQ benchmark reward:          {avg_lcq:.1f}")
improvement = ((avg_eval - avg_lcq) / abs(avg_lcq)) * 100
print(f"  Relative to LCQ:               {improvement:+.1f}%")
print()


# STEP 4: Plot — Reward vs Episode
def rolling_average(data, window=20):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

smoothed = rolling_average(episode_rewards, window=20)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(episode_rewards, color="#aed6f1", linewidth=0.7, alpha=0.7, label="Episode reward (raw)")
ax.plot(smoothed, color="#2980b9", linewidth=2.0, label="Smoothed reward (20-ep avg)")

ax.axhline(avg_lcq,   color="#2ecc71", linestyle="--", linewidth=1.8, label=f"LCQ Rule: {avg_lcq:.0f}")
ax.axhline(avg_wfq,   color="#f39c12", linestyle="--", linewidth=1.5, label=f"Static WFQ: {avg_wfq:.0f}")
ax.axhline(avg_cmu,   color="#e74c3c", linestyle="--", linewidth=1.2, label=f"c-μ Rule: {avg_cmu:.0f}")

ax.set_title("DDPG Learning Curve (8-Queue Multi-Router Network)", fontsize=13, fontweight="bold")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Total Episode Reward (Negative Cost)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
print("  Plot saved to: reward_curve.png")
print("Step 2 complete.")