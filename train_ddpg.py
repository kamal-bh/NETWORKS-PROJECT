import numpy as np
import matplotlib.pyplot as plt

from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, fixed_split_rule
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
        else:
            action = fixed_split_rule(env.n_queues)

        _, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward


# STEP 1: Measure baseline rewards (average over 5 episodes)

print("=" * 55)
print("  Measuring baseline rewards...")
print("=" * 55)

env = QueueEnv(episode_length=200)
N_BASELINE_EPISODES = 5

cmu_rewards   = [run_baseline_episode("cmu",   QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]
lcq_rewards   = [run_baseline_episode("lcq",   QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]
fixed_rewards = [run_baseline_episode("fixed", QueueEnv(), seed=i) for i in range(N_BASELINE_EPISODES)]

avg_cmu   = np.mean(cmu_rewards)
avg_lcq   = np.mean(lcq_rewards)
avg_fixed = np.mean(fixed_rewards)

print(f"  c-μ Rule  (target to beat): {avg_cmu:.1f}")
print(f"  LCQ Rule:                   {avg_lcq:.1f}")
print(f"  Fixed 50/50 (floor):        {avg_fixed:.1f}")
print()


# STEP 2: Train the DDPG agent

print("=" * 55)
print("  Training DDPG agent (300 episodes)...")
print("=" * 55)

NUM_EPISODES    = 300
EPISODE_LENGTH  = 200
NOISE_DECAY     = 0.997   # reduce exploration noise each episode
MIN_NOISE       = 0.05    # never go fully deterministic (a little exploration always helps)

agent = DDPGAgent()
env   = QueueEnv(episode_length=EPISODE_LENGTH)

episode_rewards = []   # track reward each episode (for the plot)
noise_scale     = 1.0  # start with full noise (lots of exploration)

for episode in range(NUM_EPISODES):

    # Reset environment and noise at start of each episode
    state = env.reset()
    agent.noise.reset()
    total_reward = 0

    for t in range(EPISODE_LENGTH):

        # Agent picks an action 
        action = agent.select_action(state, add_noise=True, noise_scale=noise_scale)

        # Environment responds 
        next_state, reward, done, _ = env.step(action)

        # Store this experience in memory 
        agent.store(state, action, reward, next_state, float(done))

        # Learn from a random batch of past experiences
        agent.learn()

        state        = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    # Decay exploration noise
    noise_scale = max(MIN_NOISE, noise_scale * NOISE_DECAY)

    # Print progress every 20 episodes
    if (episode + 1) % 20 == 0:
        recent_avg = np.mean(episode_rewards[-20:])  # average of last 20 episodes
        print(f"  Episode {episode+1:3d}/300  |  "
              f"Last-20 avg reward: {recent_avg:7.1f}  |  "
              f"Noise scale: {noise_scale:.3f}")


# STEP 3: Evaluate the trained agent

print()
print("  Evaluating trained agent (no noise)...")

eval_rewards = []
for seed in range(5):
    np.random.seed(seed)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, add_noise=False)  # pure policy, no exploration
        state, reward, done, _ = env.step(action)
        total_reward += reward
    eval_rewards.append(total_reward)

avg_eval = np.mean(eval_rewards)
print(f"  Trained agent final avg reward: {avg_eval:.1f}")
print(f"  c-μ target:                     {avg_cmu:.1f}")
gap_pct = abs((avg_eval - avg_cmu) / avg_cmu) * 100
print(f"  Gap from optimal:               {gap_pct:.1f}%")
print()


# STEP 4: Plot — Reward vs Episode

# Smooth the noisy reward curve with a rolling average (window=20)
def rolling_average(data, window=20):
    return [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]

smoothed = rolling_average(episode_rewards, window=20)

fig, ax = plt.subplots(figsize=(10, 5))

# Raw episode rewards (faint, so you can see the noise)
ax.plot(episode_rewards, color="#aed6f1", linewidth=0.7, alpha=0.7, label="Episode reward (raw)")

# Smoothed curve (the actual trend)
ax.plot(smoothed, color="#2980b9", linewidth=2.0, label="Smoothed reward (20-ep avg)")

# Reference lines (horizontal dashed lines for baselines)
ax.axhline(avg_cmu,   color="#2ecc71", linestyle="--", linewidth=1.8, label=f"c-μ Rule (optimal): {avg_cmu:.0f}")
ax.axhline(avg_lcq,   color="#f39c12", linestyle="--", linewidth=1.2, label=f"LCQ Rule: {avg_lcq:.0f}")
ax.axhline(avg_fixed, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Fixed 50/50: {avg_fixed:.0f}")

ax.set_title("DDPG Learning Curve — Reward vs Episode", fontsize=13, fontweight="bold")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Total Episode Reward", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
print("  Plot saved to: reward_curve.png")
plt.show()
print()
print("Step 2 complete.")