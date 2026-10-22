import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, fixed_split_rule, wfq_static_rule, priority_queueing_rule
from ddpg_agent import DDPGAgent
from sac_agent import SACAgent
from scenarios import SCENARIOS, get_scenario, get_scenarios_by_split
from metrics import compute_episode_metrics

EVAL_SEEDS = [101, 102, 103, 104, 105]
EPISODE_LENGTH = 200

def create_env_for_scenario(scenario, episode_length=EPISODE_LENGTH):
    return QueueEnv(
        n_queues=8,
        arrival_rates=scenario.arrival_rates,
        service_rate=scenario.service_rate,
        ingress_service_rate=scenario.ingress_service_rate,
        burst_prob=scenario.burst_prob,
        burst_scale=scenario.burst_scale,
        episode_length=episode_length,
        traffic_fn=scenario.get_traffic_fn(),
    )


def run_episode_and_record(env, policy_name, agent=None, seed=42):
    np.random.seed(seed)
    state = env.reset()
    if agent is not None and hasattr(agent, "noise"):
        agent.noise.reset()

    arrivals_hist, services_hist, queues_hist, drops_hist, costs_hist = [], [], [], [], []

    done = False
    while not done:
        if policy_name == "ddpg":
            action = agent.select_action(state, add_noise=False)
        elif policy_name == "sac":
            action = agent.select_action(state, deterministic=True)
        elif policy_name == "wfq":
            action = wfq_static_rule(env.n_queues)
        elif policy_name == "cmu":
            action = cmu_rule(env.queues, env.holding_costs)
        elif policy_name == "lcq":
            action = lcq_rule(env.queues)
        elif policy_name == "pq":
            action = priority_queueing_rule(env.queues)
        elif policy_name == "fixed":
            action = fixed_split_rule(env.n_queues)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        next_state, reward, done, info = env.step(action)

        arrivals_hist.append(info["arrivals"].copy())
        services_hist.append(info["service"].copy())
        queues_hist.append(info["queue_lengths"].copy())
        drops_hist.append(info["per_queue_drops"].copy())
        costs_hist.append(info["raw_cost"])

        state = next_state

    records = {
        "timesteps": len(arrivals_hist),
        "arrivals": np.array(arrivals_hist),
        "services": np.array(services_hist),
        "queues": np.array(queues_hist),
        "drops": np.array(drops_hist),
        "raw_costs": np.array(costs_hist),
    }
    return compute_episode_metrics(records)


def train_ddpg_multiscenario(agent, total_episodes=200, checkpoint_path="ddpg_generalization_checkpoint.pth"):
    """
    Trains DDPG strictly on the 5 Training Scenarios.
    Ensures zero exposure to any test scenario during training.
    """
    train_scenarios = get_scenarios_by_split("train")
    print("\n" + "=" * 75)
    print(f"  TRAINING DDPG ON {len(train_scenarios)} TRAINING SCENARIOS EXCLUSIVELY")
    print(f"  Total episodes: {total_episodes} (cycled across Train 1..5)")
    print("=" * 75)

    noise_scale = 1.0
    NOISE_DECAY = 0.995
    MIN_NOISE = 0.05
    rewards_log = []

    for ep in range(total_episodes):
        # Round-robin selection of training scenarios
        scenario = train_scenarios[ep % len(train_scenarios)]
        env = create_env_for_scenario(scenario)

        # Set training seed
        train_seed = 2000 + ep
        np.random.seed(train_seed)
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

        rewards_log.append(ep_reward)
        noise_scale = max(MIN_NOISE, noise_scale * NOISE_DECAY)

        if (ep + 1) % 25 == 0:
            avg_rew = np.mean(rewards_log[-25:])
            print(f"  Episode {ep+1:3d}/{total_episodes} | Scenario: {scenario.name[:25]:25s} | "
                  f"Last-25 Avg Reward: {avg_rew:7.1f} | Noise: {noise_scale:.3f}")

    agent.save(checkpoint_path)
    print(f"\n  ✓ Successfully saved generalized DDPG model to {checkpoint_path}")
    return rewards_log


def train_sac_multiscenario(agent, total_episodes=120, checkpoint_path="sac_generalization_checkpoint.pth"):
    train_scenarios = get_scenarios_by_split("train")
    print("\n" + "=" * 75)
    print(f"  TRAINING SAC ON {len(train_scenarios)} TRAINING SCENARIOS EXCLUSIVELY")
    print(f"  Total episodes: {total_episodes} (cycled across Train 1..5)")
    print("=" * 75)

    rewards_log = []
    for ep in range(total_episodes):
        scenario = train_scenarios[ep % len(train_scenarios)]
        env = create_env_for_scenario(scenario)

        train_seed = 3000 + ep
        np.random.seed(train_seed)
        state = env.reset()
        ep_reward = 0.0

        for _ in range(EPISODE_LENGTH):
            action = agent.select_action(state, deterministic=False)
            next_state, r, done, _ = env.step(action)
            agent.store(state, action, r, next_state, float(done))
            agent.learn()
            state, ep_reward = next_state, ep_reward + r
            if done:
                break

        rewards_log.append(ep_reward)

        if (ep + 1) % 25 == 0:
            avg_rew = np.mean(rewards_log[-25:])
            print(f"  Episode {ep+1:3d}/{total_episodes} | Scenario: {scenario.name[:25]:25s} | "
                  f"Last-25 Avg Reward: {avg_rew:7.1f} | Alpha: {agent.alpha:.4f}")

    agent.save(checkpoint_path)
    print(f"\n  ✓ Successfully saved generalized SAC model to {checkpoint_path}")
    return rewards_log


def evaluate_all(ddpg_ckpt="ddpg_generalization_checkpoint.pth", sac_ckpt="sac_generalization_checkpoint.pth"):
    # Load DDPG
    ddpg_agent = DDPGAgent(state_dim=8, action_dim=8, hidden=128)
    if os.path.exists(ddpg_ckpt):
        ddpg_agent.load(ddpg_ckpt)
    elif os.path.exists("ddpg_checkpoint.pth"):
        ddpg_agent.load("ddpg_checkpoint.pth")

    # Load SAC
    sac_agent = SACAgent(state_dim=8, action_dim=8, hidden=128)
    if os.path.exists(sac_ckpt):
        sac_agent.load(sac_ckpt)
    elif os.path.exists("sac_checkpoint.pth"):
        sac_agent.load("sac_checkpoint.pth")

    policies = {
        "SAC (RL)": ("sac", sac_agent),
        "DDPG (RL)": ("ddpg", ddpg_agent),
        "Static WFQ": ("wfq", None),
        "c-μ Rule": ("cmu", None),
        "Strict PQ": ("pq", None),
        "LCQ Rule": ("lcq", None),
    }

    results = {}
    print("\n  Evaluating 10 scenarios (5 Train, 5 Unseen Test) over 5 seeds...", end="", flush=True)

    for scenario in SCENARIOS:
        sc_id = scenario.scenario_id
        sc_name = scenario.name
        results[sc_name] = {"split": scenario.split, "id": sc_id, "policies": {}}

        for pol_name, (pol_code, pol_agent) in policies.items():
            seed_metrics = []
            for seed in EVAL_SEEDS:
                env = create_env_for_scenario(scenario)
                m = run_episode_and_record(env, pol_code, pol_agent, seed=seed)
                seed_metrics.append(m)

            avg_m = {k: float(np.mean([sm[k] for sm in seed_metrics])) for k in seed_metrics[0].keys()}
            results[sc_name]["policies"][pol_name] = avg_m

    print(" Done.")
    return results


def print_formatted_tables(results, show_per_scenario=False):
    metric_keys = [
        ("Avg Reward", "reward", "{:8.1f}"),
        ("Drops (pkts)", "packet_loss_count", "{:7.1f}"),
        ("Avg Delay (ticks)", "avg_queue_delay", "{:6.2f}"),
        ("Throughput (pkts/t)", "throughput", "{:6.2f}"),
        ("Starvations", "starvation_incidents", "{:6.1f}"),
        ("Jain's Index", "jains_fairness_index", "{:6.3f}"),
    ]

    if show_per_scenario:
        print("\n" + "=" * 115)
        print("  PER-SCENARIO EVALUATION BREAKDOWN")
        print("=" * 115)
        for sc_name, sc_data in results.items():
            sc_id = sc_data["id"]
            split = sc_data["split"].upper()
            print(f"\n▶ Scenario {sc_id}: {sc_name} [{split}]")
            header = f"  {'Policy':18s}" + "".join([f" | {label:>18s}" for label, _, _ in metric_keys])
            print("  " + "-" * 115)
            print(header)
            print("  " + "-" * 115)
            for pol_name, pol_metrics in sc_data["policies"].items():
                row = f"  {pol_name:18s}"
                for _, key, fmt in metric_keys:
                    val = pol_metrics[key]
                    row += f" | {fmt.format(val):>18s}"
                print(row)

    print("\n" + "=" * 115)
    print("  GENERALIZATION BENCHMARK SUMMARY (ALL SCENARIOS AGGREGATED TOGETHER)")
    print("=" * 115)

    for target_split in ["train", "test"]:
        split_title = "IN-DISTRIBUTION TRAINING SCENARIOS (1-5)" if target_split == "train" else "OUT-OF-DISTRIBUTION UNSEEN TEST SCENARIOS (6-10)"
        print(f"\n● {split_title}:")
        header = f"  {'Policy':18s}" + "".join([f" | {label:>18s}" for label, _, _ in metric_keys])
        print("  " + "-" * 115)
        print(header)
        print("  " + "-" * 115)

        matching_scenarios = [d for d in results.values() if d["split"] == target_split]
        first_pol_keys = list(matching_scenarios[0]["policies"].keys())

        for pol_name in first_pol_keys:
            row = f"  {pol_name:18s}"
            for _, key, fmt in metric_keys:
                mean_val = np.mean([sc["policies"][pol_name][key] for sc in matching_scenarios])
                row += f" | {fmt.format(mean_val):>18s}"
            print(row)

    print("\n" + "=" * 115 + "\n")


def plot_generalization_charts(results, output_file="generalization_comparison.png"):
    scenario_names = [s.name.replace("Train_", "Tr: ").replace("Test_", "Te: ") for s in SCENARIOS]
    pol_names = list(results[SCENARIOS[0].name]["policies"].keys())

    colors = {
        "SAC (RL)": "#e74c3c",        # Red
        "DDPG (RL)": "#2980b9",       # Blue
        "Static WFQ": "#27ae60",      # Green
        "c-μ Rule": "#e67e22",        # Orange
        "Strict PQ": "#34495e",       # Slate
        "LCQ Rule": "#8e44ad",        # Purple
    }

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#ffffff")
    plt.suptitle("Generalization Analysis: DDPG vs Baseline Schedulers across 10 Scenarios\n"
                 "(Scenarios 1-5: Training Sets | Scenarios 6-10: Strict Out-of-Distribution Unseen Tests)",
                 fontsize=15, fontweight="bold", y=0.98)

    sub_specs = [
        ("Throughput (packets / tick)", "throughput", 1, "higher_better"),
        ("Total Packet Loss (Drops)", "packet_loss_count", 2, "lower_better"),
        ("Average Queue Delay (ticks)", "avg_queue_delay", 3, "lower_better"),
        ("Starvation Incidents", "starvation_incidents", 4, "lower_better"),
        ("Jain's Fairness Index", "jains_fairness_index", 5, "higher_better"),
    ]

    x = np.arange(len(scenario_names))
    width = 0.13

    for title, metric_key, idx, direction in sub_specs:
        ax = plt.subplot(3, 2, idx)
        for p_idx, pol in enumerate(pol_names):
            vals = [results[s.name]["policies"][pol][metric_key] for s in SCENARIOS]
            offset = (p_idx - len(pol_names)/2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=pol, color=colors[pol], alpha=0.9)

        ax.axvline(4.5, color="#e74c3c", linestyle="--", linewidth=1.5, label="Train | Test Border")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        if idx == 1:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel 6: Aggregate Comparison on Unseen Tests
    ax6 = plt.subplot(3, 2, 6)
    test_scenarios = [s for s in SCENARIOS if s.split == "test"]
    agg_metrics = ["throughput", "packet_loss_count", "avg_queue_delay", "starvation_incidents", "jains_fairness_index"]
    metric_labels = ["Throughput (↑)", "Packet Drops (↓)", "Avg Delay (↓)", "Starvations (↓)", "Jain's Fairness (↑)"]

    x_agg = np.arange(len(metric_labels))
    w_agg = 0.15

    pols_to_show = ["SAC (RL)", "DDPG (RL)", "Static WFQ", "c-μ Rule"]
    for i, pol in enumerate(pols_to_show):
        vals = [np.mean([results[s.name]["policies"][pol][m] for s in test_scenarios]) for m in agg_metrics]
        max_vals = [max([np.mean([results[s.name]["policies"][p][m] for s in test_scenarios]) for p in pols_to_show]) or 1e-4 for m in agg_metrics]
        norm_vals = [vals[k] / max_vals[k] for k in range(len(agg_metrics))]
        ax6.bar(x_agg + (i - 1.5) * w_agg, norm_vals, w_agg, label=pol, color=colors[pol], alpha=0.9)

    ax6.set_title("Unseen Test Scenarios: Normalized Comparison", fontsize=12, fontweight="bold")
    ax6.set_xticks(x_agg)
    ax6.set_xticklabels(metric_labels, fontsize=8.5)
    ax6.set_ylabel("Normalized (fraction of max)", fontsize=9)
    ax6.set_ylim(0, 1.15)
    ax6.grid(axis="y", linestyle=":", alpha=0.6)
    ax6.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"  ✓ Saved visualization chart to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Generalization across 10 Traffic Scenarios")
    parser.add_argument("--train", action="store_true", help="Train DDPG strictly on 5 training scenarios")
    parser.add_argument("--episodes", type=int, default=120, help="Total training episodes")
    parser.add_argument("--checkpoint", type=str, default="ddpg_generalization_checkpoint.pth", help="Checkpoint file")
    parser.add_argument("--json-out", type=str, default="generalization_results.json", help="JSON output file")
    parser.add_argument("--plot-out", type=str, default="generalization_comparison.png", help="PNG plot output file")
    args = parser.parse_args()

    if args.train:
        agent = DDPGAgent(state_dim=8, action_dim=8, hidden=128)
        train_ddpg_multiscenario(agent, total_episodes=args.episodes, checkpoint_path=args.checkpoint)

    results = evaluate_all(ddpg_ckpt=args.checkpoint)
    print_formatted_tables(results)
    plot_generalization_charts(results, output_file=args.plot_out)

    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to {args.json_out}\n")


if __name__ == "__main__":
    main()
