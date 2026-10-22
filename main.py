import sys
import os
import argparse
import numpy as np

from queue_env import QueueEnv
from baselines import cmu_rule, lcq_rule, wfq_static_rule, priority_queueing_rule
from ddpg_agent import DDPGAgent
from sac_agent import SACAgent
from evaluate_generalization import evaluate_all, print_formatted_tables, plot_generalization_charts

EVAL_SEEDS = [101, 102, 103, 104, 105]

def evaluate_standard_benchmark(ddpg_agent, sac_agent):
    policies = {
        "SAC (RL)": ("sac", sac_agent),
        "DDPG (RL)": ("ddpg", ddpg_agent),
        "Static WFQ": ("wfq", None),
        "c-μ Rule": ("cmu", None),
        "Strict PQ": ("pq", None),
        "LCQ Rule": ("lcq", None),
    }

    print("\n" + "=" * 90)
    print("  STANDARD BENCHMARK (DEFAULT NETWORK TOPOLOGY, 5 SEEDS)")
    print("=" * 90)

    results = {}
    for name, (ptype, agent) in policies.items():
        seed_rewards, seed_drops, seed_delays, seed_jfi = [], [], [], []

        for seed in EVAL_SEEDS:
            np.random.seed(seed)
            env = QueueEnv()
            state = env.reset()
            done = False
            ep_reward = 0.0
            queues_hist = []
            arrivals_hist = []
            service_hist = []

            while not done:
                if ptype == "sac":
                    act = agent.select_action(state, deterministic=True)
                elif ptype == "ddpg":
                    act = agent.select_action(state, add_noise=False)
                elif ptype == "wfq":
                    act = wfq_static_rule(env.n_queues)
                elif ptype == "cmu":
                    act = cmu_rule(env.queues, env.holding_costs)
                elif ptype == "pq":
                    act = priority_queueing_rule(env.queues)
                elif ptype == "lcq":
                    act = lcq_rule(env.queues)

                state, r, done, info = env.step(act)
                ep_reward += r
                queues_hist.append(info["queue_lengths"])
                arrivals_hist.append(info["arrivals"])
                service_hist.append(info["service"])

            # Compute stats
            tot_arr = np.sum(arrivals_hist, axis=0)
            tot_srv = np.sum(service_hist, axis=0)
            act_mask = tot_arr > 1e-3
            ratios = np.clip(tot_srv[act_mask] / (tot_arr[act_mask] + 1e-6), 0.0, 1.0)
            jfi = (np.sum(ratios) ** 2) / (len(ratios) * np.sum(ratios ** 2))

            mean_delay = np.mean(np.sum(queues_hist, axis=1)) / (np.sum(tot_arr) / 200.0)

            seed_rewards.append(ep_reward)
            seed_drops.append(env.total_drops)
            seed_delays.append(mean_delay)
            seed_jfi.append(jfi)

        results[name] = {
            "reward": np.mean(seed_rewards),
            "drops": np.mean(seed_drops),
            "delay": np.mean(seed_delays),
            "jfi": np.mean(seed_jfi),
        }

    header = f"  {'Policy':16s} | {'Avg Reward':>12s} | {'Drops (pkts)':>14s} | {'Avg Delay (ticks)':>18s} | {'Jains Index':>14s}"
    print(header)
    print("  " + "-" * 84)
    for name, r in results.items():
        print(f"  {name:16s} | {r['reward']:12.1f} | {r['drops']:14.1f} | {r['delay']:18.2f} | {r['jfi']:14.3f}")
    print("=" * 90 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Network QoS Benchmarking & Generalization")
    parser.add_argument("--train", action="store_true", help="Train DDPG on the 5 training scenarios before evaluating")
    parser.add_argument("--episodes", type=int, default=120, help="Training episodes if --train is passed (default: 120)")
    parser.add_argument("--standard-only", action="store_true", help="Run only standard topology benchmark")
    parser.add_argument("--scenarios-only", action="store_true", help="Run only 10-scenario generalization test")
    args = parser.parse_args()

    ddpg_ckpt = "ddpg_generalization_checkpoint.pth"
    sac_ckpt = "sac_generalization_checkpoint.pth"

    # Train if requested
    if args.train:
        from evaluate_generalization import train_ddpg_multiscenario, train_sac_multiscenario
        print("\n" + "=" * 80)
        print("  TRAINING BOTH DDPG AND SAC ON MULTI-SCENARIO TRAINING SET")
        print("=" * 80)
        
        # 1. Train DDPG
        ddpg_agent = DDPGAgent(state_dim=8, action_dim=8, hidden=128)
        train_ddpg_multiscenario(ddpg_agent, total_episodes=args.episodes, checkpoint_path=ddpg_ckpt)

        # 2. Train SAC
        sac_agent = SACAgent(state_dim=8, action_dim=8, hidden=128)
        train_sac_multiscenario(sac_agent, total_episodes=args.episodes, checkpoint_path=sac_ckpt)

    # Load agents
    ddpg_agent = DDPGAgent(state_dim=8, action_dim=8, hidden=128)
    if os.path.exists(ddpg_ckpt):
        ddpg_agent.load(ddpg_ckpt)
    elif os.path.exists("ddpg_checkpoint.pth"):
        ddpg_agent.load("ddpg_checkpoint.pth")

    sac_agent = SACAgent(state_dim=8, action_dim=8, hidden=128)
    if os.path.exists(sac_ckpt):
        sac_agent.load(sac_ckpt)
    elif os.path.exists("sac_checkpoint.pth"):
        sac_agent.load("sac_checkpoint.pth")

    if not args.scenarios_only:
        evaluate_standard_benchmark(ddpg_agent, sac_agent)

    if not args.standard_only:
        results = evaluate_all(ddpg_ckpt=ddpg_ckpt, sac_ckpt=sac_ckpt)
        print_formatted_tables(results)
        plot_generalization_charts(results, output_file="generalization_comparison.png")

    print("✓ All evaluations completed successfully.\n")


if __name__ == "__main__":
    main()
