import numpy as np

def compute_jains_fairness_index(served_per_queue, arrivals_per_queue, eps=1e-6):
    """
    Computes Jain's Fairness Index based on Demand Satisfaction Ratio:
        x_i = served_i / (arrivals_i + eps)
    for all active queues (queues with non-zero arrivals).
    
    Formula:
        J(x) = (sum(x_i))^2 / (n * sum(x_i^2))
    
    Returns a score in [1/n, 1.0], where 1.0 denotes perfectly fair satisfaction.
    """
    served = np.array(served_per_queue, dtype=np.float64)
    arrivals = np.array(arrivals_per_queue, dtype=np.float64)

    active_mask = arrivals > 1e-3
    if not active_mask.any():
        return 1.0

    ratios = served[active_mask] / (arrivals[active_mask] + eps)
    # Clip ratio to 1.0 max since serving more than arrived just drains prior backlog
    ratios = np.clip(ratios, 0.0, 1.0)
    
    n = len(ratios)
    sum_r = np.sum(ratios)
    sum_sq_r = np.sum(ratios ** 2)

    if sum_sq_r < 1e-9:
        return 0.0

    jfi = (sum_r ** 2) / (n * sum_sq_r)
    return float(np.clip(jfi, 1.0 / n, 1.0))


def compute_episode_metrics(episode_records):
    """
    Computes standard network QoS and fairness metrics over an episode trajectory.
    
    episode_records is a dict containing:
      - 'timesteps': int
      - 'arrivals': (T, 8) array of packet arrivals per queue
      - 'services': (T, 8) array of served packets per queue (router 2 egress)
      - 'queues':   (T, 8) array of core router queue lengths
      - 'drops':    (T, 8) array of packet drops per queue
      - 'raw_costs':(T,) array of step penalties
    """
    T = episode_records["timesteps"]
    arrivals = np.array(episode_records["arrivals"])  # shape: (T, 8)
    services = np.array(episode_records["services"])  # shape: (T, 8)
    queues = np.array(episode_records["queues"])      # shape: (T, 8)
    drops = np.array(episode_records["drops"])        # shape: (T, 8)
    raw_costs = np.array(episode_records["raw_costs"])

    total_arrivals = arrivals.sum()
    total_served = services.sum()
    total_drops = drops.sum()

    # 1. Throughput (packets per tick)
    throughput_per_tick = total_served / max(T, 1)

    # 2. Packet Loss Rate
    loss_rate = total_drops / max(total_arrivals, 1e-6)

    # 3. Average Queue Delay (Little's Law: Delay = L / lambda)
    avg_total_queue_backlog = queues.sum(axis=1).mean()
    mean_arrival_rate = total_arrivals / max(T, 1)
    avg_delay_ticks = avg_total_queue_backlog / max(mean_arrival_rate, 1e-4)

    # 4. Starvation Incidents
    # Timesteps where a queue has backlog > 0.05 but receives 0 service (< 1e-4)
    backlogged = (queues > 0.05)
    unserved = (services < 1e-4)
    starvation_mask = backlogged & unserved
    starvation_incidents = int(starvation_mask.sum())

    # 5. Jain's Fairness Index
    tot_served_per_q = services.sum(axis=0)
    tot_arrivals_per_q = arrivals.sum(axis=0)
    jfi = compute_jains_fairness_index(tot_served_per_q, tot_arrivals_per_q)

    # 6. Cumulative Penalty Cost
    total_cost = float(raw_costs.sum())

    return {
        "throughput": float(throughput_per_tick),
        "total_served": float(total_served),
        "packet_loss_count": float(total_drops),
        "packet_loss_rate": float(loss_rate),
        "avg_queue_delay": float(avg_delay_ticks),
        "avg_queue_backlog": float(avg_total_queue_backlog),
        "starvation_incidents": starvation_incidents,
        "jains_fairness_index": float(jfi),
        "total_penalty_cost": float(total_cost),
        "reward": float(-total_cost * 0.01),
    }
