import numpy as np

def cmu_rule(queue_lengths, holding_costs):
    """
    c-mu Rule: Allocates bandwidth to the non-empty queue with the highest holding cost.
    Proven mathematically optimal for infinite buffers and linear holding costs.
    """
    n = len(queue_lengths)
    action = np.zeros(n)
    scores = np.array(holding_costs) * (np.array(queue_lengths) > 0).astype(float)

    if scores.sum() == 0:
        action = np.ones(n) / n
    else:
        best = np.argmax(scores)
        action[best] = 1.0

    return action

def lcq_rule(queue_lengths):
    """
    Longest Connected Queue (LCQ): Allocates bandwidth to the queue with the largest backlog.
    """
    n = len(queue_lengths)
    action = np.zeros(n)

    if np.sum(queue_lengths) == 0:
        action = np.ones(n) / n
    else:
        longest = np.argmax(queue_lengths)
        action[longest] = 1.0

    return action

def fixed_split_rule(n_queues=8):
    """Equal bandwidth split across all queues."""
    return np.ones(n_queues) / n_queues

def wfq_static_rule(n_queues=8):
    """
    Static Weighted Fair Queueing (WFQ) weights based on QoS priority tiers.
    Higher priority queues receive larger static bandwidth fractions.
    """
    if n_queues == 8:
        weights = np.array([0.24, 0.20, 0.16, 0.13, 0.10, 0.08, 0.05, 0.04], dtype=np.float64)
        return weights / weights.sum()
    elif n_queues == 2:
        return np.array([0.7, 0.3], dtype=np.float64)
    else:
        raw = np.linspace(n_queues, 1, n_queues)
        return raw / raw.sum()

def priority_queueing_rule(queue_lengths):
    """
    Strict Priority Queueing (PQ): Highest priority non-empty queue receives 100% bandwidth.
    """
    n = len(queue_lengths)
    action = np.zeros(n)
    non_empty = np.where(np.array(queue_lengths) > 0)[0]
    if len(non_empty) == 0:
        action = np.ones(n) / n
    else:
        action[non_empty[0]] = 1.0
    return action
