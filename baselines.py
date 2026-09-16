"""
baselines.py  —  Three hand-crafted bandwidth-splitting policies
=================================================================

Each function takes the current state and returns an action [w1, w2].

None of these "learn" anything — they follow fixed rules.
We'll use them as reference lines to see if our RL agent can match
or beat them.
"""

import numpy as np
def cmu_rule(queue_lengths, holding_costs):
    """
    Args:
        queue_lengths: array [q1, q2] — actual (unnormalized) queue lengths
        holding_costs: array [h1, h2] — cost per packet per step

    Returns:
        action: array [w1, w2] summing to 1.0
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
    Args:
        queue_lengths: array [q1, q2] — actual (unnormalized) queue lengths

    Returns:
        action: array [w1, w2] summing to 1.0
    """
    n = len(queue_lengths)
    action = np.zeros(n)

    if np.sum(queue_lengths) == 0:
        action = np.ones(n) / n
    else:
        longest = np.argmax(queue_lengths)
        action[longest] = 1.0

    return action

def fixed_split_rule(n_queues=2):
    """
    Returns:
        action: array [0.5, 0.5] (equal split)
    """
    return np.ones(n_queues) / n_queues
