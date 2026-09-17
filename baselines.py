import numpy as np

def cmu_rule(queue_lengths, holding_costs):
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
    n = len(queue_lengths)
    action = np.zeros(n)

    if np.sum(queue_lengths) == 0:
        action = np.ones(n) / n
    else:
        longest = np.argmax(queue_lengths)
        action[longest] = 1.0

    return action

def fixed_split_rule(n_queues=2):
    return np.ones(n_queues) / n_queues
