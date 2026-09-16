"""
queue_env.py  —  A simple 2-queue router environment
=====================================================

Imagine you're a router with 2 incoming traffic queues.
Every timestep:
  1. Random packets ARRIVE in each queue (Poisson random).
  2. YOU decide how to split your bandwidth between the two queues.
  3. Packets get SERVED (removed) according to your split.
  4. You get a REWARD = negative holding cost (shorter queues = better).

The state is just the current lengths of both queues: [q1, q2].
The action is weights [w1, w2] that sum to 1.0 (your bandwidth split).
"""

import numpy as np


class QueueEnv:
    def __init__(
        self,
        n_queues=2,               # Number of queues (we use 2)
        arrival_rates=None,       # Average packets arriving per step in each queue
        service_rate=1.5,         # Total bandwidth (packets you can serve per step)
        holding_costs=None,       # Cost per packet per step in each queue
        max_queue_len=50,         # If a queue exceeds this, packets are "dropped" (lost)
        episode_length=200,       # How many timesteps per episode
    ):
        self.n_queues = n_queues
        # Default: queue 0 gets more traffic, queue 1 gets less
        self.arrival_rates = arrival_rates if arrival_rates is not None else [0.6, 0.4]
        self.service_rate = service_rate
        # Default: queue 0 is "expensive" to hold, queue 1 is "cheap"
        self.holding_costs = holding_costs if holding_costs is not None else [3.0, 1.0]
        self.max_queue_len = max_queue_len
        self.episode_length = episode_length

        # --- internal bookkeeping ---
        self.queues = np.zeros(n_queues, dtype=np.float64)  # current queue lengths
        self.step_count = 0
        self.total_drops = 0  # packets lost due to overflow

    # ------------------------------------------------------------------
    # reset()  —  called at the start of each episode
    # Returns the initial state (both queues empty).
    # ------------------------------------------------------------------
    def reset(self):
        self.queues = np.zeros(self.n_queues, dtype=np.float64)
        self.step_count = 0
        self.total_drops = 0
        return self._get_state()

    # ------------------------------------------------------------------
    # step(action)  —  one timestep of the simulation
    #   action: array of weights [w1, w2] that sum to ~1.0
    #   Returns: (next_state, reward, done, info_dict)
    # ------------------------------------------------------------------
    def step(self, action):
        # --- 1. ARRIVALS: random packets arrive (Poisson distribution) ---
        arrivals = np.random.poisson(self.arrival_rates)
        self.queues += arrivals

        # --- 2. SERVICE: remove packets according to your bandwidth split ---
        # action = [w1, w2], e.g. [0.7, 0.3] means 70% bandwidth to queue 1
        weights = np.array(action, dtype=np.float64)
        weights = np.clip(weights, 0, None)          # no negative weights
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum                 # normalize to sum to 1
        else:
            weights = np.ones(self.n_queues) / self.n_queues  # fallback: equal split

        # How many packets each queue gets served
        service = weights * self.service_rate
        # Can't serve more than what's in the queue
        service = np.minimum(service, self.queues)
        self.queues -= service

        # --- 3. OVERFLOW: drop packets if queue exceeds max ---
        drops = np.maximum(self.queues - self.max_queue_len, 0)
        self.queues = np.minimum(self.queues, self.max_queue_len)
        self.total_drops += drops.sum()

        # --- 4. REWARD: negative holding cost (lower queues = higher reward) ---
        # reward = -( h1*q1 + h2*q2 )
        # A perfect policy keeps queues near 0, so reward is near 0.
        # A bad policy lets queues grow, so reward is a large negative number.
        holding_cost = np.sum(np.array(self.holding_costs) * self.queues)
        reward = -holding_cost

        # --- 5. CHECK IF EPISODE IS OVER ---
        self.step_count += 1
        done = self.step_count >= self.episode_length

        info = {
            "arrivals": arrivals,
            "service": service,
            "drops": drops.sum(),
            "queue_lengths": self.queues.copy(),
        }

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # _get_state()  —  what the agent "sees"
    # ------------------------------------------------------------------
    def _get_state(self):
        # Normalize queue lengths to [0, 1] range so neural networks are happy later
        return self.queues.copy() / self.max_queue_len
