import numpy as np

class Router:
    """Represents a network router with 8 priority queues on its egress port."""
    def __init__(self, router_id, n_queues=8, service_rate=3.5, max_queue_len=50, holding_costs=None, drop_costs=None):
        self.router_id = router_id
        self.n_queues = n_queues
        self.service_rate = service_rate
        self.max_queue_len = max_queue_len

        # DiffServ / IEEE 802.1p standard holding costs (delay penalty) and drop penalties
        # Q0 (Network Control) to Q7 (Background)
        self.holding_costs = holding_costs if holding_costs is not None else np.array([
            10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5
        ], dtype=np.float64)
        
        self.drop_costs = drop_costs if drop_costs is not None else np.array([
            25.0, 20.0, 15.0, 10.0, 8.0, 5.0, 2.5, 1.0
        ], dtype=np.float64)

        self.queues = np.zeros(n_queues, dtype=np.float64)
        self.total_drops = 0.0

    def reset(self):
        self.queues = np.zeros(self.n_queues, dtype=np.float64)
        self.total_drops = 0.0

    def receive_packets(self, arrivals):
        """Enqueue arriving packets and drop any exceeding max buffer K."""
        self.queues += arrivals
        drops = np.maximum(self.queues - self.max_queue_len, 0.0)
        self.queues = np.minimum(self.queues, self.max_queue_len)
        self.total_drops += drops.sum()
        return drops

    def service_queues(self, weights):
        """Service packets according to continuous bandwidth weights summing to 1."""
        weights = np.array(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(self.n_queues) / self.n_queues

        service_allocated = weights * self.service_rate
        actual_served = np.minimum(service_allocated, self.queues)
        self.queues -= actual_served
        return actual_served

    def get_cost(self, drops):
        """Dual-objective reward penalty: r = -(h*q + c*drops)."""
        holding_penalty = np.sum(self.holding_costs * self.queues)
        drop_penalty = np.sum(self.drop_costs * drops)
        return holding_penalty + drop_penalty


class QueueEnv:
    """
    Multi-node, multi-router network environment with 8 DiffServ priority queues.
    
    Topology:
      [Host Node 1] ───┐                                                ┌───> [Host Node 3]
                       ├──> [Router 1: Ingress] ──> [Router 2: Core] ──>┤
      [Host Node 2] ───┘        (8 Queues)            (8 Queues)        └───> [Host Node 4]
                                                          │
                                                    (Bottleneck Link)
    """
    def __init__(
        self,
        n_queues=8,
        n_routers=2,
        n_nodes=4,
        arrival_rates=None,
        service_rate=3.2,
        ingress_service_rate=4.0,
        holding_costs=None,
        drop_costs=None,
        max_queue_len=50,
        episode_length=200,
        burst_prob=0.15,
        burst_scale=2.5,
    ):
        self.n_queues = n_queues
        self.n_routers = n_routers
        self.n_nodes = n_nodes
        self.service_rate = service_rate
        self.max_queue_len = max_queue_len
        self.episode_length = episode_length
        self.burst_prob = burst_prob
        self.burst_scale = burst_scale

        # Base arrival rates for 8 traffic classes (VoIP, Video, Web, Bulk, etc.)
        if arrival_rates is not None:
            self.arrival_rates = np.array(arrival_rates, dtype=np.float64)
        else:
            # Aggregate arrivals ~ 2.6 packets/slot (load factor ~ 0.81 relative to core service_rate 3.2)
            self.arrival_rates = np.array([
                0.10,  # Q0: Network Control
                0.20,  # Q1: Voice (EF)
                0.35,  # Q2: Interactive Video (AF4)
                0.40,  # Q3: Streaming Media (AF3)
                0.45,  # Q4: Critical Data (AF2)
                0.50,  # Q5: Standard Data (AF1)
                0.40,  # Q6: Best Effort (BE)
                0.20,  # Q7: Background
            ], dtype=np.float64)

        # Standard QoS holding costs (delay sensitivity)
        if holding_costs is not None:
            self.holding_costs = np.array(holding_costs, dtype=np.float64)
        else:
            self.holding_costs = np.array([10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=np.float64)

        # Drop penalty costs (packet loss penalty)
        if drop_costs is not None:
            self.drop_costs = np.array(drop_costs, dtype=np.float64)
        else:
            self.drop_costs = np.array([25.0, 20.0, 15.0, 10.0, 8.0, 5.0, 2.5, 1.0], dtype=np.float64)

        # Routers in tandem/bottleneck topology
        self.router1 = Router(
            router_id=1,
            n_queues=n_queues,
            service_rate=ingress_service_rate,
            max_queue_len=max_queue_len,
            holding_costs=self.holding_costs,
            drop_costs=self.drop_costs,
        )
        self.router2 = Router(
            router_id=2,
            n_queues=n_queues,
            service_rate=service_rate,
            max_queue_len=max_queue_len,
            holding_costs=self.holding_costs,
            drop_costs=self.drop_costs,
        )

        self.step_count = 0
        self.total_drops = 0.0

    @property
    def queues(self):
        """Returns the core bottleneck router queue lengths for agent observation/baselines."""
        return self.router2.queues

    def reset(self):
        self.router1.reset()
        self.router2.reset()
        self.step_count = 0
        self.total_drops = 0.0
        return self._get_state()

    def _generate_host_traffic(self):
        """Generates realistic bursty (Poisson + burst) traffic from host nodes."""
        rates = self.arrival_rates.copy()
        # Non-Poisson sudden burst model (as highlighted in Huawei paper)
        if np.random.rand() < self.burst_prob:
            burst_q = np.random.randint(0, self.n_queues)
            rates[burst_q] += np.random.exponential(self.burst_scale)

        arrivals_node1 = np.random.poisson(rates * 0.6)
        arrivals_node2 = np.random.poisson(rates * 0.4)
        return arrivals_node1 + arrivals_node2

    def step(self, action):
        """
        Executes one timestep slot:
        1. Host traffic arrives at Ingress Router 1.
        2. Router 1 services packets using WFQ baseline (or action).
        3. Served packets from Router 1 flow into Core Bottleneck Router 2 + cross traffic.
        4. Core Router 2 services packets according to action weights [w0..w7].
        5. Serviced packets leave Core Router 2 to Destination Nodes 3 & 4.
        """
        # 1. Host Arrivals to Router 1
        host_arrivals = self._generate_host_traffic()
        r1_drops = self.router1.receive_packets(host_arrivals)

        # Router 1 default WFQ weights (static fair allocation proportional to priority)
        r1_weights = np.array([0.22, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05])
        r1_served = self.router1.service_queues(r1_weights)

        # 2. Router 1 forwarding + cross-traffic arrives at Core Router 2
        # Cross traffic from Node 2 local burst
        cross_traffic = np.random.poisson(self.arrival_rates * 0.15)
        r2_arrivals = r1_served + cross_traffic
        r2_drops = self.router2.receive_packets(r2_arrivals)

        # 3. Core Router 2 allocates bandwidth according to agent action
        r2_served = self.router2.service_queues(action)

        # 4. Total packet drops across the multi-router network
        step_drops = r1_drops.sum() + r2_drops.sum()
        self.total_drops += step_drops

        # 5. Reward: Little's Law queue delay penalty + packet loss penalty across routers
        cost_r1 = self.router1.get_cost(r1_drops)
        cost_r2 = self.router2.get_cost(r2_drops)
        reward = -(cost_r2 + 0.3 * cost_r1)  # Focus primarily on bottleneck router with upstream feedback

        self.step_count += 1
        done = self.step_count >= self.episode_length

        info = {
            "arrivals": host_arrivals,
            "r1_queues": self.router1.queues.copy(),
            "r2_queues": self.router2.queues.copy(),
            "queue_lengths": self.router2.queues.copy(),
            "drops": step_drops,
            "r1_drops": r1_drops.sum(),
            "r2_drops": r2_drops.sum(),
            "service": r2_served,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """Normalized state vector of the bottleneck router [0, 1]."""
        return (self.router2.queues.copy() / self.max_queue_len).astype(np.float32)
