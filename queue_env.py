import numpy as np

class Router:
    """
    Represents a network router with 8 priority queues on its egress port.
    Implements standard work-conserving Weighted Fair Queueing (WFQ).
    """
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
            50.0, 40.0, 30.0, 25.0, 20.0, 18.0, 15.0, 12.0
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
        """
        Work-conserving Weighted Fair Queueing (WFQ):
        Services packets according to continuous bandwidth weights.
        If a queue has fewer packets than its allocated quota, unused capacity
        is redistributed to the remaining backlogged queues.
        """
        weights = np.array(weights, dtype=np.float64)
        weights = np.clip(weights, 1e-4, None)
        
        served = np.zeros(self.n_queues, dtype=np.float64)
        remaining_capacity = float(self.service_rate)
        active_mask = (self.queues > 1e-6)

        # Redistribute unused capacity over active queues (up to 3 passes)
        for _ in range(3):
            if remaining_capacity <= 1e-6 or not active_mask.any():
                break

            sub_w = weights * active_mask
            w_sum = sub_w.sum()
            if w_sum <= 1e-6:
                sub_w = active_mask.astype(np.float64)
                w_sum = sub_w.sum()
            sub_w = sub_w / w_sum

            allocated = sub_w * remaining_capacity
            actual = np.minimum(allocated, self.queues - served)
            served += actual
            remaining_capacity -= actual.sum()
            active_mask = ((self.queues - served) > 1e-6)

        self.queues -= served
        return served

    def get_cost(self, drops):
        """Dual-objective penalty: holding delay + packet loss + buffer overflow warning."""
        holding_penalty = np.sum(self.holding_costs * self.queues)
        drop_penalty = np.sum(self.drop_costs * drops)
        buffer_warning = np.sum(np.maximum(self.queues - 0.75 * self.max_queue_len, 0.0) * 2.0)
        return holding_penalty + drop_penalty + buffer_warning


# =====================================================================
# 5 Specialized Traffic Generator Functions (Voice, Video, Web, Bulk, Bursty)
# =====================================================================

def generate_voice_traffic(base_rate=0.20, talkspurt_prob=0.40, packet_size_bytes=128):
    """
    1. Voice Traffic (VoIP / EF - Queue 1):
    - Models ITU-T P.59 talkspurt ON/OFF voice frame arrivals.
    - Mean arrival rate: ~0.20 pkts/tick.
    """
    is_active = (np.random.rand() < talkspurt_prob)
    pkts = np.random.poisson(base_rate / talkspurt_prob) if is_active else 0
    return pkts, packet_size_bytes


def generate_video_traffic(base_rate=0.35, iframe_prob=0.10, iframe_mult=2.5, packet_size_bytes=1100):
    """
    2. Video Traffic (Interactive Video / AF4 - Queue 2):
    - Models Variable Bit Rate (VBR) video stream with periodic I-frame surges.
    - Mean arrival rate: ~0.35 pkts/tick.
    """
    p_rate = base_rate * 0.7
    i_rate = base_rate * 0.3 * iframe_mult
    rate = i_rate if np.random.rand() < (0.3 / iframe_mult) else p_rate
    pkts = np.random.poisson(rate)
    return pkts, packet_size_bytes


def generate_web_traffic(base_rate=0.50, pareto_alpha=2.5, packet_size_bytes=600):
    """
    3. Web Traffic (Standard Data / AF1 - Queue 5):
    - Models HTTP browsing transactions with Pareto heavy-tailed document sizes.
    - Mean arrival rate: ~0.50 pkts/tick.
    """
    if np.random.rand() < 0.25:
        pkts = np.random.poisson(base_rate * 2.0)
    else:
        pkts = np.random.poisson(base_rate * 0.65)
    return pkts, packet_size_bytes


def generate_bulk_traffic(base_rate=0.20, flow_prob=0.15, flow_scale=0.8, packet_size_bytes=1500):
    """
    4. Bulk Traffic (File Transfer / Background - Queue 7):
    - Models continuous bulk data / FTP transfers at Ethernet MTU (1500 bytes).
    - Mean arrival rate: ~0.20 pkts/tick.
    """
    if np.random.rand() < flow_prob:
        pkts = np.random.poisson(base_rate + flow_scale)
    else:
        pkts = np.random.poisson(base_rate * 0.4)
    return pkts, packet_size_bytes


def generate_bursty_traffic(base_rate=0.45, burst_prob=0.10, burst_scale=1.2, packet_size_bytes=800):
    """
    5. Bursty / Flash Crowd Traffic (Streaming AF3 / Q3 & Critical Data AF2 / Q4):
    - Models sudden arrival spikes simulating flash crowds across queues.
    - Mean arrival rate: ~0.40 (Q3) and ~0.45 (Q4) pkts/tick.
    """
    pkts_q3 = np.random.poisson(0.40 * 0.8)
    pkts_q4 = np.random.poisson(0.45 * 0.8)
    if np.random.rand() < burst_prob:
        if np.random.rand() < 0.5:
            pkts_q3 += np.random.poisson(burst_scale)
        else:
            pkts_q4 += np.random.poisson(burst_scale)
    return {3: pkts_q3, 4: pkts_q4}, packet_size_bytes


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
        reward_scale=0.01,
    ):
        self.n_queues = n_queues
        self.n_routers = n_routers
        self.n_nodes = n_nodes
        self.service_rate = service_rate
        self.max_queue_len = max_queue_len
        self.episode_length = episode_length
        self.burst_prob = burst_prob
        self.burst_scale = burst_scale
        self.reward_scale = reward_scale

        # Base arrival rates for 8 traffic classes (VoIP, Video, Web, Bulk, etc.)
        if arrival_rates is not None:
            self.arrival_rates = np.array(arrival_rates, dtype=np.float64)
        else:
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
            self.drop_costs = np.array([50.0, 40.0, 30.0, 25.0, 20.0, 18.0, 15.0, 12.0], dtype=np.float64)

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
        """
        Generates host traffic from Node 1 and Node 2 using the 5 specialized traffic functions:
        1. Voice (Q1)
        2. Video (Q2)
        3. Web (Q5)
        4. Bulk (Q7)
        5. Bursty (Q3, Q4)
        Plus baseline Network Control (Q0) and Best Effort (Q6).
        """
        arrivals = np.zeros(self.n_queues, dtype=np.float64)

        # 1. Voice traffic (Q1 - Voice EF)
        v_pkts, _ = generate_voice_traffic(self.arrival_rates[1])
        arrivals[1] += v_pkts

        # 2. Video traffic (Q2 - Interactive Video AF4)
        vid_pkts, _ = generate_video_traffic(self.arrival_rates[2])
        arrivals[2] += vid_pkts

        # 3. Web traffic (Q5 - Standard Data AF1)
        web_pkts, _ = generate_web_traffic(self.arrival_rates[5])
        arrivals[5] += web_pkts

        # 4. Bulk traffic (Q7 - Background / Bulk)
        bulk_pkts, _ = generate_bulk_traffic(self.arrival_rates[7])
        arrivals[7] += bulk_pkts

        # 5. Bursty / Flash Crowd traffic (Q3 - Streaming AF3 & Q4 - Critical Data AF2)
        burst_dict, _ = generate_bursty_traffic(self.arrival_rates[4], self.burst_prob, self.burst_scale)
        arrivals[3] += burst_dict[3]
        arrivals[4] += burst_dict[4]

        # Background Network Control (Q0) & Best Effort (Q6)
        arrivals[0] += np.random.poisson(self.arrival_rates[0])
        arrivals[6] += np.random.poisson(self.arrival_rates[6])

        # Partition traffic offered by Host Node 1 (60%) and Host Node 2 (40%)
        arrivals_node1 = arrivals * 0.6
        arrivals_node2 = arrivals * 0.4
        return arrivals_node1 + arrivals_node2

    def step(self, action):
        """
        Executes one timestep slot:
        1. Host traffic arrives at Ingress Router 1.
        2. Router 1 services packets using WFQ baseline.
        3. Served packets from Router 1 flow into Core Bottleneck Router 2 + cross traffic.
        4. Core Router 2 services packets according to action weights [w0..w7] using work-conserving WFQ.
        5. Serviced packets leave Core Router 2 to Destination Nodes 3 & 4.
        """
        # 1. Host Arrivals to Router 1
        host_arrivals = self._generate_host_traffic()
        r1_drops = self.router1.receive_packets(host_arrivals)

        # Router 1 static WFQ weights
        r1_weights = np.array([0.22, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05])
        r1_served = self.router1.service_queues(r1_weights)

        # 2. Router 1 forwarding + cross-traffic arrives at Core Router 2
        cross_traffic = np.random.poisson(self.arrival_rates * 0.15)
        r2_arrivals = r1_served + cross_traffic
        r2_drops = self.router2.receive_packets(r2_arrivals)

        # 3. Core Router 2 allocates bandwidth using work-conserving WFQ
        r2_served = self.router2.service_queues(action)

        # 4. Total packet drops
        step_drops = r1_drops.sum() + r2_drops.sum()
        self.total_drops += step_drops

        # 5. Dual cost: holding cost + drop cost
        cost_r1 = self.router1.get_cost(r1_drops)
        cost_r2 = self.router2.get_cost(r2_drops)
        raw_cost = cost_r2 + 0.3 * cost_r1

        # Scaled reward for stable RL gradients
        reward = -raw_cost * self.reward_scale

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
            "raw_cost": raw_cost,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """Normalized state vector of the bottleneck router [0, 1]."""
        return (self.router2.queues.copy() / self.max_queue_len).astype(np.float32)
