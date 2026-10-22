import numpy as np
from queue_env import (
    generate_voice_traffic,
    generate_video_traffic,
    generate_web_traffic,
    generate_bulk_traffic,
    generate_bursty_traffic,
)

class TrafficScenario:
    """Represents a defined traffic scenario with specific arrival statistics and dynamics."""
    def __init__(
        self,
        scenario_id,
        name,
        split,
        description,
        arrival_rates,
        burst_prob=0.15,
        burst_scale=2.5,
        service_rate=3.2,
        ingress_service_rate=4.0,
        custom_traffic_fn=None,
    ):
        self.scenario_id = scenario_id
        self.name = name
        self.split = split  # 'train' or 'test'
        self.description = description
        self.arrival_rates = np.array(arrival_rates, dtype=np.float64)
        self.burst_prob = burst_prob
        self.burst_scale = burst_scale
        self.service_rate = service_rate
        self.ingress_service_rate = ingress_service_rate
        self.custom_traffic_fn = custom_traffic_fn

    def get_traffic_fn(self):
        """Returns a traffic generation function bound to this scenario."""
        if self.custom_traffic_fn is not None:
            return lambda env: self.custom_traffic_fn(env, self)

        def default_scenario_traffic(env):
            arrivals = np.zeros(env.n_queues, dtype=np.float64)

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

            # 5. Bursty traffic (Q3 - Streaming AF3 & Q4 - Critical Data AF2)
            burst_dict, _ = generate_bursty_traffic(self.arrival_rates[4], self.burst_prob, self.burst_scale)
            arrivals[3] += burst_dict[3]
            arrivals[4] += burst_dict[4]

            # Background Network Control (Q0) & Best Effort (Q6)
            arrivals[0] += np.random.poisson(self.arrival_rates[0])
            arrivals[6] += np.random.poisson(self.arrival_rates[6])

            # Two ingress hosts (60% / 40%)
            return arrivals * 0.6 + arrivals * 0.4

        return default_scenario_traffic


# =====================================================================
# Custom Scenario Generator Callables
# =====================================================================

def periodic_shock_pulses_traffic(env, scenario):
    """
    Test Scenario 4: Periodic Shock Pulses
    Alternates between a 20-tick lull (0.3x arrival rate) and a 10-tick severe shock burst (3.0x arrival rate).
    """
    cycle = env.step_count % 30
    if cycle >= 20:
        multiplier = 3.0
        burst_prob = 0.50
        burst_scale = 4.5
    else:
        multiplier = 0.35
        burst_prob = 0.05
        burst_scale = 1.0

    rates = scenario.arrival_rates * multiplier
    arrivals = np.zeros(env.n_queues, dtype=np.float64)

    v_pkts, _ = generate_voice_traffic(rates[1])
    arrivals[1] += v_pkts

    vid_pkts, _ = generate_video_traffic(rates[2])
    arrivals[2] += vid_pkts

    web_pkts, _ = generate_web_traffic(rates[5])
    arrivals[5] += web_pkts

    bulk_pkts, _ = generate_bulk_traffic(rates[7])
    arrivals[7] += bulk_pkts

    burst_dict, _ = generate_bursty_traffic(rates[4], burst_prob, burst_scale)
    arrivals[3] += burst_dict[3]
    arrivals[4] += burst_dict[4]

    arrivals[0] += np.random.poisson(rates[0])
    arrivals[6] += np.random.poisson(rates[6])

    return arrivals * 0.6 + arrivals * 0.4


# =====================================================================
# The 10 Scenarios: 5 Training & 5 Unseen Test
# =====================================================================

SCENARIOS = [
    # ------------------ 5 TRAINING SCENARIOS ------------------
    TrafficScenario(
        scenario_id=1,
        name="Train_1_Balanced_Standard",
        split="train",
        description="Standard DiffServ baseline mix at moderate utilization (~85%).",
        arrival_rates=[0.10, 0.20, 0.35, 0.40, 0.45, 0.50, 0.40, 0.20],
        burst_prob=0.15,
        burst_scale=2.5,
    ),
    TrafficScenario(
        scenario_id=2,
        name="Train_2_Multimedia_Heavy",
        split="train",
        description="High VoIP (Q1) and Interactive Video (Q2) demand with frequent I-frame surges (~98%).",
        arrival_rates=[0.10, 0.45, 0.70, 0.60, 0.30, 0.35, 0.20, 0.10],
        burst_prob=0.20,
        burst_scale=3.0,
    ),
    TrafficScenario(
        scenario_id=3,
        name="Train_3_Web_Transactional",
        split="train",
        description="Dense e-commerce transactions and HTTP traffic loading Q4 & Q5 (~92%).",
        arrival_rates=[0.10, 0.15, 0.25, 0.30, 0.70, 0.90, 0.30, 0.15],
        burst_prob=0.18,
        burst_scale=2.5,
    ),
    TrafficScenario(
        scenario_id=4,
        name="Train_4_Bulk_Transfer_Saturation",
        split="train",
        description="Heavy file transfers / backups heavily loading Best Effort Q6 & Background Q7 (~105%).",
        arrival_rates=[0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.70, 0.85],
        burst_prob=0.15,
        burst_scale=2.0,
    ),
    TrafficScenario(
        scenario_id=5,
        name="Train_5_Flash_Crowd_Bursts",
        split="train",
        description="Frequent high-magnitude burst spikes on Streaming Q3 and Critical Data Q4 (~112%).",
        arrival_rates=[0.10, 0.20, 0.35, 0.50, 0.55, 0.45, 0.40, 0.25],
        burst_prob=0.35,
        burst_scale=4.0,
    ),

    # ------------------ 5 UNSEEN TEST SCENARIOS ------------------
    TrafficScenario(
        scenario_id=6,
        name="Test_1_Extreme_Overload",
        split="test",
        description="System-wide overload (1.7x traffic across all classes, ~155% load); tests buffer collapse defense.",
        arrival_rates=[0.18, 0.36, 0.60, 0.68, 0.75, 0.85, 0.68, 0.36],
        burst_prob=0.25,
        burst_scale=3.5,
    ),
    TrafficScenario(
        scenario_id=7,
        name="Test_2_Emergency_VoIP_Surge",
        split="test",
        description="Disaster condition: Network Control (Q0) and Voice (Q1) surge 5x while bulk data is suppressed.",
        arrival_rates=[0.55, 0.95, 0.20, 0.15, 0.15, 0.20, 0.10, 0.05],
        burst_prob=0.10,
        burst_scale=2.0,
    ),
    TrafficScenario(
        scenario_id=8,
        name="Test_3_Off_Peak_Quiet",
        split="test",
        description="Light off-peak nocturnal traffic (~35% load); tests if policy wastes capacity or starves idle queues.",
        arrival_rates=[0.04, 0.08, 0.12, 0.14, 0.16, 0.18, 0.14, 0.08],
        burst_prob=0.04,
        burst_scale=1.2,
    ),
    TrafficScenario(
        scenario_id=9,
        name="Test_4_Periodic_Shock_Pulses",
        split="test",
        description="Non-stationary square-wave shock pulses (20-tick lull then 10-tick 3.0x shock burst); tests dynamic agility.",
        arrival_rates=[0.10, 0.20, 0.35, 0.40, 0.45, 0.50, 0.40, 0.20],
        custom_traffic_fn=periodic_shock_pulses_traffic,
    ),
    TrafficScenario(
        scenario_id=10,
        name="Test_5_Inverted_Priority_Load",
        split="test",
        description="Inverted distribution: 80%+ traffic is in low-priority Best Effort Q6 & Background Q7, high-priority idle.",
        arrival_rates=[0.02, 0.04, 0.06, 0.08, 0.15, 0.30, 1.15, 1.25],
        burst_prob=0.15,
        burst_scale=2.5,
    ),
]


def get_scenario(identifier):
    """Retrieve scenario by ID (1-10) or name."""
    for s in SCENARIOS:
        if s.scenario_id == identifier or s.name == identifier:
            return s
    raise ValueError(f"Scenario '{identifier}' not found in registered scenarios.")


def get_scenarios_by_split(split):
    """Retrieve all scenarios for a given split ('train' or 'test')."""
    return [s for s in SCENARIOS if s.split == split]
