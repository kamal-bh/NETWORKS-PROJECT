# Research Paper Deep Dive & Project Preview Viva Guide

**Paper Title:** *Deep Reinforcement Learning-based Dynamic Bandwidth Allocation in Weighted Fair Queues of Routers*  
**Authors:** Jinyan Pan, Gang Chen, Haoran Wu, Xi Peng (Huawei Technologies Theory Lab), Li Xia  
**Published in:** IEEE 18th International Conference on Automation Science and Engineering (IEEE CASE 2022)  
**Affiliations:** Sun Yat-Sen University, Guangzhou University, Huawei Technologies Co. Ltd.  

---

## 1. Executive Summary

Modern high-speed Internet routers experience bursty, unpredictable, and heterogeneous traffic. Traditionally, routers use static scheduling algorithms like **Weighted Fair Queueing (WFQ)** or **Deficit Round Robin (DRR)** with fixed weights, or rule-based heuristics like the **$c\mu$-rule** and **Longest Connected Queue (LCQ)**. These classical methods either ignore instantaneous queue spikes, cause buffer overflows, or depend on simplistic Poisson arrival assumptions that fail in real-world deployments.

This paper formulates the router output port as a **$G/G/1/K$ parallel queueing system** and introduces **WFQ-DRL**, a model-free Deep Reinforcement Learning framework. By observing only instantaneous queue lengths, an RL controller dynamically adjusts continuous bandwidth weights ($w_1, \dots, w_n$ summing to $1$) to minimize both **queueing delay** (via Little's Law) and **packet loss rate**. The authors deploy and compare two state-of-the-art DRL algorithms—**DDPG** (Deep Deterministic Policy Gradient) and **SAC** (Soft Actor-Critic)—demonstrating superior performance over classical rules on synthetic non-Poisson traffic and real production traffic traces from Huawei routers.

---

## 2. The Problem Statement

### 2.1 The Router Output Port Bottleneck
When packets travel through the Internet, internal router switching fabrics direct packets to destination output ports. Because multiple incoming streams converge on a single outbound link with finite bandwidth ($b$), packets must be queued in buffers:
* Traffic flows have **differentiated Quality of Service (QoS)** requirements (e.g., VoIP and gaming need ultra-low latency; video streaming needs high throughput; bulk file downloads tolerate delay).
* Router output ports organize packets into multiple parallel FIFO queues (typically 8 priority queues in commercial routers).

### 2.2 Why Traditional Scheduling Mechanisms Fail
1. **Priority Queueing (PQ):** Always serves the highest-priority queue first.  
   * *Fatal flaw:* High-priority traffic completely **starves** lower-priority queues.
2. **Static Weighted Fair Queueing (WFQ) / Deficit Round Robin (DRR):** Allocates fixed bandwidth shares (e.g., 50%-30%-20%).  
   * *Fatal flaw:* **Rigid and state-blind.** If high-priority queues are temporarily idle, static allocation wastes service opportunities; if a sudden burst arrives on queue 2, static weights cannot adapt, leading to buffer overflow and severe packet drops.
3. **Traditional Queueing Theory (M/M/1, M/G/1, etc.):**
   * *The Poisson Trap:* Assumes arrivals are memoryless (Poisson) and service times are exponential. Real Internet traffic is **self-similar, long-range dependent, and highly bursty** (non-Poisson).
   * *Curse of Dimensionality:* In a $G/G/1/K$ parallel queueing system with $n$ queues and buffer size $K$, the state space has size $\prod_{i=1}^n (K_i + 1)$. Inverting transition matrices or solving dynamic programming Bellman equations becomes computationally impossible.
4. **Huawei's "Ten Mathematical Challenges" (2019):**
   * Challenge #3 explicitly asked: *How do we model Internet traffic without Poisson assumptions and devise an optimal control policy to improve queueing performance in routers under limited bandwidth?*

### 2.3 Shortcomings of Classical Heuristic Rules
* **The $c\mu$-rule:** Serves the queue with the highest $c_i \mu_i$ (holding cost $\times$ service rate). Proven mathematically optimal **only** for infinite buffers ($K=\infty$), linear costs, and event-by-event switching. It completely fails under finite buffers because it ignores packet loss penalties and traffic burstiness.
* **Longest Connected Queue (LCQ):** Allocates all service capacity to the largest queue. Proven optimal **only** when arrival rates and packet size statistics across all queues are strictly symmetric/identical. In real heterogeneous routers, LCQ performs poorly.

---

## 3. How the Paper Solves It: The WFQ-DRL Framework

### 3.1 Mathematical Formulation as a Markov Decision Process (MDP)
The router output port is modeled as a tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} \rangle$:

```
Incoming Flows (λ_i) ---> [ Queue 1 (K1) ] ───(w1 * b)───┐
                          [ Queue 2 (K2) ] ───(w2 * b)───┼──> Shared Outbound Link (b)
                          [ Queue n (Kn) ] ───(wn * b)───┘
                                 │
                         Observe Queue Lengths (q)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   DRL Controller      │
                     │     (DDPG / SAC)      │
                     └───────────────────────┘
                                 │
                     Outputs Bandwidth Weights
                     w = [w1, w2, ..., wn],  Σ w_i = 1
```

#### 1. State Space ($\mathcal{S}$):
* Observed vector of current queue lengths: $q = (q_1, q_2, \dots, q_n)$, where $q_i \le K_i$.
* *Engineering normalization:* Converted to binary vector or normalized float ($q_i / K_i \in [0, 1]$) to eliminate gradient explosion across heterogeneous buffer sizes.

#### 2. Action Space ($\mathcal{A}$):
* Continuous bandwidth weight vector:
  $$\mathcal{A} = \left\{ a = (w_1, w_2, \dots, w_n) \in \mathbb{R}^n_+ \;\middle|\; \sum_{i=1}^n w_i = 1 \right\}$$
* Service rate allocated to queue $i$ is $\mu_i = w_i \times b$, where $b$ is the total physical bandwidth.

#### 3. Reward Function ($\mathcal{R}$):
* Dual-objective penalty minimizing both delay and drops:
  $$r(q, a) = - \sum_{i=1}^n (h_i \cdot q_i + c_i \cdot l_i)$$
  * $q_i$: Number of packets in queue $i$. By **Little's Law** ($L = \lambda W$), minimizing average queue length directly minimizes average queueing delay.
  * $h_i$: Holding cost per unit time for queue $i$ (differentiates QoS priority; higher $h_i$ for delay-sensitive queues).
  * $l_i$: Number of dropped packets in queue $i$ due to buffer overflow ($q_i > K_i$).
  * $c_i$: Penalty cost per dropped packet.

#### 4. Decision Timing Modes:
* **State-dependent:** Decisions triggered at every arrival/departure event. (Theoretical benchmark; infeasible in hardware due to CPU overhead).
* **Slot-dependent:** Decisions made periodically every $\Delta t$ time slot (e.g., every $200\ \mu s$ in production or $20\ s$ in macroscopic tests). This reflects real router ASIC/control-plane architectures.

---

### 3.2 The Two DRL Algorithms

The paper implements **off-policy** actor-critic algorithms utilizing an **Experience Replay Buffer** $\mathcal{D} = \{(s_t, a_t, r_{t+1}, s_{t+1})\}$ to break temporal correlation and maximize sample efficiency.

```
+---------------------------------------------------------------------------------------+
| Feature              | DDPG (Deep Deterministic Policy Gradient) | SAC (Soft Actor-Critic)             |
+----------------------+-------------------------------------------+-------------------------------------+
| Policy Type          | Deterministic: a = μ(s|φ)                 | Stochastic: a ~ π(·|s) (Gaussian)   |
| Action Exploration   | External noise (Ornstein-Uhlenbeck)       | Maximum Entropy Objective (H(π))    |
| Critics              | Single Critic Q(s, a|θ)                   | Twin Critics min(Q1, Q2)            |
| Overestimation Bias  | Prone to overestimation                   | Protected via clipped double-Q      |
| Sample Efficiency    | Very high; direct gradient                | High; balanced exploration          |
| Hyperparameter Tuning| Sensitive to noise tuning                 | Self-tuning temperature (α)         |
+---------------------------------------------------------------------------------------+
```

#### DDPG (Deep Deterministic Policy Gradient)
* **Actor $\mu_\phi(s)$:** Directly outputs continuous weights $w$ passed through a Softmax layer to guarantee $\sum w_i = 1$.
* **Critic $Q_\theta(s, a)$:** Evaluates expected return using Bellman error minimization:
  $$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma Q_{\bar{\theta}}(s', \mu_{\bar{\phi}}(s')) - Q_\theta(s, a) \right)^2 \right]$$
* **Target Networks:** Soft Polyak updates $\theta' \leftarrow \tau \theta + (1-\tau)\theta'$.
* **Exploration:** Ornstein-Uhlenbeck (OU) temporally correlated noise added to actions.

#### SAC (Soft Actor-Critic)
* **Maximum Entropy RL:** Augmented objective balances maximizing reward and maximizing policy entropy:
  $$J(\pi) = \sum_t \mathbb{E} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]$$
  This prevents the policy from prematurely collapsing to sub-optimal deterministic extremes.
* **Reparameterization Trick:** Samples action $z \sim \mathcal{N}(\mu(s), \sigma(s))$, then transforms via Sigmoid/Softmax so gradients backpropagate smoothly.
* **Twin Critics:** Uses $\min(Q_{\theta_1}, Q_{\theta_2})$ to eliminate the positive value overestimation bias that plagues continuous actor-critic algorithms.
* **Auto-Tuned Alpha ($\alpha$):** Dynamically adjusts entropy temperature.

---

### 3.3 Key Experimental Findings from the Paper
1. **2 Queues (State-dependent):** DDPG quickly converged to the exact performance of the mathematically optimal $c\mu$-rule, proving the RL agent discovers optimal scheduling without knowing arrival distributions.
2. **2 Queues (Heavy Load $\rho=0.9$, Gamma non-Poisson arrivals, finite buffer $K=32$):** Both DDPG and SAC significantly outperformed static baselines. SAC converged faster and demonstrated smoother variance due to entropy exploration.
3. **8 Queues (Standard Router Output Port):** Both algorithms successfully scaled to 8 queues without suffering from the curse of dimensionality. DDPG achieved slightly better asymptotic reward due to direct deterministic gradient efficiency.
4. **Real Production Traces from Huawei Routers:** Tested on 8 heterogeneous queues with microsecond-level time slots ($\Delta t = 200\ \mu s$, link speed $8\ \text{bytes}/\mu s$). Both DDPG and SAC comprehensively outperformed the **Longest Connected Queue (LCQ)** baseline, maintaining minimal delay and zero packet loss during sudden traffic surges.

---

## 4. Mapping the Research Paper to Your Project Codebase

Your project workspace directly mirrors the architecture, models, and evaluation methodologies introduced in the paper:

```
Paper Concept / Formulation                  Your Project File & Implementation
────────────────────────────────────────────────────────────────────────────────────────
G/G/1/K Router Simulation Environment    -->  queue_env.py (QueueEnv)
                                              - Implements 2 queues with Poisson/bursty arrivals
                                              - Service rate b = 1.5, finite buffer K = 50
                                              - Differentiated holding costs: h = [3.0, 1.0]
                                              - Packet drop logic and reward: -(h1*q1 + h2*q2)

Classical Benchmark Policies             -->  baselines.py & run_baselines.py
                                              - cmu_rule(): Greedy allocation to high-cost queue
                                              - lcq_rule(): Allocates to longest queue
                                              - fixed_split_rule(): Static 50/50 WFQ allocation
                                              - Generates: baseline_comparison.png

DDPG Implementation (Paper Alg. 2)       -->  ddpg_agent.py & train_ddpg.py
                                              - Actor (Linear 64->64->Softmax)
                                              - Critic (State+Action concatenated -> 64 -> 1)
                                              - OUNoise (Ornstein-Uhlenbeck exploration)
                                              - ReplayBuffer (10,000 capacity)
                                              - Polyak target network updates (tau = 0.005)
                                              - Generates: reward_curve.png

SAC Implementation (Paper Alg. 3)        -->  sac_agent.py
                                              - Gaussian SACActorNetwork with reparameterization
                                              - Sigmoid mapping for simplex constraint: w2 = 1 - w1
                                              - Twin SACCriticNetwork (critic1 & critic2)
                                              - Automatic entropy temperature alpha optimization

Comparative Benchmarking                 -->  compare_agents.py & demo.py
                                              - Trains DDPG vs. SAC head-to-head
                                              - Evaluates against c-mu and LCQ across random seeds
                                              - Generates: ddpg_vs_sac.png and final_demo.png
```

---

## 5. Comprehensive Project Preview / Viva Q&A Guide

Prepare for your project preview and external reviews with these questions and model answers, categorized by panel focus.

### Category 1: Networking & Fundamentals

#### Q1: What is the fundamental problem in a router output port?
> **Answer:** An output port receives packets destined for an outbound physical link. Because incoming traffic is bursty, heterogeneous, and shares a link with finite bandwidth, buffers fill up. If queues are mismanaged, delay-sensitive traffic experiences high latency, and finite buffers overflow, causing packet drops. The challenge is dynamically distributing total bandwidth among competing queues to minimize both delay and packet loss.

#### Q2: Why not just use Priority Queueing (PQ) or static Weighted Fair Queueing (WFQ)?
> **Answer:**  
> * **Priority Queueing (PQ)** strictly serves high-priority queues first. If high-priority traffic is sustained, lower-priority queues experience **complete starvation**.  
> * **Static WFQ / Deficit Round Robin (DRR)** guarantees fairness by assigning fixed weights (e.g., 50/50), preventing starvation. However, it is **state-blind**: if queue 1 is empty, static rules underutilize bandwidth; if queue 2 receives a sudden burst, static weights cannot adapt, causing buffer overflow and high latency.

#### Q3: How does Little's Law justify your reward function?
> **Answer:** Little's Law states:
> $$L = \lambda W$$
> where $L$ is the average number of packets in the queue (queue length), $\lambda$ is average arrival rate, and $W$ is average waiting time (delay). Assuming arrival rates are fixed by external users, **delay ($W$) is directly proportional to queue length ($L$)**. Therefore, penalizing queue lengths $-(h_1 q_1 + h_2 q_2)$ directly minimizes packet queueing delay.

#### Q4: Why can't classical queueing theory (M/M/1, M/G/1) solve this analytically?
> **Answer:**  
> 1. Classical solutions assume memoryless Poisson arrivals, whereas real Internet traffic exhibits burstiness, heavy tails, and long-range autocorrelation.  
> 2. Real routers have finite buffers ($K$) and packet drops ($G/G/1/K$).  
> 3. Calculating the optimal dynamic policy analytically requires inverting high-dimensional transition matrices. With multiple queues and realistic buffer sizes, this faces the **curse of dimensionality** ($|S| = \prod(K_i+1)$), making closed-form analytical solutions mathematically intractable.

---

### Category 2: Problem Formulation & Environment

#### Q5: How do you formulate bandwidth allocation as an MDP?
> **Answer:**  
> * **State ($S$):** Normalized vector of instantaneous queue lengths: $s = [q_1/K_1, \dots, q_n/K_n]$.  
> * **Action ($A$):** Continuous bandwidth allocation weights $a = [w_1, \dots, w_n]$ satisfying $w_i \ge 0$ and $\sum w_i = 1$.  
> * **Transition Probability ($P$):** Determined by real-world packet arrivals, service completions, and buffer drops. Model-free RL learns this implicitly.  
> * **Reward ($R$):** Negative cost combining weighted queue lengths and drop counts:
>   $$r = -\sum (h_i q_i + c_i l_i)$$

#### Q6: Why did you choose holding costs $h = [3.0, 1.0]$ in your simulation?
> **Answer:** Holding costs represent differentiated QoS tiers. Queue 0 has $h_0 = 3.0$ (representing high-priority, latency-sensitive traffic such as video conferencing or VoIP), while Queue 1 has $h_1 = 1.0$ (best-effort bulk data transfer). A 1-packet buildup in Queue 0 incurs three times the penalty of a 1-packet buildup in Queue 1, incentivizing the agent to drain Queue 0 aggressively while preventing Queue 1 from overflowing.

#### Q7: What is the difference between state-dependent and slot-dependent control? Why is slot-dependent preferred in real routers?
> **Answer:**  
> * **State-dependent control** recomputes actions on every single packet arrival or departure. While fine-grained, modern routers process millions of packets per second (100 Gbps+ line rate). Running a neural network inference on every packet arrival would trigger massive CPU interrupt storms and high packet latency.  
> * **Slot-dependent control** makes decisions periodically at fixed time intervals $\Delta t$ (e.g., $200\ \mu s$). The controller sets weights into hardware registers, and the router hardware (ASIC/FPGA) schedules packets using WFQ/DRR at line rate during that slot.

---

### Category 3: Reinforcement Learning Architecture

#### Q8: Why use continuous action space algorithms (DDPG / SAC) instead of DQN?
> **Answer:** DQN requires discrete actions. In bandwidth allocation, dividing bandwidth among $N$ queues into discrete slices (e.g., 5% increments across 8 queues) leads to an exponential combinatorial explosion of discrete actions. Continuous actor-critic methods output real-valued weight vectors directly and naturally enforce constraints via Softmax or Sigmoid functions.

#### Q9: How do your networks enforce the constraint that bandwidth weights must be positive and sum to 1 ($\sum w_i = 1$)?
> **Answer:**  
> * In **DDPG** (`ddpg_agent.py`), the output layer of the Actor network uses a **Softmax** activation:
>   $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
>   This guarantees $\forall w_i > 0$ and $\sum w_i = 1.0$.  
> * In **SAC** (`sac_agent.py`), for 2 queues, the network outputs a single continuous latent variable $z$. We apply the **Sigmoid** activation:
>   $$w_1 = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad w_2 = 1.0 - w_1$$
>   This guarantees bounded simplex allocation without violating probability constraints.

#### Q10: How does SAC differ from DDPG, and why does SAC achieve more stable training?
> **Answer:**  
> 1. **Stochastic vs. Deterministic:** DDPG learns a deterministic policy $a = \mu(s)$ and relies on external heuristic noise (Ornstein-Uhlenbeck) for exploration. SAC learns a Gaussian distribution $\pi(a|s)$ and explores systematically.  
> 2. **Maximum Entropy Formulation:** SAC maximizes expected reward plus entropy: $J(\pi) = \mathbb{E}[r + \alpha \mathcal{H}(\pi)]$. This prevents premature policy collapse to suboptimal local minima.  
> 3. **Twin Critics (Clipped Double-Q):** DDPG uses a single Critic network, suffering from value overestimation bias. SAC maintains two Critics ($Q_1, Q_2$) and takes $\min(Q_1, Q_2)$ when computing target values.  
> 4. **Automatic Temperature Tuning:** SAC automatically tunes the entropy coefficient $\alpha$ via gradient descent.

#### Q11: What is the purpose of the Replay Buffer and Target Networks?
> **Answer:**  
> * **Replay Buffer:** Consecutive transitions in a router are strongly autocorrelated (a burst at time $t$ continues at $t+1$). Randomly sampling mini-batches from a replay buffer breaks temporal correlation, satisfying the i.i.d. assumption necessary for stable neural network training.  
> * **Target Networks:** Deep Q-learning without target networks creates a "moving target" problem where updating weights changes both current and target values simultaneously, leading to divergence. Target networks maintain slowly updated weights ($\tau = 0.005$) using Polyak averaging to stabilize updates.

---

### Category 4: Results, Real-World Deployment & Defense Questions

#### Q12: Under what conditions is the $c\mu$-rule optimal, and how does your AI compare against it?
> **Answer:** The $c\mu$-rule is mathematically optimal **only** when buffers are infinite ($K=\infty$), arrival statistics are stationary, costs are linear, and decisions occur at every event. In our simulation experiments with random bursts and finite buffers, our AI agents (DDPG and SAC) match the $c\mu$-rule baseline under ideal conditions and **outperform** it during discrete burst spikes because the neural network learns non-linear adjustments to prevent buffer overflows before they happen.

#### Q13: Can a Deep Neural Network actually run inside a real 100 Gbps router? (The Line-Rate Feasibility Question)
> **Answer:** **Yes, using a split Control Plane / Data Plane architecture:**  
> * **Data Plane (Fast Path - Nanoseconds):** Physical ASICs or FPGAs execute standard Deficit Round Robin (DRR) or Weighted Fair Queueing (WFQ) at line rate using hardware registers.  
> * **Control Plane (Slow Path - Microseconds/Milliseconds):** The lightweight DRL agent (a 2-layer MLP with 64 neurons) runs on the router's embedded CPU or smartNIC. Every time slot $\Delta t$ (e.g., $200\ \mu s$), the agent reads current queue lengths from hardware registers, executes a forward pass ($< 10\ \mu s$ inference time), and writes updated weight values ($w_1, \dots, w_n$) to the hardware registers. The fast path schedules packets according to these updated weights.

#### Q14: What happens if the router experiences unseen traffic patterns or a DDoS attack?
> **Answer:**  
> 1. Because the state vector only depends on queue lengths ($q_i / K_i$) and not external packet headers, the agent remains responsive to physical congestion regardless of traffic source.  
> 2. If a massive DDoS burst fills a queue to capacity ($K_i$), the agent balances allocating bandwidth to drain the burst while preserving minimum bandwidth for other queues based on the reward function.  
> 3. As a safety guarantee, router firmware can enforce hard minimum bandwidth floors (e.g., $w_i \ge 0.05$) to ensure zero starvation even under adversarial edge cases.

#### Q15: What are the key limitations of the paper and your project?
> **Answer:**  
> 1. **Single-Node Optimization:** The model optimizes a single router output port. In end-to-end networks, optimizing one router might shift congestion to downstream routers. A future improvement is Multi-Agent Reinforcement Learning (MARL).  
> 2. **Scalability Beyond 8 Queues:** While 8 queues reflect standard router hardware (DiffServ standard), systems with hundreds of virtual queues would require hierarchical or distributed agents.

---

## 6. Key Formulas Reference Sheet

| Formula Name | Mathematical Equation | Meaning in Project |
|---|---|---|
| **Little's Law** | $L = \lambda W$ | Connects mean queue length ($L$) to mean waiting time ($W$). Minimizing queue length minimizes delay. |
| **Simplex Constraint** | $\sum_{i=1}^n w_i = 1, \quad w_i \ge 0$ | Ensures allocated bandwidth does not exceed physical link capacity $b$. |
| **Multi-Objective Reward** | $r(q, a) = - \sum_{i=1}^n (h_i q_i + c_i l_i)$ | Penalizes holding cost (delay) and packet drop count (buffer overflow). |
| **DDPG Bellman Loss** | $\mathcal{L}(\theta) = \mathbb{E}[(r + \gamma Q_{\bar{\theta}}(s', \mu_{\bar{\phi}}(s')) - Q_\theta(s, a))^2]$ | Mean squared error for training the Critic value network. |
| **Deterministic Policy Gradient** | $\nabla_\phi J \approx \mathbb{E}[\nabla_a Q(s, a)\vert_{a=\mu(s)} \cdot \nabla_\phi \mu_\phi(s)]$ | Updates the Actor network to maximize Critic score. |
| **SAC Entropy-Augmented Return** | $J(\pi) = \mathbb{E}[\sum_t \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot\vert s_t)))]$ | Encourages exploration and prevents premature policy saturation. |
| **Twin-Critic Target** | $y = r + \gamma \min_{j=1,2} Q_{\bar{\theta}_j}(s', a') - \alpha \log \pi(a'\vert s')$ | Mitigates overestimation bias in SAC value estimation. |
| **Polyak Averaging (Soft Update)**| $\theta_{\text{target}} \leftarrow \tau \theta + (1-\tau) \theta_{\text{target}}$ | Smoothly tracks trained weights to maintain learning stability ($\tau = 0.005$). |

---

## 7. Recommended 2-Minute Presentation Pitch

> "Good morning. Our project tackles a fundamental problem proposed by Huawei in their 'Ten Mathematical Challenges': **Dynamic Bandwidth Allocation in Routers under non-Poisson traffic.**
> 
> Modern routers use static scheduling algorithms like Weighted Fair Queueing with fixed weights. But real Internet traffic has unpredictable, bursty surges. If you use static rules, high-priority packets suffer from delay spikes, and finite buffers overflow, causing packet drops. Traditional queueing theory fails to solve this because non-Poisson traffic and high-dimensional buffers suffer from the curse of dimensionality.
> 
> Inspired by the IEEE CASE 2022 paper, we implemented **WFQ-DRL**—a model-free Deep Reinforcement Learning framework. We built a custom router simulation environment in Python and implemented two continuous actor-critic agents from scratch: **DDPG** and **Soft Actor-Critic (SAC)**.
> 
> Our agent observes only instantaneous queue lengths and dynamically adjusts continuous bandwidth weights to minimize both delay and packet loss. As shown in our experimental results, our RL agents not only converge to the theoretical gold-standard $c\mu$-rule, but under bursty traffic conditions, they learn nuanced predictive strategies that outperform classical human-engineered heuristics. Finally, because our controller operates on a slotted schedule, it can be deployed directly into the control plane of commercial routers without modifying fast-path line-rate hardware."
