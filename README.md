# Deep Reinforcement Learning for Real-World Router Bandwidth Allocation (WFQ-DRL)

An implementation of Deep Reinforcement Learning (DRL) algorithms—**DDPG** (Deep Deterministic Policy Gradient) and **SAC** (Soft Actor-Critic)—for dynamic bandwidth allocation in multi-router, multi-node network environments with **8 DiffServ / IEEE 802.1p priority queues**.

Based on the research paper:  
*Deep Reinforcement Learning-based Dynamic Bandwidth Allocation in Weighted Fair Queues of Routers* (IEEE CASE 2022).

---

## 🚀 Key Features

- **Multi-Node & Multi-Router Environment**: Simulates realistic network topology (Host sources/sinks, Ingress Router, Core Bottleneck Router) with packet forwarding, cross-traffic, and buffer backpressure.
- **8-Class Differentiated Services (DiffServ / IEEE 802.1p)**:
  - Q0: Network Control ($h=10.0, c=25.0$)
  - Q1: Voice / EF ($h=8.0, c=20.0$)
  - Q2: Interactive Video / AF4 ($h=6.0, c=15.0$)
  - Q3: Streaming Media / AF3 ($h=4.0, c=10.0$)
  - Q4: Critical Business Data / AF2 ($h=3.0, c=8.0$)
  - Q5: Standard Data / AF1 ($h=2.0, c=5.0$)
  - Q6: Best Effort ($h=1.0, c=2.5$)
  - Q7: Background / Scavenger ($h=0.5, c=1.0$)
- **Continuous Action Space on 8-Simplex**: Dynamic allocation of bandwidth weights $(w_0, \dots, w_7)$ with $\sum_{i=0}^7 w_i = 1.0$.
- **Advanced RL Algorithms**:
  - **DDPG**: Deterministic policy gradient with Ornstein-Uhlenbeck (OU) exploration noise.
  - **SAC**: Soft Actor-Critic with stochastic policy, entropy maximization, twin critics (clipped double-Q), and auto-tuned temperature $\alpha$.
- **Classical Rule Baselines**:
  - $c\mu$-rule (greedy delay reduction)
  - Strict Priority Queueing (PQ)
  - Static Weighted Fair Queueing (WFQ)
  - Longest Connected Queue (LCQ)

---

## 📁 Repository Structure

```
├── queue_env.py             # Multi-node, multi-router simulation environment (8 Queues)
├── ddpg_agent.py            # DDPG Actor, Critic, ReplayBuffer, and OUNoise
├── sac_agent.py             # SAC Actor (8-simplex Gaussian), Twin Critics, and Entropy Tuner
├── baselines.py             # Classical heuristics (c-mu, LCQ, Static WFQ, PQ)
├── run_baselines.py         # Evaluates all baseline policies and plots queue dynamics
├── train_ddpg.py            # Trains DDPG agent on the 8-queue environment
├── compare_agents.py        # Head-to-head training and evaluation: DDPG vs SAC vs Baselines
├── demo.py                  # Master presentation script with 3-panel summary
├── baseline_comparison.png  # Baseline queue trajectory visualizations
├── reward_curve.png         # DDPG learning curve
├── ddpg_vs_sac.png          # Comparison plot (DDPG vs SAC vs Baselines)
└── final_demo.png           # Master 3-panel presentation graph
```

---

## 🛠️ Usage

### 1. Run Baseline Evaluations
```bash
python run_baselines.py
```

### 2. Train DDPG Agent
```bash
python train_ddpg.py
```

### 3. Compare DDPG vs SAC Head-to-Head
```bash
python compare_agents.py
```

### 4. Run the Master Demo
```bash
python demo.py
```
