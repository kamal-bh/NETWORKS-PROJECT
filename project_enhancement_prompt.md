# Prompt: Upgrade the Router Queue-Management Project (Multi-Host, Burst Traffic, Record-and-Replay, DDPG/SAC vs Classic Policies)

You are a senior engineer working on an **existing** project that simulates bandwidth allocation across queues and compares scheduling policies and RL agents. I want to extend it into a realistic, defensible, demo-ready system that I can present to my professor with full clarity.

**Step 0 (before writing any code):** Read the existing project completely. Summarize in a short note what already exists (environment, state/action spaces, reward, policies, agents, plotting) and which parts you will reuse, refactor, or replace. Then give me a milestone plan. Keep the existing code working wherever possible; do not rewrite things that already work.

---

## 1. Goal in one paragraph

Build a simulated network with **5 hosts connected through a single router**. The router has **multiple outgoing (egress) queues**, one per traffic class. Only this one router's egress side is observed and controlled. I manually generate bursty traffic with keyboard keys (any host to any host, different traffic types, different intervals). The traffic is **recorded first**, then **replayed identically** against every policy and agent so that the comparison is fair. Two RL agents (**DDPG** and **SAC**) are trained for a fixed time (default 5 minutes) and then evaluated against the classic policies (**c-mu rule, Longest Connected Queue (LCQ), 50/50 split**). Results are shown periodically as text during the run and as graphs at the end.

---

## 2. Simulation environment

### 2.1 Topology
- 5 hosts (H1..H5), each connected to **one router R** by a link. Hosts are in a simple topology (star is fine; make the topology a config structure so it can be changed).
- Each host link has configurable bandwidth, propagation delay, and buffer.
- The router has a **fixed total egress capacity** C (configurable, e.g. 100 Mbps equivalent or packets/tick). This capacity is what the policies/agents divide among the egress queues.
- Traffic goes host to router to destination host. Any host can send to any other host.

### 2.2 Router egress queues
- The router keeps **K egress queues**, one per traffic class (default K = 4, see section 4). Each queue has: max capacity (finite buffer), FIFO order, drop-tail on overflow.
- Each tick, a **scheduler/agent chooses an allocation vector** (the bandwidth share of each queue, summing to 1, or equivalently to total capacity C). The router then serves packets from each queue according to that share.
- Track per packet: arrival time, departure time (delay), size, class, source, destination, and whether it was dropped.

### 2.3 Time model
- Discrete-time simulation with a fixed tick (e.g. 10 ms or 100 ms of simulated time, configurable). The simulation is **deterministic given a trace and a seed**. No dependence on wall-clock time inside the engine.
- Define a "decision interval" (the agent acts every N ticks) and a "reporting interval" (metrics aggregated and printed every M ticks).

### 2.4 State, action, reward (must be identical for every policy)
- **State** (what agents see): per-queue length (normalized), per-queue arrival rate estimate over a recent window, per-queue head-of-line delay, per-queue drop count in the last interval, previous allocation. Document the exact vector and normalization.
- **Action**: allocation vector over K queues on the simplex (softmax output for agents, or direct for classic policies).
- **Reward / penalty**: one function used by **all** policies for evaluation, e.g.
  `penalty = w_q * sum(queue_len_i / cap_i) + w_d * mean_delay + w_drop * drops (+ optional fairness / starvation term)`, `reward = -penalty`.
  Weights are configurable and documented. Per-class cost weights `c_i` must be consistent with the c-mu rule (see section 5). Explain every term and why it is there.

---

## 3. Traffic engine (the most important part; build and verify this first)

### 3.1 Traffic types (at least 6)
Implement each as a separate generator class with parameters and a clear mathematical description in the docstring:
1. **Poisson burst**: Poisson arrivals with a high rate lambda_high for the burst duration.
2. **Normal (Gaussian) burst**: rate follows a bell-shaped profile over the burst window (ramps up, peaks, ramps down).
3. **Random burst**: random rate levels with random switching times (uniform random rate in [a, b], changing at random intervals).
4. **On-Off / MMPP burst**: Markov-modulated source alternating between ON (high rate) and OFF (low or zero rate) with exponential holding times.
5. **Heavy-tailed (Pareto) burst**: Pareto-distributed burst sizes / inter-burst gaps, producing self-similar-like traffic.
6. **Constant-bit-rate spike / flash crowd**: sudden step to a constant high rate, then a sharp drop (or a linear ramp up for the flash-crowd variant).

Each generator emits packets with: timestamp, source host, destination host, size, and a **traffic class** (see section 4).

### 3.2 Manual keyboard control
- Keys map to traffic types, for example: `1` Poisson, `2` Normal, `3` Random, `4` On-Off, `5` Pareto, `6` CBR spike. Make the mapping a config table and print it as a help screen at start.
- On a key press, generate a burst of that type for a **configurable duration** (default a few seconds of simulated time).
- **Source/destination selection**: default is random src and dst (different hosts) per burst; support a mode where several hosts send simultaneously, and an optional key sequence/prefix to pick specific hosts (for example `h` then `2` then `4`). Document which modes exist.
- Utility keys: `s` stop and finish recording, `p` pause/resume recording, `u` undo last burst, `r` toggle fully-random auto mode (random type, host pair, and interval every few seconds, for long training data), `?` help.
- Choose a keyboard library that works on the user's OS (for example `pynput`, `keyboard`, `curses`, or `readchar`) and provide a fallback (for example typing a key then Enter) if global key capture is unavailable. Tell me which you chose and why.

### 3.3 Record-then-replay (critical requirement)
- **Do not run the policies while I am generating traffic.** First, record my key-driven traffic into a **trace file** (CSV or JSON Lines): one row per packet (`t, src, dst, size, class, traffic_type, burst_id`). Also store metadata: seed, tick size, topology config, key log (which key, when).
- After I press stop, **replay the identical trace** through each policy/agent in turn, each on a **fresh copy of the same environment** with the same seed. This guarantees every policy sees exactly the same packets at exactly the same times.
- Traces are reusable: I can save, load, and combine them (`--trace file.csv`), and also generate a synthetic random trace without keyboard input.

### 3.4 Training data vs evaluation data (avoid unfair comparison)
- DDPG and SAC are trained for a fixed budget (default **5 minutes of wall-clock training time**, configurable, also configurable in episodes/steps) on **training traffic**: either a recorded training trace, an auto-random mode, or a mix.
- After training, agents are **frozen** (no learning, no exploration noise) and evaluated on the **recorded evaluation trace**, the same one used for the classic policies.
- Strongly prefer the evaluation trace to be **different** from the training traces (held-out), and say clearly in the report which was used. If I only provide one trace, split it by time (for example the first 70% for training episodes, the final 30% for evaluation) or train on auto-random traffic and evaluate on my recorded trace.
- Also support an option "continue learning during evaluation" as a separate, clearly labeled experiment, never mixed with the frozen comparison.

---

## 4. Packet classification

- Packets carry header-like fields: `src, dst, size, protocol/app_type, priority_tag (DSCP-like)`.
- Implement a **Classifier** component at the router ingress that maps each packet to one of K classes using these fields (rule table, not hard-coded by traffic type), for example:
  - Class 0: real-time (voice/VoIP; small packets, delay-sensitive)
  - Class 1: video/streaming
  - Class 2: interactive/web
  - Class 3: bulk/background (file transfer)
- Each class has: its own queue, a delay sensitivity, and a cost weight `c_i` and service rate `mu_i` used by the c-mu rule and the reward.
- Traffic generators should attach realistic header values (for example the CBR spike mostly produces video, Poisson burst a mix) so classification is meaningful. Log the classification result per packet and report a class-mix summary.

---

## 5. Policies and agents to compare (all through one common interface)

All implement `allocate(state) -> allocation_vector` and are run by the same evaluation harness.

1. **50/50 split baseline**: equal share across queues (for K queues, 1/K each; with 2 active classes, 50/50). Document the exact definition used.
2. **c-mu rule**: serve queues by priority index `c_i * mu_i` (cost of delay times service rate); in allocation form, favor the queue with the highest index among non-empty queues (state clearly whether you implement strict priority or a proportional version, and why).
3. **LCQ (Longest Connected Queue)**: allocate to the queue with the longest backlog (state the exact tie-break and the allocation form).
4. **DDPG agent**: actor-critic, deterministic policy, replay buffer, target networks, exploration noise during training only. Actor output goes through softmax to produce the simplex allocation.
5. **SAC agent**: maximum-entropy actor-critic with twin critics, automatic temperature tuning if feasible, same action parametrization. Use the stochastic policy for training and its **deterministic (mean) action for evaluation**.

Rules for fairness: same environment code, same reward function, same trace, same seed, same decision interval, same metrics for all five. Keep the agent hyperparameters in config files and log them.

Be honest in the output: **report the measured results as they are.** Do not tune the reward or evaluation to force the agents to win. If the agents do not yet beat the classic policies, show it, and then diagnose (reward scaling, training length, state design) and improve legitimately. My expectation (penalty trending down and ending below static policies) is a hypothesis to test, not something to fake.

---

## 6. Metrics and results output

### 6.1 Periodic on-screen reporting (during each replay)
Every reporting interval (for example every 1 s of simulated time), print a compact status block per policy:
- time, active traffic type(s) at this moment
- **bandwidth distribution**: allocation share of each queue (as numbers and a small text bar)
- **queue lengths** per class (and occupancy %)
- **instantaneous reward/penalty** and running average
- drops and mean delay in the last interval

Since replays run one policy after another, print a header per policy, and optionally a final side-by-side table.

### 6.2 Final graphs (plotted at the end, after all replays)
All graphs share the same x-axis (time intervals) and use one consistent color per policy:
1. Penalty (or reward) vs time for all 5 policies, plus the cumulative penalty.
2. Queue length vs time for each class, one subplot per class, all policies overlaid.
3. Bandwidth allocation vs time per policy (stacked area), so the allocation behavior is visible.
4. Drops and mean/95th-percentile delay per class per policy (bar chart).
5. DDPG and SAC **training curves** (episode reward vs training progress), shown separately from the evaluation comparison.
6. A traffic timeline showing which traffic types were injected when (from the key log), so the professor can see why queues grew.
Also output a **summary table** (CSV and printed): total penalty, mean queue length, drops, mean delay, fairness index, per policy. Optionally add confidence via multiple seeds if time allows.

### 6.3 Metrics correctness
Add automated checks: packet conservation (arrived = served + dropped + still queued), allocation sums to 1, queue never negative or above capacity, identical packet streams across policies (hash the trace), determinism (same seed gives same result).

---

## 7. Control flow of a full run

1. Start: print help, config summary, chosen topology.
2. **Record phase**: I press keys; traffic is generated and written to the trace. Show a minimal live log of which burst was added (this is only a log, no policy runs).
3. I press the stop key: recording ends, trace saved.
4. **Training phase** (if not already trained/loaded): train DDPG and SAC for the time budget on the training traffic; show training progress periodically; save checkpoints.
5. **Evaluation phase**: replay the recorded trace through 50/50, c-mu, LCQ, DDPG (frozen), SAC (frozen), one after another, each with periodic status output.
6. **Report phase**: draw the graphs, save PNGs and the summary CSV, print the comparison table.
The user can stop recording at any time, and a stop key during training must save a checkpoint and move on cleanly.

---

## 8. Priorities and order of work

**Important: make the traffic engine and metrics correct before spending any time on dashboard animations.**

- Milestone 1: environment with 5 hosts, router, K classified queues, deterministic tick engine, metrics, conservation tests.
- Milestone 2: traffic generators (all types), trace format, record and replay, keyboard control.
- Milestone 3: classic policies on replayed traces, periodic text reporting, final graphs (so a complete pipeline works end to end without RL).
- Milestone 4: DDPG and SAC integrated and trained, frozen evaluation, comparison.
- Milestone 5 (only after 1-4 are verified): optional polish such as live plots, animated dashboard, nicer topology drawing.

After each milestone, run the tests, show me the output, and wait or continue only if everything passes.

---

## 9. Deliverables

- Working code with a clear structure (`env/`, `traffic/`, `policies/`, `agents/`, `metrics/`, `plots/`, `configs/`, `tests/`) and a single entry point with CLI flags (`--record`, `--trace`, `--train-minutes`, `--seed`, `--eval-only`).
- `README.md`: install, run, key map, config options.
- `EXPLANATION.md` written for a professor audience: system model, each traffic type with its math, classification, reward function and weights with justification, each policy's logic, the RL algorithms in plain language, the fairness protocol (record-then-replay, same seed, held-out evaluation), how to read each graph, and known limitations.
- Saved trace files, trained model checkpoints, result CSV, and all graphs.
- A short "demo script": the exact sequence of keys to press and what the professor should see.

## 10. Style requirements
- Prefer clarity over cleverness; comment the reasoning, not the syntax.
- Every configurable number lives in config, nothing magic in the code.
- Realism with honesty: state every simplification (for example packets modeled as fixed-time service units, no TCP dynamics) in the explanation.
- Ask me a question only if something blocks you; otherwise choose a sensible default, state it, and continue.
