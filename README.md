# VQR-DQN: Variational Quantum Rainbow Deep Q-Network for Optimizing Resource Allocation Problem

[![SAC '26](https://img.shields.io/badge/ACM%20SAC-2026-blue)](https://doi.org/10.1145/3748522.3779769)
[![arXiv](https://img.shields.io/badge/arXiv-2512.05946-b31b1b)](https://arxiv.org/abs/2512.05946)
[![TensorFlow Quantum](https://img.shields.io/badge/TensorFlow%20Quantum-Cirq-orange)](https://www.tensorflow.org/quantum)

VQR-DQN is a hybrid quantum–classical deep reinforcement learning framework that integrates ring-topology variational quantum circuits (VQCs) into the Rainbow DQN pipeline to solve the Human Resource Allocation Problem (HRAP).

This repository accompanies the paper published in the *Quantum Software Engineering Practices (QSE)* track of **The 41st ACM/SIGAPP Symposium on Applied Computing (SAC '26)**, Thessaloniki, Greece.

Given a set of officers, events, and tasks, VQR-DQN:

1. encodes officer capabilities, event occurrence times, and travel times into a state vector;
2. extracts classical features with noisy dense layers;
3. maps those features to rotation angles of a ring-topology VQC;
4. reads out Pauli-Z expectation values as a quantum feature vector;
5. fuses classical and quantum features;
6. estimates a dueling distributional (C51) Q-value distribution for each officer; and
7. assigns officers to tasks one decision at a time to minimize the maximum event completion time (makespan).

On four HRAP benchmarks, VQR-DQN achieves up to **26.8% normalized makespan reduction** over a random baseline and outperforms Double DQN and classical Rainbow DQN.

---

## Requirements

* Python **3.9–3.11**
* TensorFlow **2.15** with **TensorFlow Quantum 0.7.3**
* Cirq, SymPy, NumPy, tqdm
* Jupyter (optional, for circuit visualization)

TensorFlow Quantum is pinned to specific TensorFlow versions. Use a dedicated environment to avoid conflicts.

---

## Installation

```bash
git clone https://github.com/JayNguyen0505-es/VQR.git
cd VQR

python -m venv .venv
source .venv/bin/activate

pip install tensorflow==2.15.0 tensorflow-quantum==0.7.3
pip install cirq sympy numpy tqdm jupyter
```

Verify the quantum stack:

```bash
python -c "import tensorflow_quantum as tfq, cirq; print(tfq.__version__, cirq.__version__)"
```

---

## Problem: Human Resource Allocation (HRAP)

HRAP is formulated as a Markov decision process `(S, A, P, R, γ)` over three entities:

| Entity   | Symbol | Description                                                  |
| -------- | ------ | ------------------------------------------------------------ |
| Officers | `O`    | Personnel who can be dispatched to perform tasks             |
| Events   | `E`    | Scheduled incidents, each requiring all of its tasks         |
| Tasks    | `T`    | Sequential tasks within each event                           |

**Objective:** assign officers to tasks so that the **maximum completion time across all events** is minimized, accounting for both task execution and travel between events.

### State

The observation is the concatenation of:

```text
Officer capability matrices   C_o ∈ Z^{E×T}     time for officer o to complete task t of event e, sampled from [1, 20)
Event occurrence times        Ω   ∈ Z^{E}       start time of each event, sampled from [1, 20) and sorted
Transition matrix             M   ∈ Z^{(E+1)×(E+1)}  symmetric travel times, zero diagonal (extra node = base station)
```

```text
s = Concat(Flatten(C_1, ..., C_O), Ω, Flatten(M))
d = O·E·T + E + (E+1)²
```

During an episode, the agent also appends the partially filled assignment table (`E·T` entries, `-1` for unassigned), so the network input size is `d + E·T`.

### Action

The full assignment space has size `O^(E×T)`. The agent decomposes it into `E×T` sequential decisions, each selecting one of `O` officers for the next task.

### Reward

```text
Ψ   = max(C)·E·T + max(M)·E·T
r_t = − makespan / Ψ
```

Normalizing by the worst-case completion time `Ψ` keeps rewards in a consistent range across configurations.

### Simulation

`Environment.step(assignment)` runs a discrete-event simulation of an `E × T` assignment matrix:

* officers start at the base station and travel between events using `M`;
* a task's duration is the officer's capability plus travel time;
* if the assigned officer is busy, the event waits until that officer is free;
* simultaneous events are resolved with priority `Running > Idle > Completed`.

It returns the end time of every event; the makespan is their maximum.

---

## Method: VQR-DQN

```text
State s
  │
  ▼
NoisyDense(256, ReLU) → NoisyDense(256, ReLU) ─────────────────┐
  │                                                            │
  ▼                                                            │
Dense(2·n_q·n_l, tanh)          angle encoding                 │
  │                                                            │
  ▼                                                            │
Ring-topology VQC  →  ⟨Z_1⟩, ..., ⟨Z_nq⟩   quantum features     │
  │                                                            │
  ▼                                                            ▼
Concatenate(classical, quantum) → NoisyDense(256, ReLU)
  │
  ▼
Dueling distributional head (C51): value + advantage streams
  │
  ▼
p(s, a) over 51 atoms  →  Q(s, a) = Σ z_i · p_i(s, a)
```

### Variational Quantum Circuit

Each layer applies parameterized single-qubit rotations followed by CNOTs in a **ring**:

```text
|0⟩ ─ H ─ RX(θ0) ─ RZ(θ4) ─ ● ─────────── X ─ RX(θ8)  ─ RZ(θ12) ─ ...
|0⟩ ─ H ─ RX(θ1) ─ RZ(θ5) ─ X ─ ● ─────── │ ─ RX(θ9)  ─ RZ(θ13) ─ ...
|0⟩ ─ H ─ RX(θ2) ─ RZ(θ6) ───── X ─ ● ─── │ ─ RX(θ10) ─ RZ(θ14) ─ ...
|0⟩ ─ H ─ RX(θ3) ─ RZ(θ7) ───────── X ─── ● ─ RX(θ11) ─ RZ(θ15) ─ ...
```

* **Initialization:** Hadamard on every qubit to create an equal superposition.
* **Rotations:** `RX(θ)` and `RZ(θ)` per qubit per layer, giving `2 · n_q · n_l` angles.
* **Entanglement:** `CNOT(q_i → q_{(i+1) mod n_q})` for every qubit, closing the ring.
* **Readout:** Pauli-Z expectation on each qubit, yielding a feature vector in `[-1, 1]^{n_q}`.

The rotation angles are produced by a trainable `tanh` encoding layer, so the circuit acts as a learned, entangled feature extractor inside the Q-network. Circuits are simulated with TensorFlow Quantum's `Expectation` layer.

### Rainbow DQN Components

| Component                         | Implementation                                                |
| --------------------------------- | ------------------------------------------------------------- |
| Noisy networks                    | `NoisyDense` with trainable `μ` and `σ` for weights and biases |
| Prioritized experience replay     | `Memory_Prioritized` with `α` prioritization and annealed `β`  |
| n-step returns                    | `NStepBuffer`                                                 |
| Double DQN                        | Main network selects `a*`, target network evaluates it        |
| Dueling distributional Q-learning | `create_dueling_distributional_head` (C51 projection)         |
| Target network                    | Hard update every `update_target_interval` training steps     |

---

## Running

### Random baseline

```bash
python random_algorithm.py
```

Assigns officers uniformly at random and reports the average normalized reward.

### Classical Rainbow DQN

```bash
python rainbow.py
```

Rainbow DQN with a classical front-end (dense layers, batch normalization, residual connections, and multi-head self-attention).

### VQR-DQN

```bash
python qt.py
```

Training progress is printed every 10 episodes:

```text
Episode 1230, Reward: -0.3912, Avg: -0.4021, Eps: 0.54, Loss: 2.8143
```

At the end of training, the script reports the average reward per 1,000 episodes and the average return over the last 100 episodes.

### Changing the configuration

The problem size is set in the `__main__` block of each script:

```python
NUM_OFFICERS = 3
NUM_EVENTS = 2
NUM_TASKS = 2
EPISODES = 50000
```

The four configurations in the paper are:

| Config.    | Officers | Tasks | Events | Action space `O^(E×T)` |
| ---------- | -------: | ----: | -----: | ---------------------: |
| 3O-2T-2E   | 3        | 2     | 2      | 3⁴                     |
| 4O-3T-2E   | 4        | 3     | 2      | 4⁶                     |
| 4O-3T-3E   | 4        | 3     | 3      | 4⁹                     |
| 5O-4T-4E   | 5        | 4     | 4      | 5¹⁶                    |

### Default VQR-DQN hyperparameters

| Parameter                | Value           |
| ------------------------ | --------------- |
| Discount `γ`             | 0.99            |
| Learning rate            | 1e-5 (Adam, clipnorm 1.0) |
| Atoms / support          | 51 over `[-0.5, 0]` |
| n-step                   | 5               |
| PER `α`                  | 0.6             |
| PER `β`                  | 0.4 → 1.0 over 100,000 steps |
| Replay capacity          | 100,000         |
| Warm-up transitions      | 1,000           |
| Batch size               | 64              |
| `ε` schedule             | 1.0 → 0.1, decay 0.9995 per episode |
| Target update interval   | 1,000 steps     |
| Qubits / VQC layers      | 16 / 3          |
| Gradient clipping        | norm 10.0       |

---

## Circuit Visualization

`quantum_viz.ipynb` builds the ansatz, prints its statistics, renders it as SVG, and exports LaTeX (`qcircuit`) for papers.

```python
from qt import QuantumLayer

quantum_layer = QuantumLayer(n_qubits=4, n_layers=2)
circuit = quantum_layer.build_circuit()
print(quantum_layer.get_circuit_info())
```

```text
{'number_of_qubits': 4, 'number_of_layers': 2, 'number_of_parameters': 16,
 'circuit_depth': 13, 'circuit_moments': 13,
 'operation_types': {'HPowGate', 'Rx', 'Rz', 'CXPowGate'}}
```

```python
import cirq.contrib
print(cirq.contrib.circuit_to_latex_using_qcircuit(circuit))
```

---

## Results

All methods were trained for **50,000 episodes** under identical conditions, and the best checkpoint was evaluated on **200 test episodes**. Quantum circuits were simulated with TensorFlow Quantum, with computation run on the **IonQ Aria-1** QPU via the IonQ quantum computing service.

### Comparison with baselines

Averaged rewards (higher is better) and normalized makespan reduction over the random baseline:

| Config.  | Baseline | DDQN              | Rainbow DQN       | **VQR-DQN (Ours)**    |
| -------- | -------: | ----------------: | ----------------: | --------------------: |
| 3O-2T-2E | -0.5225  | -0.4539 (▲ 13.1%) | -0.4189 (▲ 19.8%) | **-0.3823 (▲ 26.8%)** |
| 4O-3T-2E | -0.3689  | -0.3132 (▲ 15.1%) | -0.2957 (▲ 19.8%) | **-0.2815 (▲ 23.7%)** |
| 4O-3T-3E | -0.3316  | -0.3032 (▲ 8.6%)  | -0.3012 (▲ 9.2%)  | **-0.2872 (▲ 13.4%)** |
| 5O-4T-4E | -0.2488  | -0.2366 (▲ 4.9%)  | -0.2309 (▲ 7.2%)  | **-0.2236 (▲ 10.1%)** |

### Impact of VQC topology (3O-2T-2E)

| Algorithm                | Reward                |
| ------------------------ | --------------------: |
| Baseline                 | -0.5225               |
| VQR-DQN + Linear         | -0.4249 (▲ 18.7%)     |
| VQR-DQN + Star           | -0.4514 (▲ 13.6%)     |
| **VQR-DQN + Ring**       | **-0.3823 (▲ 26.8%)** |
| VQR-DQN + All-to-All     | -0.4103 (▲ 21.5%)     |

The ordering **Ring > All-to-All ≈ Linear ≫ Star** is consistent with prior findings that ring connectivity yields the highest average expressibility and Meyer–Wallach entanglement for a given number of qubits and layers.

---

## Repository Structure

```text
env.py                 HRAP environment: Officers, Events, EventsWithPriority, Environment
qt.py                  VQR-DQN: QuantumLayer (ring VQC), NoisyDense, PER, n-step buffer, agent
rainbow.py             Classical Rainbow DQN baseline (attention + residual front-end)
random_algorithm.py    Random assignment baseline
quantum_viz.ipynb      Circuit statistics, SVG rendering, and LaTeX export
```

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@inproceedings{nguyen2026vqrdqn,
  author    = {Nguyen, Truong Thanh Hung and Nguyen, Truong Thinh and Cao, Hung},
  title     = {Variational Quantum Rainbow Deep Q-Network for Optimizing Resource Allocation Problem},
  booktitle = {Proceedings of the 41st ACM/SIGAPP Symposium on Applied Computing},
  series    = {SAC '26},
  year      = {2026},
  location  = {Thessaloniki, Greece},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  isbn      = {979-8-4007-2294-3},
  doi       = {10.1145/3748522.3779769},
  url       = {https://doi.org/10.1145/3748522.3779769}
}
```

* Paper (ACM DL): <https://doi.org/10.1145/3748522.3779769>
* Preprint (arXiv): <https://arxiv.org/abs/2512.05946>
* Official lab implementation: <https://github.com/Analytics-Everywhere-Lab/qtrl>

---

## Acknowledgement

This work was supported by NSERC Discovery Grant No. RGPIN 2025-00129.

Developed at the [Analytics Everywhere Lab](https://www.cs.unb.ca/~hcao3/), University of New Brunswick, Canada, in collaboration with the University of Science and Technology of Hanoi (USTH), Vietnam.
