# DuQuantum / Pasqal hackathon

## Setup

1. Install and start Docker, then install VS Code's **Dev Containers** extension.
2. Open this repository in VS Code and run **Dev Containers: Reopen in Container**.
3. Wait for setup to finish. Python 3.12 and the locked dependencies are installed
   automatically into `.venv`.
4. Open a notebook, select `.venv/bin/python` as the kernel, and run the cells
   from top to bottom.

The first build requires internet access. The pre-reading notebooks run locally
without credentials.

## Repository structure

| Path | Contents |
| --- | --- |
| [prereading/](prereading/) | Participant PDF and three local tutorial notebooks: Pulser programming, registers and Rydberg interactions, and QUBO Solver. |
| [pasqal_first_job.ipynb](pasqal_first_job.ipynb) | First cloud-emulator job; requires a Pasqal account and submits jobs when run. |
| [.devcontainer/](.devcontainer/) | Container image and VS Code setup. |
| [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) | Python dependencies and locked versions. |
| [.env_template](.env_template) | Cloud credential template; copy to `.env` for the cloud notebook. Never commit credentials. |

## Quick reference: neutral atoms and notebook terminology

### Atoms, qubits, and interactions

| Term | Meaning |
| --- | --- |
| **Neutral atom / atom array** | An atom with no net electric charge. Individual atoms can be trapped with focused laser beams (optical tweezers) and arranged into a programmable array. |
| **Qubit** | A quantum two-level system. In these Rydberg-drive examples, the levels are a ground state and a highly excited Rydberg state, measured as `0` and `1`, respectively. |
| **Ground state / Rydberg state** | The atom's low-energy reference state / a highly excited electronic state. Rydberg atoms interact strongly with one another, enabling collective quantum behavior. |
| **Superposition** | A quantum state combining `0` and `1` with probability amplitudes. Measurement returns one outcome, not both; the squared magnitudes of the amplitudes give the probabilities. |
| **Entanglement** | Correlations in a joint quantum state that cannot be described by assigning each atom its own independent quantum state. Interacting atoms can become entangled. |
| **Register** | The mapping from qubit identifiers to atom positions. Geometry matters: changing distances changes interaction strengths. |
| **Rydberg interaction, `U` or `V`** | The shift in energy when two atoms are both excited. In these examples it is repulsive and scales as `C6 / R^6`, where `R` is their separation and `C6` sets the interaction strength. Doubling `R` weakens the interaction by a factor of 64. |
| **Rydberg blockade** | Suppression of simultaneous excitation of nearby atoms when their interaction is much stronger than the resonant drive. It is not an absolute ban on adjacent `1`s. |
| **Blockade radius** | A characteristic distance where interaction strength equals the Rabi frequency: `Rb = (C6 / Omega)^(1/6)` in Pulser's units. It is not a sharp cutoff; atoms farther apart still interact. A stronger drive reduces this radius. |
| **Connectivity / blockade graph** | A simplified picture with atoms as vertices and edges joining pairs closer than a chosen blockade radius. In the register notebook, circles have half that radius, so overlapping circles indicate an edge. |

### Laser control and Pulser

| Term | Meaning |
| --- | --- |
| **Analog quantum computing** | Programming continuous controls and atom geometry so the system evolves under a chosen Hamiltonian, rather than specifying a list of discrete quantum gates. |
| **Hamiltonian** | The operator describing the system's energies and quantum evolution. Here it combines laser-driven transitions, detuning, and pairwise Rydberg interactions. |
| **Rabi frequency / amplitude, `Omega`** | The laser's drive strength for transitions between ground and Rydberg states. It sets the rate of resonant population oscillations; it is not the laser's optical frequency. |
| **Detuning, `delta` or `Delta`** | How far the drive is from the atomic transition frequency. Zero means resonance. With the convention used here, positive detuning lowers the Rydberg state's energy relative to the ground state, favoring excitation; negative detuning disfavors it. |
| **Phase, `phi`** | The drive's phase, which sets the direction of the qubit rotation in the equatorial plane of the Bloch sphere. The example pulses use phase `0`. |
| **Pulse / waveform** | A pulse specifies amplitude, detuning, and phase over a duration. A waveform describes how a control varies with time, such as a constant value or a linear ramp. |
| **Pi pulse** | A resonant pulse with integrated Rabi frequency equal to pi radians. Ideally, it transfers an isolated atom from ground to Rydberg state. The notebook uses `Omega = pi` rad/us for `1000` ns. |
| **Channel / global drive** | A device control path. A global Rydberg channel applies the same drive to all atoms in the register; it does not address just one atom. |
| **Sequence / device** | A `Sequence` schedules pulses on a register using a device's available channels and constraints. `AnalogDevice` supplies the device model used in these notebooks. |
| **Adiabatic sweep** | Changing controls slowly enough for the system to approximately follow an instantaneous energy eigenstate, often the ground state. Finite-duration sweeps can produce imperfect preparation. |
| **Antiferromagnetic / checkerboard pattern** | Alternating neighboring states, such as excited and unexcited atoms on a square array. Here repulsion discourages neighboring excitations, helping favor this pattern. |

**Units to watch:** Pulser uses **micrometers** (`um`) for positions,
**nanoseconds** (`ns`) for pulse durations, **radians per microsecond** (`rad/us`)
for amplitude, detuning, and interaction strength, and **radians** for phase.
`1000 ns = 1 us`; angular frequency is `2*pi` times ordinary frequency, so
`2*pi rad/us` corresponds to `1 MHz`. For a constant resonant pulse, the
rotation angle is `Omega * duration`, with duration converted to microseconds.
Pulser's `interaction_coeff` already includes the conversion from energy to
angular-frequency units; do not divide it by Planck's constant again.

### Measurement, optimization, and execution

| Term | Meaning |
| --- | --- |
| **Shot / run** | One repetition of state preparation and measurement. Cloud job parameter `runs` specifies how many shots to request. |
| **Bitstring / counts / probability** | A bitstring is one measured outcome or binary candidate, such as `110`. Counts record how often an outcome appears; counts divided by total shots estimate its probability. Interpret bit positions using the result's qubit or variable order. |
| **QUBO** | Quadratic Unconstrained Binary Optimization: minimize `x^T Q x` with each `x_i` either `0` or `1`. Constraints, when needed, are encoded as objective penalties rather than separate restrictions. |
| **Cost / objective / coupling** | Cost is the value being minimized: lower is better. Diagonal entries of `Q` reward or penalize selecting a variable; off-diagonal entries couple pairs. For symmetric `Q`, a pair contributes `2 * Q_ij * x_i * x_j` because both matrix entries appear in `x^T Q x`. |
| **Maximum independent set (MIS)** | The largest set of graph vertices with no edge between any selected pair. It motivates selecting many excited atoms while using blockade to discourage neighboring selections. A blockade graph is an approximation, not the full interaction model. |
| **Embedding / atom placement** | Mapping a problem's variables and couplings onto atom positions and controls. Arbitrary QUBO matrices cannot generally be reproduced exactly by a 2D array with distance-dependent `1/R^6` interactions. |
| **Heuristic / classical baseline** | A heuristic searches for good solutions without generally guaranteeing the optimum. The QUBO notebook compares a quantum-emulation workflow with CPU-based tabu search, a heuristic that discourages revisiting recent choices. |
| **Postprocessing** | Classical processing of sampled candidates, potentially improving or changing them. QUBO Solver's default results are therefore not necessarily raw emulator measurements; `probs` and `counts` describe the reported samples. |
| **Emulator / backend / QPU** | An emulator simulates quantum evolution on classical hardware; a backend is the execution interface; a QPU is physical quantum hardware. `QutipBackendV2` and `LocalEmulator` run locally here. `RemoteEmuFreeBackend` runs in Pasqal Cloud, not on a QPU. |
| **Job / batch** | A job requests execution with particular parameters and shot count; a batch groups jobs. The first-job notebook submits two parameter settings, requesting 20 and 50 shots. |

**Where to start:** [Pulser programming](prereading/01_programming_with_pulser.ipynb)
introduces pulses and state preparation;
[registers and interactions](prereading/02_register_and_rydberg_interactions.ipynb)
explains geometry and blockade;
[QUBO Solver basics](prereading/03_qubo_solver_basics.ipynb) connects these ideas
to optimization; and [the first cloud job](pasqal_first_job.ipynb) introduces
remote execution.
