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

## Track A: selectable S/M clustering benchmark

Both optimization notebooks have a first code cell with:

```python
INPUT_FILE = "seeds_s.json"  # Change to "seeds_m.json" for M.
OUTPUT_DIRECTORY = "track_a/outputs/seeds_s_comparison_full"
```

For M, also change the directory to `"track_a/outputs/seeds_m_comparison_full"`.
Use the same selection in classical and quantum; results selects both saved
directories with `SAVED_OUTPUTS`. Bare filenames resolve
under `track_a/inputs/`; repository-relative and absolute input paths also work.
An explicit `OUTPUT_DIRECTORY` may be absolute or repository-relative.

| Tier | Customers | Vehicles | Capacity | Demands | QUBO bits | Emulator |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| S | 4 | 2 | 7 | 1-4 | 14 | `emu-sv` |
| M | 10 | 3 | 20 | 1-8 | 45 | `emu-mps` |

When the callable input helper receives `OUTPUT_DIRECTORY=None`, the default S
file uses its legacy `track_a/outputs/all_demands/` directory and M uses
`track_a/outputs/seeds_m/`; other files use
`track_a/outputs/<input-file-stem>/`. Saved input manifests reject attempts to
reuse a directory for different demands or geometry. L remains unsupported.

If you edit the demands, number of demand rows, or coordinates after running,
choose a fresh `OUTPUT_DIRECTORY` in all three notebooks, for example
`"track_a/outputs/seeds_s_comparison"`. Then rerun classical, quantum, and results.
This preserves the previous benchmark instead of mixing incompatible evidence.

**Every demand vector** in the input is processed: currently three instances,
`s-0` through `s-2` for S or `m-0` through `m-2` for M.
The notebooks and callable experiment stages use the same selection.
The fixed S example, `[2, 3, 4, 2]`, has a detailed hand-check in both notebooks:
all candidate groupings, loads, pair costs and QUBO/slack arithmetic.
The classical and quantum notebooks focus only on clustering. Customer-coordinate
tables are omitted; coordinates remain in the input to define pair costs.
The required full-CVRP reference and fixed-group routing analysis are available
through the explicit Python evaluation stages (Track A, challenge section 9.2).
The results notebook displays matching saved CVRP gaps only; it never computes tours.
S includes demand `1` in `s-2`; its supplied values span the allowed range
1-4 and are not all permutations of the same multiset.

The quantum notebook displays the exact target-QUBO ground states and
the lowest-energy strings actually sampled, including counts, feasibility and
assignment/slack bit splits. It explicitly reports when no exact target ground
state was sampled. The full bitstring evidence is saved under
the selected output directory's `groundstates/` folder. S validates all 16,384
bitstrings. M uses the analytic penalty bound, checks every feasible distinct
partition, and constructs all optimal vehicle relabelings; it never enumerates
all `2**45` strings. Validation records state which method was used.
The notebook also maps every bit position to its variable and shows each
vehicle's slack group and decoded unused-capacity value: weights 1, 2, 4 for S
or 1, 2, 4, 8, 16 for M.
Classical and quantum comparison tables use the same `customer_labels` and
`canonical_assignment_bits` columns: `[0, 1, 1, 0]` and `10010110` describe the
same assignment, not competing solutions. Vehicle labels are normalized so
equivalent vehicle swaps compare equally. Raw quantum assignment/slack bits
remain unchanged and separately labeled. The quantum notebook compares the
classical exact answer with the best feasible quantum sample for every instance;
infeasible samples have no decoded comparison assignment.

### Additional slack-free QUBO comparison

Both optimization notebooks also run **unbalanced penalization**, retaining all
original methods. With vehicle headroom `h_k = Q - sum_j d_j*x_jk`, it minimizes
the same normalized compactness objective plus one-hot assignment squares and
`sum_k(-lambda1*h_k + lambda2*h_k**2)`. It uses only assignment variables:
**S 8 bits instead of 14; M 30 instead of 45**. S still uses `emu-sv`, M `emu-mps`.

Set the same parameters in the classical and quantum notebooks:

```python
unbalanced_lambda1 = None
unbalanced_lambda2 = None
unbalanced_assignment_penalty = None
```

`None` reads the saved tier presets in
[unbalanced_penalties.json](track_a/inputs/unbalanced_penalties.json):
**S: lambda1=0.2, lambda2=0.01** and **M: lambda1=0.75, lambda2=0.02**.
Both notebooks print the effective weights and accept independent numeric
overrides. The assignment penalty remains unchanged.

Measured S comparison against the old **1.0, 0.1** weights:

| Check | Old weights | New S weights |
| --- | ---: | ---: |
| Instances whose exact QUBO ground states are all feasible constrained optima | 1/3 | 3/3 |
| SA feasible shots, averaged over seeds 3, 4, 5 | 48.31% | 88.76% |
| SA optimal hits, averaged over seeds 3, 4, 5 | 14.84% | 27.99% |
| Quantum optimal hits, seed 0, 256 shots per instance | 2/768 | 7/768 |
| Quantum feasible shots, same run settings | 14/768 | 10/768 |

The SA sweep used seeds 0, 1, 2 before checking seeds 3, 4, 5; all use 400
steps and 256 independent chains per instance. Quantum candidates used the
existing 1000/2000 ns pulse calibration and unchanged embedding/emulator
settings; old and selected weights both chose 2000 ns. The new weights found
an optimal feasible quantum sample on each S instance, whereas the old weights
missed it on the first. Quantum feasibility did **not** improve overall, and
these small shot counts do not establish a statistically robust gain.
This is tuning on benchmark instances, not held-out instance validation.

For M, 80 pairs were searched on the current three M vectors using seeds 0-2,
400 SA steps and 256 chains per instance. The five training leaders were checked
on seeds 3-5; the selected pair minimizes mean best feasible clustering gap in
those checks. Seeds 6-11 were then used for confirmation, not reselection:

| M check | Old 1.0, 0.1 | Saved 0.75, 0.02 |
| --- | ---: | ---: |
| Mean best gap, selection seeds 3-5 | 15.72% | 7.07% |
| Mean best gap, confirmation seeds 6-11 | 10.95% | 10.50% |
| Feasibility, confirmation seeds 6-11 | 72.40% | 59.05% |

This preset prioritizes clustering quality, not maximum feasibility, and is
best-tested rather than globally optimal. Confirmation showed only a small
mean-gap gain, with worse feasibility and worst-case gap. Baseline and selected
MPS checks at bond dimension 64, precision `1e-5`, 10 ns steps and a 1000 ns
pulse both exceeded a 60-second check limit; quantum improvement is not
established. A lower-cost screen (bond dimension 4, precision `1e-4`, 20 ns steps)
completed for both pairs but found zero feasible shots out of 256 each on the
first M instance; it does not establish convergence. Complete grid measurements, input records, reference costs, seeds
and MPS statuses are saved in
[m_unbalanced_search.json](track_a/outputs/penalty_tuning/m_unbalanced_search.json).

These weights are not a general guarantee of feasibility or optimal clustering.
`None` for `unbalanced_assignment_penalty` uses the original certificate-based numeric assignment weight,
but the original slack-QUBO proof does **not** apply to this formulation.
The quadratic headroom penalty can favor different feasible groupings and can
produce infeasible ground states. All raw samples are checked against the
original constraints, with no warm start or repair.

The additional methods are `classical-unbalanced-sa` (the same raw Metropolis
annealer) and `quantum-unbalanced` (a separate embedding and pulse calibration).
Their runs have distinct filenames and encoding metadata, so existing slack
results are not overwritten. Detailed unbalanced samples go under `samples/`;
separate quantum calibration, embeddings and sequences go under `unbalanced/`.
Results tables and CSV include all three formulations. Compare feasible
`model_cost`, gaps and feasibility rates, **not target energies across
formulations**. Tables explicitly distinguish no feasible samples from failed
or resource-limited runs. Original exact constrained references remain unchanged.

### Photographed variable-reduction QUBO

Both optimization notebooks also retain the photo's literal formula:

```text
H_v = F/s + alpha * sum_k(load_k/Q - 1 - b_k)^2
          + beta * sum_j(sum_k x_jk - 1)^2,  b_k in {0,1}.
```

Despite its name, this retains **one binary slack per vehicle**: **10 S bits,
33 M bits**. Its minus sign permits zero capacity penalty at load `Q` or `2Q`,
so it is not an exact capacity-inequality encoding. The reversible substitution
`b_k = 1-z_k` gives `(load_k/Q + z_k - 2)^2` with nonnegative pair couplings,
allowing the same transformed target in classical and quantum runs. Saved
samples record both physical auxiliary bits and logical binary slack values.
Original assignment/capacity checks remain independent, without repair.

Set `variable_reduction_alpha` and `variable_reduction_beta` in either notebook.
Each defaults to `1 + certificate_cost/s`: an **untuned starting weight**, not
a ground-state feasibility guarantee. The new methods are
`classical-variable_reduction-sa` and `quantum-variable_reduction`.
Their evidence has separate filenames, and quantum calibration, embeddings
and sequences use `variable_reduction/`. S uses `emu-sv`, M `emu-mps`.
Existing slack runs and S/M unbalanced presets are preserved.

### Saved-results comparison and visual exports

Run the classical and quantum notebooks to produce solver evidence. Then run
the results notebook as a **read-only saved-results viewer** with the `.venv`
kernel. It loads both S and M output folders, compares all six QUBO methods
(three formulations times classical/quantum),
and shows coverage, shot-probability bars, best-gap dot charts, per-instance
costs, side-by-side chosen-cluster maps, recorded weights, runtimes, and matching cached CVRP gaps.
Set `CLUSTER_INSTANCES` to choose saved instances (`None` shows all). Maps use
saved coordinates, customer IDs/demands and vehicle loads, with unavailable
assignments shown in gray rather than fabricated. Every comparison chart and
cluster map is also exported as a 200-DPI PNG to `track_a/outputs/result_visuals/`;
`VISUALS_DIRECTORY` overrides that path. Stable descriptive filenames are reused
on reruns. The notebook never
invokes a solver, emulator, reference search, or routing algorithm, and does
not rewrite benchmark evidence.

The results parameter cell selects previous runs, not current input files:

```python
SAVED_OUTPUTS = {
    "S": "track_a/outputs/seeds_s_comparison_full",
    "M": "track_a/outputs/seeds_m_comparison_full",
}
seed = 0
```

Saved manifests define the instances even if seeds files have changed.
The fresh comparison folders contain runs for all six QUBO methods on the
current three instances per tier. Any absent solver files in another selected
folder are explicitly shown as **Not saved**. No current preset
is retroactively substituted for the weights recorded in an older run.
Lightweight statistics are derived from saved raw counts; no sampling or
optimization is repeated. Exact routing evaluation and four-page PDF generation
remain explicit callable stages shown below, outside this read-only notebook.
There is no command-line runner.

The default is 256 raw shots per stochastic method, sampling seed 0, and
per-emulation limits of 180 seconds / 4096 MB address space.
The quantum parameter cell also exposes `dt_ns`, `mps_max_bond_dim` (default 64)
and `mps_precision` (default `1e-5`). M always uses `emu-mps`, not a large
state vector. MPS bond/truncation and timestep convergence are not established;
resource limits may prevent a run from completing. Qubit-order optimization is
disabled so raw strings preserve the documented variable order.
All execution is local and requires no credentials. The quantum phase can
take several minutes. Resource failures are recorded, not replaced by
fabricated success metrics. No classical repair or cloud/QPU jobs are used.

Edit [track_a/inputs/seeds_s.json](track_a/inputs/seeds_s.json) or
[track_a/inputs/seeds_m.json](track_a/inputs/seeds_m.json) to specify the
fleet parameters, allowed demand range, customer coordinates, depot, and demand vectors:

```json
{
  "N": 4,
  "K": 2,
  "Q": 7,
  "demand_range": [1, 4],
  "depot": [50, 50],
  "coordinates": [[10, 10], [20, 30], [60, 70], [80, 90]],
  "demands": [[2, 3, 4, 2], [4, 2, 2, 3]]
}
```

Each demand row defines an instance (`s-0`, `s-1`, ... or `m-0`, `m-1`, ...). `coordinates[i]` is the
`[x, y]` location of customer `i`, corresponding to entry `i` in each demand
vector. Coordinates are shared across vectors and read directly from the JSON,
never regenerated. The depot remains fixed at `(50, 50)`. The input needs no
`certificate`, `permutation` or `attempts` fields.
An internal feasibility witness is computed and independently checked for
validation and penalty bounds; it is **not** a solver warm start. Arbitrary
feasible S/M vectors are accepted, not just permutations of one multiset.
Malformed and capacity-infeasible inputs are rejected explicitly.

Default S outputs for all supplied instances are saved in
[track_a/outputs/all_demands/](track_a/outputs/all_demands/): references, validation
evidence, configurations, geometry, atom layouts, complete interaction matrices,
serialized sequences, raw shot counts, exact evaluation tours,
`results.csv`, and `report.pdf`. Separate references and gaps are reported for
pairwise clustering compactness and full CVRP. The deterministic constrained
baseline repeats one result to the shot budget and is labeled accordingly.
Physical embedding is approximate; its errors are reported separately from
target-QUBO correctness.

Earlier outputs in the parent `outputs/` folder and `outputs/first_demand/` are
left untouched. Results uses only the explicitly selected per-tier directories.
The optimization stages save six methods when all baselines are run; the
results viewer focuses on the six QUBO methods and their saved exact references.
Each formulation's pulse calibration uses the first selected instance;
the selected controls are then frozen for every instance.

The former runner's functionality remains available as ordinary Python functions.
For example, from a notebook or Python session at the repository root:

```python
from track_a.common.instances import notebook_inputs
from track_a.common.experiments import (
    prepare_experiment, classical_runs, quantum_runs, evaluation_references,
)
from track_a.common.analysis import summarize
from track_a.common.report import write_report

input_path, output, _ = notebook_inputs("seeds_s.json")
instances = prepare_experiment(output, input_path)
classical_runs(instances, output, shots=256, seed=0)
quantum_runs(instances, output, shots=256, seed=0, timeout=180, memory_mb=4096,
             mps_max_bond_dim=64, mps_precision=1e-5)
evaluation_references(instances, output)
rows = summarize(output)
write_report(output, rows)
```

To explicitly compute routing references/evaluations and a PDF from existing
shots without rerunning sampling, call `evaluation_references`, `summarize`,
and `write_report`. To inspect existing results without any optimization or
routing, rerun only the results notebook. For changed demands or geometry,
pass a separate `Path` as `output`;
`prepare_experiment(output, input_path=...)` accepts an alternate input file.
The output-directory check rejects mismatched inputs before experiments run.

Read and run the notebooks in order; core mathematical walkthroughs stay inline,
while experiment bookkeeping, statistics, evidence writers and plotting are shared:
[classical](track_a/classical_capacity_constrained_clustering.ipynb),
[quantum](track_a/quantum_capacity_constrained_clustering.ipynb), and
[results](track_a/results.ipynb), plus the complete
[model derivations](track_a/docs/MODEL.md) and [original plan](track_a/docs/PLAN.md).
Only the three notebooks sit directly in `track_a/`; supporting files are
grouped into `common/` (Python helpers and callable experiment stages), `docs/` (write-ups),
`inputs/` (seed JSON), and `outputs/` (saved evidence).
The notebooks use short explanations and small examples. The classical notebook
enumerates distinct feasible S/M assignments, but no routing. S has an independent
CPLEX target-QUBO cross-check; M uses an independent CPLEX set-partition check.
The quantum notebook derives penalties, shows bit/slack decoding, checks every
S target bitstring (or validates M using the penalty proof), then constructs the
atom layout and pulse. Pulse selection uses
clustering metrics only. The results notebook reads existing evidence for both
tiers, with missing runs and stale routing caches made explicit. The separate
Python evaluation API computes exact subset/Held-Karp tours and full-CVRP
references only when explicitly called. Missing references are never generated
by the saved-results viewer.
Input validation, file
utilities, emulator protection and PDF layout remain shared. The isolated emulator
worker is internal infrastructure for timeout/memory enforcement, not a user-facing
runner. Both notebook and Python experiment calls write exact and sampled
ground-state evidence. Evaluation caches are invalidated when runs, references,
evaluation versions, routing algorithms or hit thresholds change. Recorded
annealing, embedding and emulator settings are the values passed to execution;
annealing-specific configuration is saved only for the annealing baseline.

**Submission scope:** this is an S/M-capable local baseline, not a complete Track A
submission. Feasibility witnesses are independently checked. Inspect actual saved
statuses and raw shots to determine which quantum runs completed; supporting MPS
does not establish M-tier success or convergence. Inputs are explicit rows, not a
seed-based generator (A2).
The added unbalanced formulation removes slack variables; its penalty does
not inherit the original slack-QUBO feasibility proof.
No automated test suite is included.
The plan's scope note records the tier/input-schema/presentation revisions.
[requirements.txt](requirements.txt) is pinned and hash-locked, exported from
[uv.lock](uv.lock) with
`uv export --locked --no-dev --no-emit-project --format requirements-txt --output-file requirements.txt`.
AI implementation assistance is disclosed in the results notebook and report.

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
