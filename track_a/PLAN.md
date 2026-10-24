# Pasqal Track A plan: classical vs quantum capacity-constrained clustering

## Scope

Focus exclusively on the **capacity-constrained clustering** problem from
Track A of [the challenge guide](../Pasqal-challenge-final.pdf), version 3.0,
October 24, 2026: assign customers to vehicles, ignoring visit order.
Both the classical baseline and quantum solution stop at customer groups.
No TSP solver, route construction, or full-CVRP reference computation is in scope.
Compare both approaches on the same instances, clustering objective, assignment
constraints, and capacity constraints. No Track B modeling or implementation is
included.

**Submission boundary:** the guide also requires a CVRP gap for Track A, which
needs classical routing and a full-CVRP reference. Defer that evaluation and the
guide's full-CVRP derivation to a later phase. This phase addresses clustering,
not every requirement of a complete Track A submission.

## Planned notebook structure

The existing `track_a/` folder contains this plan (`PLAN.md`). After plan
approval, add the following notebooks alongside it:

- `classical_capacity_constrained_clustering.ipynb`: shared instance generation
  and validation, constrained clustering formulation, classical solutions, and
  clustering reference values.
- `quantum_capacity_constrained_clustering.ipynb`: load the same saved instances,
  derive and validate QUBOs, embed on `AnalogDevice`, design pulses, emulate, and
  save raw-shot clustering results.
- `classical_vs_quantum_comparison.ipynb`: load both sets of saved results and
  compare feasibility, clustering-model gaps, reference-hit probabilities, atom
  counts, and runtimes without rerunning the solvers.

Keep all three notebooks limited to Track A clustering; no routing or TSP cells.

## 1. Shared problem and instances

- Formally derive capacity-constrained clustering: each customer belongs to
  exactly one vehicle, demands are not split, each vehicle's total load is at
  most its capacity, and vehicles may be unused. Use symmetric Euclidean
  customer distances. Store the fixed depot `(50, 50)` as instance metadata
  for later use; it does not enter this phase's objective.
- Generate customers inside `[0, 100]^2` using the prescribed tiers:

  | Tier | Customers N | Vehicles K | Capacity Q | Integer demands |
  |---|---:|---:|---:|---|
  | S | 4 | 2 | 7 | 1-4 |
  | M | 10 | 3 | 20 | 1-8 |
  | L | 15 | 4 | 30 | 1-10 |

- Start with three seeds per tier. Construct a feasible assignment first, then
  generate demands within its capacities and independently generate locations.
  Save the seed and assignment certificate; validate every instance. Report
  `rho = sum(d_i)/(K*Q)` and `max(d_i)/Q`; include capacity-tight examples rather
  than only easy low-load cases.

## 2. Classical Track A formulation and baseline

Use binary `x_ik = 1` when customer `i` belongs to vehicle `k`, and load
`L_k = sum_i d_i*x_ik`. Choose one simple clustering objective:

$$
F(x)=\sum_k\sum_{i<j} c_{ij}x_{ik}x_{jk},
\qquad c_{ij}=\text{Euclidean distance between customers }i,j.
$$

Minimize `F`, subject to **equalities** `sum_k x_ik = 1` and **inequalities**
`L_k <= Q`. This encourages compact groups; it is not total route length.
Use this same constrained objective to evaluate both approaches.

1. **Tier S:** enumerate feasible partitions, obtain the proven optimum and
   number of distinct optima, and cross-check with CPLEX as required.
2. **Tier M:** enumerate all 9,842 distinct partitions into at most three
   nonempty groups, filter by capacity, and obtain the proven clustering
   optimum and number of distinct optima.
3. **Tier L:** assess whether exhaustive enumeration is practical first.
   Otherwise solve the constrained model directly with CPLEX under a recorded
   time limit; optionally also run multi-restart simulated annealing and retain
   the best feasible reference found. Label references according to the guide:
   brute-force optimum versus CPLEX/annealing best known.

Record the CPLEX time limit. For annealing, record initial/final temperatures,
cooling rate, iteration limit, time limit, number of restarts, and seeds.
Treat vehicle relabelings as the same partition. Reference computation is
offline benchmarking, not part of the quantum optimization loop. All references
in this phase are for clustering only.

## 3. QUBO: textbook baseline and proposed atom saving

First derive and verify the standard slack formulation:

$$
E_{\rm slack}=F+A\sum_i(\sum_kx_{ik}-1)^2
+B\sum_k(L_k+s_k-Q)^2,
$$

where each nonnegative integer slack `s_k` has `ceil(log2(Q+1))` bits.
Derive sufficient penalty bounds for this exact encoding.

As a candidate quantum model, retain assignment bits but remove slack using
[unbalanced penalization](https://arxiv.org/abs/2211.13914):

$$
h_k=Q-L_k,\qquad
E_{\rm reduced}=F+A\sum_i(\sum_kx_{ik}-1)^2
+\sum_k[-\alpha h_k+\beta h_k^2],
\quad \alpha,\beta>0.
$$

This remains quadratic and uses no auxiliary atoms. **It is a heuristic
surrogate, not an exact capacity constraint:** it also changes preferences
among feasible loads, including empty vehicles. Validate against the original
constraints and constrained-model optimum, not just QUBO energy.

**Important simplification:** whenever every customer is assigned exactly once,
total demand `D = sum_i d_i = sum_k L_k` is fixed, even if a vehicle is overloaded.
Therefore,

$$
\sum_k[-\alpha(Q-L_k)+\beta(Q-L_k)^2]
=\beta\sum_k L_k^2-\alpha(KQ-D)+\beta(KQ^2-2QD).
$$

Only the load-balancing term `beta*sum_k L_k^2` changes between such assignments.
The asymmetric linear term cancels: `alpha` cannot change their ranking or
independently penalize overloads. It can still affect assignment-invalid
bitstrings. Do not assume this literature method transfers unchanged to our
fixed-fleet problem.

| Encoding | Variable count | S | M | L | Within 80 atoms? |
|---|---|---:|---:|---:|---|
| Assignment only | `N*K` | 8 | 30 | 60 | All |
| Assignment + binary slack | `N*K + K*ceil(log2(Q+1))` | 14 | 45 | 80 | All, on count alone |
| Proposed slack-free | `N*K` | 8 | 30 | 60 | All, on count alone |

Normalize distance costs, then select `A, alpha, beta` using a small,
fixed calibration grid on separate instances. Exhaustively inspect S to
separate penalty-model errors from sampling errors. Check capacity feasibility
and the original clustering objective of the candidate QUBO's lowest-energy
states, and compare with the exact constrained reference. Tune `beta` for the
load-balancing trade-off, not `alpha` as an overload control among assignment-valid
candidates. Retain this encoding only if the checks support it; otherwise revisit
the capacity encoding, using the exact slack QUBO as the control. Freeze settings
before benchmarking. Record the symmetric-matrix convention: pair coefficients
in `x^T Q x` equal `2*Q_ij`.

## 4. Quantum Track A clustering with Pasqal

1. **Get S working first.** Use the installed `qubo-solver` pipeline as a
   reference, then keep embedding and pulses explicit with **Pulser
   `AnalogDevice`**. Follow the installed API rather than copying the guide's
   older configuration syntax.
2. **Derive the native cost explicitly (guide Derive 3.1).** At zero final
   drive amplitude, the Hamiltonian assigns bitstring `z` the following energy
   in angular-frequency units:

   $$
   E_{\rm native}(z)=
   -\delta_{\rm final}\sum_a z_a+
   \sum_{a<b}\frac{C_6/\hbar}{\|r_a-r_b\|^6}z_az_b.
   $$

   Map these linear and pair coefficients to the chosen QUBO, stating the
   positive energy scale and irrelevant constant offset. Global detuning
   controls the shared linear coefficient; atom positions control positive,
   distance-dependent pair couplings. The interaction coefficient is fixed
   (`C6/hbar` approximately `8.657e5 rad/us*um^6` in the guide); use Pulser's
   device value, which already includes this unit conversion.
3. **Check physical compatibility before scaling.** Expanding the reduced QUBO
   gives nonnegative pair couplings, but generally unequal linear coefficients.
   Global detuning cannot reproduce those exactly; 2D `C6/r^6` interactions
   cannot reproduce an arbitrary coupling matrix either. Fit a common linear
   coefficient and atom positions, and report both approximation errors.
   Do not describe an approximate embedding as an exact QUBO implementation.
4. Enforce the guide's envelope: at most 80 atoms, separation at least
   `5 um`, radius at most `38 um`, amplitude at most `12.57 rad/us`,
   detuning within `+/-125.66 rad/us`, and duration at most `6000 ns`.
   Validate against the installed device; do not relax constraints to make a
   run succeed. Include unwanted interactions in the realized model.
5. Start with a smooth three-point amplitude ramp and linear detuning sweep:
   negative initial detuning, final model detuning, zero final amplitude.
   Try only a small fixed set of legal durations and amplitudes. Select using
   best feasible gap, reference-hit probability, and feasibility, not mean cost.
6. Use exact state-vector emulation for S and `emu-mps` for M/L. Record bond
   dimension and other emulator settings, and check convergence on representative
   runs. Sixty atoms fitting the device does not guarantee tractable emulation.
7. Primary results use **global detuning only**. If unequal linear terms dominate
   errors, add a separately labeled **emulation-only DMM** comparison respecting
   its one-sided detuning restrictions. No QPU execution during the hackathon.

**Checkpoint:** do not scale beyond S until decoding, device legality, QUBO
coefficients, and feasible sampling have been checked. If the reduced model
fails, report it and revise its penalties/encoding before scaling.

## 5. Fair comparison and clustering-phase outputs

- Primary quantum results use **raw shots**, with no classical repair or local
  search. Disable solver defaults that add postprocessing, or label their
  outputs separately. Quantum sampling must perform the optimization.
- Compare classical constrained optimization, classical optimization of the
  same reduced QUBO, quantum sampling, and a random-bitstring sampler using
  the same decoding/filtering and sample budget. Separate raw-shot results
  from any later repaired results.
- For each `(instance, method, seed)`, report atom count, shot budget, feasibility
  rate, best feasible **clustering-model gap**, reference status, runtimes,
  and probability of reaching reference. Choose and record a gap tolerance,
  initially `epsilon = 0.1%`; count only feasible hits over all shots.
- Report embedding error as
  `||J_realized - s*J_target||_F / ||s*J_target||_F`, over all off-diagonal
  pairs, stating the fitted positive scale `s`; also report linear-term error.
  Report undefined gaps when no feasible shot exists. Never clip negative gaps
  against best-known references; handle zero reference costs explicitly.
- Deliver customer-to-vehicle assignments, generator, certificates, validator,
  clustering references, model derivation and variable table, `results.csv`,
  a one-command reproduction in the README, a
  pinned `requirements.txt` exported from the existing lock, and a report of
  at most four pages including **what did not work** and AI-use disclosure.

**Execution order after approval:** shared model/validator and S references ->
textbook and slack-free QUBO checks -> legal S quantum pipeline -> M/L ->
comparison tables and report. Stop at validated customer groups; no classical
or quantum routing in this phase.
