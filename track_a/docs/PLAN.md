# Pasqal Track A plan: classical vs quantum capacity-constrained clustering

> **Current implementation scope (updated by user): S only.**
> Only `demands[0]` is currently processed. Other input rows are retained but
> skipped. Single-instance evidence lives under `outputs/first_demand/`,
> separate from the earlier multi-instance results.
> M/L execution and scaling are deferred. The input in `inputs/seeds_s.json`
> contains `N`, `K`, `Q`, `demand_range`, explicit customer `coordinates`,
> the fixed `depot`, and an array of demand vectors. It does not require
> permutation provenance, retry counts, or saved feasibility certificates.
> Coordinates are read directly from the input; feasibility witnesses are computed
> internally for validation and penalty bounds. This supersedes the original
> three-tier generator/JSON requirements below for the current implementation.
> The user also revised the presentation: core algorithms and experiment loops
> now live visibly in the notebooks, with beginner-friendly explanations.
> Shared modules remain for CLI reproduction and infrastructure. Automated
> tests have been removed at the user's request; model and input validation remain.

## Scope

Focus exclusively on the **capacity-constrained clustering** problem from
Track A of [the challenge guide](../../Pasqal-challenge-final.pdf), version 3.0,
October 24, 2026: assign customers to vehicles, ignoring visit order.
Both clustering optimizers output customer groups. Add exact classical routing
and full-CVRP reference computation solely to evaluate the required CVRP gap.
Routing does not change the sampled groups or feed back into optimization.
Compare both approaches on the same instances, clustering objective, assignment
constraints, and capacity constraints. No Track B modeling or implementation is
included.

**Primary goal:** as the guide states, quantum speedup over a well-tuned
classical solver is not being evaluated. Prioritize efficient encoding into
native neutral-atom physics and extraction of feasible solutions. Classical
solutions establish correctness and solution-quality references, not an
expectation that quantum execution will be faster.

**Track A evaluation boundary:** include both the clustering-model gap and the
CVRP gap required by Section 9.2. All routing is classical evaluation; quantum
TSP, alternative quantum routing encodings, and Track B-specific metrics remain
out of scope. Include the original CVRP derivation requested by Derive 2.1.

## Track A task checklist

These are the four tasks in Section 5.1, page 6 of the challenge PDF:

| PDF task | Planned work and required evidence |
|---|---|
| **5.1 / A1: clustering model** | Sections 2-3: binary variables, compactness objective, assignment equality, capacity inequality, unconstrained QUBOs, and explicit penalty-weight selection and validation. |
| **5.2 / A2: generate instances** | Section 1: seeded generator, prescribed tiers and assumptions, saved feasible assignment for every instance, independent validator, and both demand ratios. |
| **5.3 / A3: embed** | Section 4: variable-to-atom mapping, legal coordinates, realized interactions, and quantified target-versus-realized interaction error. |
| **5.4 / A4: shape the pulse** | Section 4: explicit amplitude/detuning waveforms, execution on multiple instances, saved pulse/emulation settings, and results reported through Section 5. |

The checklist defines implementation acceptance criteria, not a claim that the
unimplemented models or experiments have already passed them. Section 2 also
covers the shared CVRP formulation and Track A evaluation requirements.

## Planned notebooks and shared modules

Only the following three notebooks live directly in `track_a/`. Documentation
is grouped in `docs/`, Python helpers and CLI code in `common/`, seed definitions
in `inputs/`, and saved evidence in `outputs/`:

- `classical_capacity_constrained_clustering.ipynb`: shared instance generation
  and validation, constrained clustering formulation, classical solutions, and
  clustering reference values; also document the original CVRP formulation and
  compute the separate full-CVRP references for evaluation.
- `quantum_capacity_constrained_clustering.ipynb`: load the same saved instances,
  derive and validate the textbook binary-slack QUBO, embed on
  `AnalogDevice`, design pulses, emulate, and save raw-shot clustering results.
- `results.ipynb`: assemble the Track A evidence required by Sections 9-10,
  using saved classical and quantum outputs. Include:
  1. A concise model/penalty summary and the required variable-count table,
     explaining the chosen formulation's compromises.
  2. Instance seeds, tier parameters, certificate-validation status, load
     ratios, largest-demand ratios, and clustering/full-CVRP reference values,
     statuses, and solver settings.
  3. Legal atom layouts, target-versus-realized embedding errors with their
     formula, pulse waveforms/parameters, and emulator settings, including
     MPS bond dimensions where used. Separate global-only and DMM results.
  4. Tables for every `(instance, method, seed)` with the required best feasible
     model/CVRP gaps, feasibility rate, reference-hit probabilities and tolerance,
     atom count, embedding fidelity, and reference status. Include classical
     and random-sampler comparisons without making quantum speedup a goal.
  5. Explicit **what did not work**, limitations, and AI-use disclosure.
  Use the shared exact classical routing evaluator for saved feasible
  clusterings, caching its outputs; do not rerun clustering solvers or quantum
  jobs. Produce the required `results.csv` and the evidence for the separate
  report of at most four pages. The notebook does not replace that report,
  the model write-up, or the README's one-command reproduction instructions.

Keep all three notebooks limited to Track A optimization and its required
evaluation. Classical routing-evaluation cells are allowed; quantum TSP and
Track B implementation are not.

The dedicated encoding-efficiency notebook and broad search across alternative
encoding families are deferred. Keep only the formulation and limited
calibration needed to deliver and explain the binary-slack quantum baseline.
Do not remove slack in this phase; alternatives such as unbalanced penalization
are later follow-up work.

Make notebooks understandable without opening implementation files:
include executable core modeling, solving, embedding, pulse, and analysis
code alongside plain-language explanations. Retain small Python modules under
`track_a/common/` for the CLI and genuinely shared infrastructure. The intentional
visible copies of core definitions also support the CLI workflow:

| Planned module | Shared responsibility |
|---|---|
| `instances.py` | Tier parameters, seeded generation, tier-JSON loading/saving, instance/certificate validation, and distance matrices. |
| `models.py` | Clustering objective and original constraints, encoding construction and metadata, penalty configuration, and candidate decoding. |
| `classical.py` | Partition enumeration, classical solver adapters, clustering/full-CVRP references, and cached exact classical routing for fixed groups. |
| `quantum.py` | Embedding, device validation, pulse construction, emulation, and raw-shot extraction for the quantum notebook. |
| `analysis.py` | Consistent feasibility/gap metrics, result persistence/loading, Track A summary tables, and reusable plotting helpers. |

Use the same objective evaluator, decoder, and feasibility validator in
classical solving, quantum execution, and reporting. Save results with encoding
and configuration identifiers so `results.ipynb` can join them to instances and
references unambiguously. Keep runtime model and input checks visible in the
notebooks. Add helpers only when needed; no generic plugin framework or separate
implementations of the same algorithm in notebooks. These are planned modules,
not files to create before implementation approval.

Also plan three shared instance JSON files, loaded by both classical and quantum
runners rather than duplicating seeds or demand generation in notebooks:

| Planned file | Fixed PDF parameters | Saved instances |
|---|---|---|
| `track_a/inputs/seeds_s.json` | `N=4, K=2, Q=7`, integer demands `1-4` | 10 |
| `track_a/inputs/seeds_m.json` (deferred) | `N=10, K=3, Q=20`, integer demands `1-8` | 10 |
| `track_a/inputs/seeds_l.json` (deferred) | `N=15, K=4, Q=30`, integer demands `1-10` | 10 |

Each file stores its tier parameters, fixed depot and customer coordinates,
base demand vector, base feasibility certificate, base seed, generator
settings/version, and an `instances` list. Each entry stores `instance_seed`,
the actual permuted demand vector in customer-ID order, and its feasibility
certificate. Use permutation seeds `0` through `9` within each tier, giving
**30 instances total**: 10 unique permutations of one feasible demand multiset
per tier, not 10 independently generated multisets or exhaustive enumeration.
Include all three JSON files in the submission. Keep solver/sampling seeds
separate from generation seeds and record them in results.

## 1. Shared problem and instances (Task 5.2 / A2)

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

- Initially construct each tier's JSON using a local seeded generator; record
  the base seed, permutation seeds, generator settings, and library versions:
  1. Sample customer coordinates uniformly in `[0, 100]^2`, independently of
     demands. Keep these coordinates and the depot fixed across the tier's
     10 permutations, so only customer demands change.
  2. Construct one feasible base demand vector and certificate. Distribute
     customers round-robin over `K` vehicles. Within each group, sample each
     integer demand from `1` to the smaller of the tier demand ceiling and
     remaining vehicle capacity minus one unit reserved for each customer
     still to receive a demand. This guarantees a feasible base assignment.
  3. Ensure the base multiset has at least 10 distinct orderings:
     `N! / product_v count(v)! >= 10`. Redraw the base demands if needed.
     This matters especially on S, where repeated demands may allow fewer
     than 10 unique permutations.
  4. Use seeds `0` through `9` to sample 10 unique demand permutations, rejecting
     duplicate demand vectors. For a permutation `p`, set `d'_i = d_p(i)` and
     certificate label `a'_i = a_p(i)`: every vehicle retains its original
     load, so feasibility is preserved. Coordinates are not permuted.
  5. Save the explicit vectors and certificates in the tier JSON, not just
     the seeds. Use stated finite retry limits for base-demand and permutation
     sampling, record attempt counts, and fail explicitly if exhausted.
     Include generator code and seeds in the submission.
- Independently validate each saved instance: exact tier parameters and customer
  count, finite coordinates in the square, fixed depot, integer demands in the
  tier range and at most `Q`, and a complete certificate assigning every
  customer exactly once to a legal vehicle with load at most `Q`. Use Euclidean
  distances, ensuring symmetry and the triangle inequality. Interpret each
  group as one unsplit load for one trip, without reloading; no tour is computed.
  Reject invalid instances explicitly before either solver runs.
- Report the actual `rho = sum(d_i)/(K*Q)` and `max(d_i)/Q` for every instance.
  Both ratios are constant across permutations within a tier. This simple
  benchmark tests different assignments of demands to fixed locations; it does
  not cover different demand multisets or load ratios within that tier.
  The certificate is only a feasibility witness: do not fix its groups, their sizes, or their
  number of nonempty vehicles in either optimization model. Unused vehicles
  remain permitted. Both solvers load the identical frozen entries from
  `seeds_s.json`, `seeds_m.json`, and `seeds_l.json`; they do not independently
  regenerate demands or coordinates. Neither receives the certificate as a warm
  start or a repair result. Identify instances by `(tier, instance_seed)` and
  verify each file's tier parameters, 10-entry count, unique seeds, distinct
  demand vectors with the same base multiset, and entry metadata before either
  runner proceeds.

## 2. Classical Track A formulation and baseline (Task 5.1 / A1)

Let customers be `i in {1, ..., N}` and vehicles `k in {1, ..., K}`.
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
offline benchmarking, not part of the quantum optimization loop. Keep clustering
and full-CVRP references distinct: neither is a substitute for the other.

### Original CVRP formulation and classical evaluation

- For guide Derive 2.1, state the customer/depot sets, demands, fleet size,
  capacity, and travel costs. Derive the original total-distance minimization
  model with vehicle-assignment and route-arc variables. Explicitly distinguish
  equalities from inequalities, including exactly-once service, depot
  departure/return for used vehicles, flow conservation, capacity, and
  connectivity/subtour elimination. Preserve unused vehicles, one trip per
  vehicle, and no split deliveries or reloading. This is a classical reference
  formulation, not an additional quantum encoding.
- Compute a separate **full-CVRP reference** that can choose both groups and
  visit orders. Apply Section 9.3's hierarchy to this original constrained
  problem: exhaustive search where practical; otherwise time-limited CPLEX,
  optionally supplemented by multi-restart annealing. On S, obtain the
  exhaustive optimum, count distinct optima modulo vehicle labels and route
  reversal, and verify CPLEX agreement. Record all settings and reference
  statuses; never use the clustering optimum or penalized QUBO energy as the
  full-CVRP reference.
- For every distinct feasible partition returned by a clustering method, solve
  each group's depot-to-depot TSP **exactly and classically**, using exhaustive
  tours for small groups or Held-Karp dynamic programming for larger ones.
  Keep the groups fixed: no reassignment, repair, or routing-based feedback.
  Empty groups cost zero. Cache exact group tours across partitions and
  methods, and validate coverage, depot endpoints, and capacity.
- Define `C_route(x)` as the sum of these exact tour lengths and report
  `CVRP_gap(x) = 100*(C_route(x) - C_CVRP_ref)/C_CVRP_ref`.
  Report the best CVRP gap over feasible sampled partitions separately from
  the best clustering-model gap: the two best partitions may differ. Preserve
  raw shot counts when computing reference-hit probabilities, even when routing
  only unique partitions. No feasible shots means an undefined gap.
- Route classical and random-sampler clusterings with the same evaluator.
  Report routing/reference computation time separately from clustering,
  embedding, and emulation. This measures grouping quality on delivery distance
  without claiming a quantum routing result.

## 3. QUBO: binary-slack baseline first

Use the standard binary-slack formulation as the primary classical-QUBO and
quantum target in this phase:

$$
E_{\rm slack}=F+A\sum_i(\sum_kx_{ik}-1)^2
+B\sum_k(L_k+s_k-Q)^2,
$$

where `s_k = sum_b 2^b*y_kb`, `y_kb in {0,1}`, and each slack uses
`ceil(log2(Q+1))` bits. A feasible load can make the capacity penalty zero with
`s_k = Q-L_k`, including partially loaded or unused vehicles.

Normalize distance costs and derive sufficient bounds for `A` and `B`.
For example, let `U` be the normalized objective of a known feasible
certificate. Since `F >= 0` and any nonzero integer constraint residual has
squared value at least one, `A > U` and `B > U` ensure that every ground state
of the exact target QUBO decodes to a constrained optimum. Start with
`A = B = U+1`; any limited calibration must preserve the proven bounds.
This mathematical guarantee applies to the target QUBO, not to its approximate
physical embedding or finite-duration quantum samples.

| Encoding | Variable count | S | M | L | Within 80 atoms? |
|---|---|---:|---:|---:|---|
| Assignment only | `N*K` | 8 | 30 | 60 | All |
| Binary-slack baseline used now | `N*K + K*ceil(log2(Q+1))` | 14 | 45 | 80 | All, on count alone |
| Potential future slack-free formulation, not run now | `N*K` before any extra auxiliaries | 8 | 30 | 60 | Count estimate only |

Exhaustively inspect the 14-bit S target QUBO, verifying both assignment and
slack residuals and comparing decoded ground states with the constrained
clustering reference. Separate formulation, embedding, and sampling errors.
Freeze calibration settings before benchmarking. Record the symmetric-matrix
convention: pair coefficients in `x^T Q x` equal `2*Q_ij`.

### Required encoding validation

Perform the small-instance derivation, decoding, and penalty checks in the
quantum clustering notebook, reusing shared modules. Count every assignment,
slack, and auxiliary atom, and explicitly identify nonquadratic terms,
unsupported coupling signs, and unequal linear terms before embedding.
Never silently discard unsupported terms or claim atom savings from an encoding
that cannot be implemented on the stated device.

Freeze the binary-slack formulation and limited calibration settings before
the shared 30-instance benchmark. Report the actual model, its limitations,
and any failed checks in `results.ipynb`; no slack-free runs or multi-family
encoding sweep are required in this phase.

### Later follow-up: reducing slack overhead

After establishing this baseline, consider methods such as
[unbalanced penalization](https://arxiv.org/abs/2211.13914) and measure whether
atom savings improve embedding and feasible-sampling performance. This is not
part of the current implementation. Preserve the modeling caveat: with
`h_k = Q-L_k`, the proposed sum of `-alpha*h_k + beta*h_k^2` reduces to
`beta*sum_k L_k^2` plus a constant whenever every customer is assigned once.
The asymmetric linear term cancels across vehicles, so this is a load-balancing
surrogate rather than an exact capacity penalty. Any future method needs its
own derivation and validation against the binary-slack baseline.

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
   distance-dependent pair couplings. The interaction coefficient is fixed;
   use Pulser's device value, which already includes the conversion to
   angular-frequency units.
3. **Check physical compatibility before scaling.** Expanding the binary-slack QUBO
   gives nonnegative pair couplings, but generally unequal linear coefficients.
   Global detuning cannot reproduce those exactly; 2D `C6/r^6` interactions
   cannot reproduce an arbitrary coupling matrix either. Fit a common linear
   coefficient and atom positions, and report both approximation errors.
   Do not describe an approximate embedding as an exact QUBO implementation.
4. **Enforce the AnalogDevice hardware envelope.** The following values are
   quoted from [the challenge guide, Section 3.1, page 4](../../Pasqal-challenge-final.pdf#page=4):

   | Parameter | Value stated in the challenge PDF |
   |---|---|
   | Maximum atoms | `80` |
   | Minimum atom separation | `5 um` |
   | Maximum register radius | `38 um` |
   | Maximum Rabi frequency, `Omega_max` | `12.57 rad/us` (`2*pi * 2 MHz`) |
   | Detuning range | `+/-125.66 rad/us` (`+/-2*pi * 20 MHz`) |
   | Maximum sequence duration | `6000 ns` |
   | Fixed interaction coefficient, `C6/hbar` | `8.657 * 10^5 rad/us * um^6` |

   Positions are in micrometers, pulse durations in nanoseconds, and amplitude,
   detuning, and interaction strengths in radians per microsecond. The PDF's
   decimal frequency values are rounded; validate numerical limits against the
   installed `AnalogDevice` at its full precision rather than using rounded
   constants as overrides. Record its parameters and flag any discrepancy with
   the guide beyond rounding before proceeding. Do not relax constraints to
   make a run succeed. Include unwanted interactions in the realized model.
   For Task 5.3 / A3, save the variable-to-atom order, coordinates, target
   couplings, realized couplings, fitted scale, and embedding error for every
   instance and embedding seed. Compute realized interactions over all atom
   pairs, including pairs whose target coupling is zero.
5. Start with a smooth three-point amplitude ramp and linear detuning sweep:
   negative initial detuning, final model detuning, zero final amplitude.
   Try only a small fixed set of legal durations and amplitudes. Select using
   best feasible gap, reference-hit probability, and feasibility, not mean cost.
   For Task 5.4 / A4, save both waveforms and all pulse parameters, run the
   selected design on the multiple saved instances in each tier, and record
   the shot budget, sampling seed, raw counts, and emulator settings with the
   reported results. Decode shots using the saved variable-to-atom order and
   validate against the original assignment and capacity constraints.
6. Use `emu-sv` for exact state-vector emulation of S's 14 atoms, and assess
   `emu-mps` for M's 45 and L's 80 atoms. Record bond dimension and other
   emulator settings, and check convergence on representative runs. L exactly
   reaches the hardware atom cap and exceeds the guide's typical 30-60 atom
   MPS range: device legality does not guarantee tractable emulation. Set
   explicit resource limits, attempt scaling only after smaller runs succeed,
   and report resource-limited or unconverged runs as such, not as valid results.
   Do not silently remove slack, relax device limits, or report invented gaps
   to make a large-tier run appear successful.
7. Primary results use **global detuning only**: every atom sees the same
   `delta(t)`. If unequal linear terms dominate errors, add a separately labeled
   **emulation-only DMM** comparison. As specified in Section 3.1, page 5, its
   additional detuning is `epsilon_a * delta_DMM(t)`, with fixed
   `epsilon_a in [0, 1]` and `delta_DMM(t) <= 0`; it can only make excitation
   less favorable relative to the global drive. Compare DMM and global-only
   results separately. QPU operation permits global detuning only, and the
   hackathon itself uses emulators only; no QPU execution is planned.

**Checkpoint:** do not scale beyond S until decoding, device legality, QUBO
coefficients, and feasible sampling have been checked. If the slack baseline
fails, diagnose its formulation, embedding, or pulse before scaling; do not
switch to a slack-free method as an undocumented fallback.

## 5. Fair comparison and clustering-phase outputs

- Primary quantum results use **raw shots**, with no classical repair or local
  search. Disable solver defaults that add postprocessing, or label their
  outputs separately. Quantum sampling must perform the optimization.
- Compare classical constrained optimization, classical optimization of the
  same binary-slack QUBO, quantum sampling, and a random-bitstring sampler using
  the same decoding/filtering and sample budget. Separate raw-shot results
  from any later repaired results.
- For each `(instance, method, seed)`, report atom count, shot budget, feasibility
  rate, best feasible **clustering-model gap** and **CVRP gap**, their separate
  reference costs/statuses, runtimes, and reference-hit probabilities at each
  gap level. Choose and record a gap tolerance, initially `epsilon = 0.1%`;
  count only feasible hits over all shots, without renormalizing after filtering.
- Report embedding error as
  `||J_realized - s*J_target||_F / ||s*J_target||_F`, over all off-diagonal
  pairs, stating the fitted positive scale `s`; also report linear-term error.
  Report undefined gaps when no feasible shot exists. Never clip negative gaps
  against best-known references; handle zero reference costs explicitly.
- Deliver customer-to-vehicle assignments, generator, certificates, validator,
  clustering and full-CVRP references, exact evaluation tours, model derivations
  and variable table, `results.csv`,
  the required variable-count comparison against the textbook formulation,
  shared modules used by the notebooks, and a `results.ipynb` narrative covering
  the chosen model, embedding, pulse design, required metrics, classical
  comparisons, and what did not work,
  a one-command reproduction in the README, a
  pinned `requirements.txt` exported from the existing lock, and a report of
  at most four pages including **what did not work** and AI-use disclosure.

**Execution order after approval:** shared instance/model/validation modules
and S clustering/CVRP references ->
binary-slack QUBO checks and a legal 14-atom S quantum baseline ->
validated quantum clustering pipeline on M/L ->
exact classical routing evaluation and remaining full-CVRP references ->
`results.ipynb`, comparison tables, and report. Extract reused logic into the
shared modules as each stage is implemented. Quantum optimization stops at
validated customer groups; routing is classical evaluation only.
