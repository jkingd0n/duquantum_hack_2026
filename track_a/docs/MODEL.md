# Track A model and derivations

Current execution is **S only** (`N=4`, `K=2`, `Q=7`). Every supplied demand vector
is processed (currently ten instances, `s-0` through `s-9`).
Outputs are isolated under `outputs/all_demands/`. Both notebooks use `s-0`,
`[2, 3, 4, 2]`, for a detailed worked example without limiting the benchmark to it.
M/L counts below are
prospective comparisons, not implemented experiments. The input
[inputs/seeds_s.json](../inputs/seeds_s.json) contains the fleet parameters, allowed
demand range, explicit customer coordinates, fixed depot, and demand vectors.
It accepts any nonempty list of feasible
S vectors, not only permutations of a common multiset. Row index identifies a
vector (`s-0`, `s-1`, ...); it is not a permutation RNG seed.

The input contains one shared customer coordinate array in `[0,100]^2`;
`coordinates[i]` corresponds to entry `i` in every demand vector. Classical
CVRP evaluation uses these coordinates and the explicit depot `[50,50]`
directly, without regenerating geometry. The initial coordinates preserve the
previous benchmark geometry. For each vector the loader
exhaustively finds one feasible assignment, called `certificate` internally,
and the validator checks it independently. This is a computed feasibility
witness, not a field users must supply and not a solver warm start.

## Capacity-constrained clustering (A1)

For customers \(i=1,\ldots,N\), vehicles \(k=1,\ldots,K\), demands \(d_i\)
and capacity \(Q\), let \(x_{ik}\in\{0,1\}\). Minimize

\[
 F(x)=\sum_k\sum_{i<j}c_{ij}x_{ik}x_{jk},\qquad
 c_{ij}=\|r_i-r_j\|_2.
\]

Assignment is the **equality** \(\sum_k x_{ik}=1\) for every customer.
Capacity is the **inequality** \(\sum_i d_ix_{ik}\le Q\) for every vehicle.
Vehicles may be unused. Demands are unsplit, and each vehicle carries one
load on one trip, without reloading. Pairwise compactness is not route length.
The fixed depot does not enter this objective.

## Original CVRP (Derive 2.1)

Let \(V=\{0,1,\ldots,N\}\), with depot 0, and
\(A=\{(i,j)\in V^2:i\ne j\}\). Travel costs \(t_{ij}\) are Euclidean distances,
including depot legs. Let \(a_{ijk}\) be a binary route-arc variable,
\(x_{ik}\) binary customer assignment, and \(u_k\) binary vehicle use. Minimize

\[
 \sum_k\sum_{(i,j)\in A}t_{ij}a_{ijk}.
\]

The required **equalities** are

\[
 \sum_kx_{ik}=1,\quad
 \sum_{j\in V\setminus\{i\}}a_{ijk}=x_{ik}
 =\sum_{j\in V\setminus\{i\}}a_{jik}\quad (i\ne0),\quad
 \sum_{j\ne0}a_{0jk}=\sum_{i\ne0}a_{i0k}=u_k.
\]

The required **inequalities** include

\[
 x_{ik}\le u_k,\quad u_k\le\sum_ix_{ik},\quad
 \sum_i d_ix_{ik}\le Q u_k.
\]

For every nonempty customer subset \(S\), vehicle \(k\), and \(h\in S\),
add connectivity/subtour inequalities

\[
 \sum_{i\in S,\ j\in V\setminus S}a_{ijk}\ge x_{hk}.
\]

Thus any served subset connects to the depot; isolated cycles are forbidden.
Unused vehicles have no arcs. Degree, connectivity, and capacity together give
one depot-to-depot trip per used vehicle with no split service or reloading.

The results stage uses an equivalent exact **set-partition** reference for CVRP:
compute the exact depot tour cost of every capacity-feasible customer subset,
then choose disjoint subsets covering every customer using at most \(K\)
groups. This optimizes both grouping and visit order. Anchoring each recursive
group on the least remaining customer removes vehicle-label symmetry.
Held-Karp dynamic programming exhaustively considers all visit orders.
For readability, the classical notebook enumerates all labeled
assignments, keeps one representative per vehicle relabeling, and tries every
feasible grouping for clustering cost only. Its independent CPLEX cross-check
solves the target QUBO and verifies decoded feasibility and the exact cost.
The results notebook tries every tour order for the separate CVRP reference
and fixed-group evaluation. On the supported S instances these searches give
the same exact objectives and counts as the CLI; they are not intended for M/L scaling.
For S, independent CPLEX set-partition MILPs verify both reference objectives;
route-optimum counts additionally identify a route with its reversal.

For a clustering result, by contrast, the evaluator fixes every group and
only optimizes the group's tour. Empty groups cost zero. There is no repair,
customer reassignment, or routing feedback.

## Binary-slack QUBO

Normalize \(c_{ij}\) by \(D=\max(1,\max c_{ij})\).
With \(b=\lceil\log_2(Q+1)\rceil\), introduce
\(s_k=\sum_{r=0}^{b-1}2^ry_{kr}\), \(y_{kr}\in\{0,1\}\), and use

\[
 E=F/D+
 A\sum_i(\sum_kx_{ik}-1)^2+
 B\sum_k(\sum_id_ix_{ik}+s_k-Q)^2.
\]

A feasible certificate has normalized objective \(U\) and zero residuals using
\(s_k=Q-L_k\), including unused vehicles. Choose \(A=B=U+1\).
All terms in \(F/D\) are nonnegative. Every nonzero integer residual has square
at least one, so any violated penalized constraint costs more than \(U\).
Consequently every target-QUBO ground state is feasible with zero slack
residual and minimizes the original constrained objective. This proof does
not extend to approximate embeddings or finite-time samples.

Expansion uses \(z_a^2=z_a\):
\((w^Tz-q)^2=z^Tww^Tz-2q\sum_aw_az_a+q^2\).
We save a symmetric matrix and constant:
\(E(z)=z^T\mathcal Qz+C\). Linear coefficients are \(\mathcal Q_{aa}\);
pair coefficients are \(2\mathcal Q_{ab}\), not \(\mathcal Q_{ab}\).
There are no higher-order terms, negative pair couplings, or extra auxiliaries.
The linear coefficients are generally unequal.

| Variables | S | M | L |
|---|---:|---:|---:|
| Assignment | 8 | 30 | 60 |
| Textbook binary slack | 6 | 15 | 20 |
| Implemented total (identical to textbook) | 14 | 45 | 80 |
| Extra auxiliaries | 0 | 0 | 0 |
| Future slack-free estimate, **not implemented** | 8 | 30 | 60 |

Atom counts alone do not establish geometric embeddability or emulator
tractability. Raw-shot feasibility uses original assignment and capacity
constraints, independently of whether the sampled slack happens to be correct.
Exhaustive ground-state verification checks **both** residual families.

## Native cost and physical approximation (Derive 3.1, A3-A4)

At zero drive amplitude, bitstring \(z\) has angular-frequency energy

\[
 E_{\rm native}(z)=-\delta_f\sum_a z_a+
 \sum_{a<b}\frac{C_6/\hbar}{\|r_a-r_b\|^6}z_az_b.
\]

Pulser's `AnalogDevice.interaction_coeff` is already \(C_6/\hbar\);
do not divide by Planck's constant again. Positions are in micrometers,
times in nanoseconds, and controls/interactions in radians per microsecond.
An exact scaled realization would require \(h_a=-\delta_f=s\mathcal Q_{aa}\)
and \(J_{ab}=2s\mathcal Q_{ab}\) for positive \(s\), with irrelevant
offset \(sC\): \(E_{\rm native}=sE-sC\).

The implementation explicitly fits an **approximation**. Legal triangular
sites and seeded greedy variable-to-site swaps maximize interaction alignment.
Fit \(s=\langle J_{\rm real},J_{\rm target}\rangle/
\|J_{\rm target}\|_F^2>0\), then set
\(\delta_f=-s\,\mathrm{mean}(\mathcal Q_{aa})\). Save all pairs, coordinates,
variable order, scale, offset and separate errors

\[
 e_J=\frac{\|J_{\rm real}-sJ_{\rm target}\|_F}
 {\|sJ_{\rm target}\|_F},\qquad
 e_h=\frac{\|h_{\rm real}-s h_{\rm target}\|_2}
 {\|s h_{\rm target}\|_2}.
\]

Diagonals of the pair matrices are zero; the Frobenius ratio includes every
off-diagonal pair, including unwanted couplings at target zeros. We do not
convert these errors into an unjustified "quantum fidelity".

The primary baseline is global-only. Smooth three-point Rabi amplitude
`[0, 8, 0]` and linear detuning `[-8, delta_final]` are tried at 1000 and
2000 ns on S-0, then frozen. Selection prioritizes best feasible model gap,
then reference-hit probability, then feasibility. Reusing the calibration
instance is disclosed. No QPU or cloud jobs are submitted.

Installed `AnalogDevice` has no DMM channels. No DMM results are claimed.
An emulation-only DMM extension would require a separately labeled virtual
device and \(\epsilon_a\in[0,1]\), \(\delta_{\rm DMM}(t)\le0\);
it must not be represented as global-only hardware operation. This baseline
does not silently substitute such a device.

## Metrics and reproducibility

For each feasible shot, compute model and exact fixed-group route cost.
Report `100 * (cost - reference) / reference`, minimizing the two gaps
**separately**. Missing feasible shots produce undefined gaps. At a zero
reference, an equal zero cost has zero gap; a nonzero cost has undefined
percentage gap. Negative gaps are retained. Hit probabilities at 0.1%, 1% and
5% divide feasible hits by **all** shots. Infeasible or failed runs never
become success-shaped fallback results.

Saved instance hashes, configuration IDs, raw counts, encoding IDs and
variable order identify the evidence. Results assembly caches exact tours
and per-run partition evaluations and does not rerun optimization.
Before assembly, the results stage computes any missing full-CVRP references,
checks their clustering cost against the saved clustering-only reference, and
saves them for reuse. Neither clustering nor pulse calibration runs routing.
The deterministic constrained baseline repeats its one result to the common
budget; this is explicitly not an independent-shot success estimate.
Classical QUBO SA uses independent random starts and no certificate warm start.
No method repairs quantum outputs. The certificate is used only for validation
and the proven penalty bound.

## Submission scope

These notebooks run all supplied explicit S demand vectors.
They demonstrate A1, the embedding and error accounting of A3, and A4's
multiple-instance pulse experiments, but do not complete A2's seed-based generation.
Calibration uses `s-0`, then controls freeze before applying them to every vector.
The saved feasibility certificates and validator remain available.
The implemented variable count equals textbook binary slack;
there is no demonstrated variable reduction.
