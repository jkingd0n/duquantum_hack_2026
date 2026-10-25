# Track A model and derivations

The notebooks support **S** (`N=4`, `K=2`, `Q=7`, demands 1-4) and **M**
(`N=10`, `K=3`, `Q=20`, demands 1-8). Set the same `INPUT_FILE` and optional
`OUTPUT_DIRECTORY` in each notebook. Every supplied demand vector is processed.
Default S outputs remain under `outputs/all_demands/`; M defaults to
`outputs/seeds_m/`. Fixed worked numbers use `s-0`, `[2, 3, 4, 2]`;
live code recalculates for the selected tier. L counts are accounting only.
The selected input, [S](../inputs/seeds_s.json) or [M](../inputs/seeds_m.json),
contains the fleet parameters, allowed
demand range, explicit customer coordinates, fixed depot, and demand vectors.
It accepts any nonempty list of feasible
S/M vectors, not only permutations of a common multiset. Row index identifies a
vector (`s-0`, `s-1`, ... or `m-0`, `m-1`, ...); it is not a permutation RNG seed.

The input contains one shared customer coordinate array in `[0,100]^2`;
`coordinates[i]` corresponds to entry `i` in every demand vector. Classical
CVRP evaluation uses these coordinates and the explicit depot `[50,50]`
directly, without regenerating geometry. The initial coordinates preserve the
previous benchmark geometry. For each vector the loader
finds one feasible assignment, called `certificate` internally,
and the validator checks it independently. This is a computed feasibility
witness, not a field users must supply and not a solver warm start.
S retains its original enumeration order; M uses capacity-pruned backtracking.

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
The explicit Python evaluation stage computes the separate full-CVRP reference
and fixed-group evaluation using subset/Held-Karp dynamic programming.
The results notebook displays only previously saved routing metrics with
matching evidence hashes; it performs no tour search or reference computation.
L is not enabled.
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
For S, every 14-bit string is checked. For M, the validator checks the penalty
bound and every feasible distinct partition's encoded energy/residuals, finds
the exact constrained optimum, then enumerates all optimal vehicle relabelings.
This derives the exact 45-bit target ground strings without enumerating `2^45`
candidates. Validation files explicitly distinguish these methods.

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
2000 ns on the first selected instance, then frozen. Selection prioritizes best feasible model gap,
then reference-hit probability, then feasibility. Reusing the calibration
instance is disclosed. No QPU or cloud jobs are submitted.

S samples with `emu-sv`. M samples with `emu-mps`, with configurable maximum
bond dimension (default 64) and truncation precision (default `1e-5`).
The timestep, timeout and address-space limits are configurable and recorded.
Qubit-order optimization is disabled to preserve assignment/slack bit order.
MPS is an approximation, and no bond/truncation/timestep convergence is claimed.
An oversized state-vector request is rejected before allocation; no automatic
state-vector fallback is used for M. Resource failures retain explicit statuses.

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
variable order identify the evidence. The read-only results notebook decodes
saved raw counts and displays statistics for S and M without solving, emulating,
or routing. Optional cached route gaps are used only if both run and reference
hashes match. Missing or stale evaluation evidence remains unavailable.
The explicit Python evaluation functions can compute missing full-CVRP
references and fixed-group tours, but are never invoked by the viewer.
Neither clustering nor pulse calibration runs routing.
The deterministic constrained baseline repeats its one result to the common
budget; this is explicitly not an independent-shot success estimate.
Classical QUBO SA uses independent random starts and no certificate warm start.
No method repairs quantum outputs. The certificate is used only for validation
and the proven penalty bound.

## Submission scope

These notebooks process all supplied explicit S or M demand vectors.
They implement A1, the embedding and error accounting of A3, and A4's
multiple-instance pulse workflow, but do not complete A2's seed-based generation.
Completed quantum execution, feasible samples and MPS convergence must be assessed
from actual evidence, not inferred from backend support.
Calibration uses the first selected instance, then controls freeze for every vector.
The saved feasibility certificates and validator remain available.
The original baseline's variable count equals textbook binary slack.
The additional formulations below reduce bits without inheriting its proof.
# Additional slack-free unbalanced formulation

The binary-slack formulation below remains the original baseline. Both
optimization notebooks additionally implement the photographed "Optimize H"
formulation, using customer-major assignment bits only:

$$
h_k(x)=Q-\sum_j d_jx_{jk},\qquad
H_u(x)=F(x)/s+\beta\sum_j(\sum_kx_{jk}-1)^2
       +\sum_k[-\lambda_1h_k(x)+\lambda_2h_k(x)^2].
$$

Here $h_k\geq0$ is the capacity constraint (equality is feasible),
$s=\max(1,\max_{ij}c_{ij})$ preserves the existing distance normalization,
and all three penalty weights are positive. The demand sums must include
$x_{jk}$; otherwise they do not depend on the vehicle assignment.
For $L_k=\sum_jd_jx_{jk}$, each capacity term is

$$
(-\lambda_1Q+\lambda_2Q^2)
+(\lambda_1-2\lambda_2Q)L_k+\lambda_2L_k^2.
$$

Using $x_a^2=x_a$, the symmetric matrix convention is
$H_u=x^\mathsf{T}Q_ux+c_u$; off-diagonal entries contribute twice.
There are $NK$ variables, no slack bits: S uses 8 and M uses 30.
`build_unbalanced_qubo` implements this expansion.

With lambda arguments left as `None`, the solver reads the saved presets from
`inputs/unbalanced_penalties.json`: S uses `lambda1=0.2`, `lambda2=0.01`;
M uses `lambda1=0.75`, `lambda2=0.02`. Explicit positive weights override either
default independently. These are benchmark presets, not universal optima.
The S choice was checked on every one of the 256 assignment bitstrings for
each of the current three S instances: all target ground states match
feasible constrained optima, versus only one instance with the original
weights. SA checks on separate random seeds improved feasibility and optimal
hits. Quantum seed-0 checks improved optimal hits (2 to 7 out of 768 shots),
but reduced feasible hits (14 to 10); this is benchmark-specific tuning, not
held-out instance validation or a proof for other data.

M used an 80-pair grid on seeds 0-2 and the current three M vectors. Among the
five training leaders, the selected pair had the lowest mean best feasible
clustering gap on seeds 3-5 (7.07%, versus 15.72% for the original weights).
Confirmation seeds 6-11 were not used to reselect: mean best gap was 10.50%
versus 10.95%, but feasibility fell from 72.40% to 59.05% and worst-case gap
increased. The preference is quality rather than maximum feasibility.
Both baseline and selected 30-atom MPS checks exceeded their 60-second limit
at bond dimension 64; quantum-optimal weights and convergence are not established.
A lower-cost bond-dimension-4 screen completed for both pairs, with zero feasible
shots out of 256 each on the first M instance. It is not a convergence study.
The complete measurements and input fingerprints are saved in
`outputs/penalty_tuning/m_unbalanced_search.json`.

A large positive `lambda1` can reward leaving customers unassigned: increasing
headroom lowers its linear penalty until the assignment penalty intervenes.
A large `lambda2` can overemphasize load balance relative to normalized
compactness. For valid one-hot assignments, total headroom is fixed, so the
linear term is constant across feasible groupings; the quadratic term is not.
Both weights therefore need checking against the normalized objective and the
assignment weight, not just against each other.

The assignment weight defaults to the original certificate-based numeric value,
but its original proof does not carry over. The headroom term is minimized at
$\lambda_1/(2\lambda_2)$, is nonconstant on feasible assignments, and can alter
the original optimum or yield an infeasible ground state. Original constraints
are therefore checked independently for every raw sample. No repair or
certificate warm start is used.

`classical-unbalanced-sa` reuses the raw Metropolis annealer; `quantum-unbalanced`
uses a separate legal embedding and pulse calibration. S uses `emu-sv`; M uses
`emu-mps` even after the bit reduction. Physical embedding remains approximate,
and exact ground states of the new target are not claimed. Compare original
feasible compactness, gaps, and shot feasibility, never target energies between
formulations. Distinct run names, encoding metadata and evidence paths retain
all original solver results.

## Additional photographed variable-reduction formulation

The requested photo is implemented literally, including its minus slack sign:

$$
H_v=F(x)/s+\alpha\sum_k\left(L_k(x)/Q-1-b_k\right)^2
       +\beta\sum_j\left(\sum_kx_{jk}-1\right)^2,\qquad b_k\in\{0,1\}.
$$

Here $L_k=\sum_jd_jx_{jk}$. There are $NK+K$ variables: **S 10, M 33**,
including one binary slack per vehicle. This is not slack-free and does not
encode the capacity inequality exactly: capacity residuals vanish at $Q$ or
$2Q$, so overload can be favored. Original feasibility is checked separately.

Direct expansion produces negative assignment/slack couplings. For repulsive
Rydberg compatibility, use the exact bijection $z_k=1-b_k$:

$$
(L_k/Q-1-b_k)^2=(L_k/Q+z_k-2)^2.
$$

For $a_j=d_j/Q$, the capacity square is
$\alpha[(\sum_ja_jx_{jk}+z_k)^2-4(\sum_ja_jx_{jk}+z_k)+4]$.
Its pair coefficients are nonnegative. The same symmetric-matrix convention
and normalized objective are used by both methods. Variable order is
customer-major assignments followed by $z_0,\ldots,z_{K-1}$; saved sample
evidence also records logical $b_k$, raw physical auxiliaries and residuals.
Complementing an auxiliary bit is reversible encoding, not sample repair.

`build_variable_reduction_qubo` builds this target.
`variable_reduction_alpha` and `variable_reduction_beta` allow independent
positive finite overrides. Each defaults to $1+F(\text{certificate})/s$,
an untuned starting value with no feasibility or optimality proof.
Exhaustive S bitstrings and randomized M bitstrings verify matrix energies
against the literal formula. No exact target-ground-state claim is made.

`classical-variable_reduction-sa` reuses the same raw Metropolis sampler;
`quantum-variable_reduction` has separate calibration, embeddings and sequences
under `variable_reduction/`. S uses `emu-sv`, M uses `emu-mps`.
All original methods remain. The saved-only results viewer identifies the three
formulations as `slack`, `unbalanced_penalty`, `variable_reduction`, comparing
original feasible costs/gaps and raw-shot probabilities, not target energies.
Each cluster map shows the reference plus all six solver assignments.
