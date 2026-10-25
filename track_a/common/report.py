"""Produce a four-page report from saved benchmark evidence only."""

from collections import Counter
import json
from pathlib import Path
import textwrap


def write_report(output: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def page(title: str, paragraphs: list[str]):
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.08, 0.94, title, fontsize=16, weight="bold")
        y = 0.89
        for paragraph in paragraphs:
            wrapped = "\n".join(textwrap.wrap(paragraph, 94))
            figure.text(0.08, y, wrapped, va="top", fontsize=10, linespacing=1.5)
            y -= (wrapped.count("\n") + 1) * 0.021 + 0.025
        return figure

    references = [json.loads(path.read_text()) for path in sorted((output / "references").glob("*.json"))]
    validation = [json.loads(path.read_text()) for path in sorted((output / "validation").glob("*.json"))]
    instances = json.loads((output / "instances.json").read_text())
    tiers = "/".join(sorted({instance["tier"].upper() for instance in instances}))
    parameters = sorted({(len(instance["demands"]), instance["vehicles"], instance["capacity"])
                         for instance in instances})
    methods = Counter(check.get("validation_method", "legacy validation") for check in validation)
    quantum_methods = {"quantum-global", "quantum-unbalanced", "quantum-variable_reduction"}
    backends = Counter(row.get("backend") for row in rows if row["method"] in quantum_methods)
    mps_limits = sorted({(row["max_bond_dim"], row["precision"]) for row in rows
                         if row.get("backend") == "emu-mps"})
    status = Counter(f"{row['method']}: {row['status']}" for row in rows
                     if row["method"] in quantum_methods)
    calibration_path = output / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    quantum = [row for row in rows if row["method"] in quantum_methods and row["status"] == "completed"]
    unbalanced_runs = []
    for path in sorted((output / "runs").glob("*-unbalanced*-*.json")):
        unbalanced_runs.append(json.loads(path.read_text()))
    unbalanced_weights = sorted({(run["configuration"]["lambda1"], run["configuration"]["lambda2"])
                                 for run in unbalanced_runs})
    assignment_weights = [run["configuration"]["assignment_penalty"] for run in unbalanced_runs]
    crosschecks = [ref["cplex_crosscheck"] for ref in references if "cplex_crosscheck" in ref]
    reported_checks = [[check[name] for name in ("clustering", "cvrp") if name in check]
                       for check in crosschecks]
    confirmed = sum(bool(checks) and all(check.get("agrees", False) for check in checks)
                    for checks in reported_checks)
    with PdfPages(output / "report.pdf") as pdf:
        fig = page("Track A: model and verification", [
            f"Scope: all {len(references)} supplied {tiers} demand vectors, capacity-constrained clustering only. "
            "Seed-generated inputs (A2) remain a submission gap. "
            "Quantum optimization returns groups; "
            "all routing is exact classical evaluation with those groups fixed. No quantum speedup claim.",
            "Binary x_ik assigns each customer exactly once. Minimize the sum of Euclidean "
            "customer-pair distances within each vehicle, subject to load <= capacity. Unused "
            "vehicles are allowed. The depot is fixed at (50, 50) and is absent from compactness.",
            "The binary-slack target QUBO divides distances by the largest customer distance, adds assignment "
            "squares and binary-slack capacity squares. With U the normalized certificate cost, "
            "A = B = U + 1 proves feasibility of every target ground state. The certificate is "
            "not a solver warm start. Pair coefficients are twice off-diagonal matrix entries.",
            "Variable counts (binary slack / unbalanced / variable reduction): S 14 / 8 / 10; M 45 / 30 / 33. "
            "Unbalanced H adds sum_k(-lambda1*h_k + lambda2*h_k^2), "
            "h_k = Q - sum_j d_j*x_jk, to normalized compactness and assignment squares. "
            "Unbalanced has no slack bits. Variable reduction retains one complemented binary "
            "slack per vehicle and the photographed load/Q - 1 - b residual. Neither "
            "alternative inherits the original slack feasibility proof.",
            f"Evidence: {len(references)} exact clustering/full-CVRP reference pairs; "
            f"{len(validation)} QUBO validations ({sum(v['passed'] for v in validation)} passed); "
            f"methods: {dict(methods)}. S checks all 14-bit strings; M uses the penalty "
            "proof and every feasible partition, not all 2^45 strings. Reported independent "
            f"CPLEX checks agree on {confirmed}/{len(crosschecks)} instances.",
            "The separate docs/MODEL.md gives the original CVRP arc/assignment formulation, "
            "connectivity constraints, penalty expansion and native Hamiltonian derivation.",
        ])
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("Instances, embedding and pulse", [
            f"{len(references)} explicit {tiers} demand vectors with (N,K,Q)={parameters}. "
            "The selected input file contains N, K, Q, demand_range, demands, "
            "customer coordinates and depot. Fixed coordinates are read directly from the input. "
            "Feasibility witnesses are computed internally, not supplied as input.",
            "References use exhaustive exact search: the Python API uses subset dynamic programming "
            "and Held-Karp tours; the S notebook also enumerates assignments and tour orders. "
            "M uses subset DP to avoid factorial tour enumeration. "
            "Capacity-filtered partitions select groups "
            "and tours jointly for full CVRP, independently of compactness. Vehicle labels "
            "are quotiented. S additionally counts tour optima modulo reversal.",
            "Embedding uses a legal triangular grid, seeded greedy atom-label swaps and a "
            "least-squares positive interaction scale. Every pair is retained, including "
            "unwanted interactions where the target is zero. Device validation enforces "
            "5 um spacing, 38 um radius, 80 atoms and all installed AnalogDevice controls.",
            "Native energy at zero drive is -delta*sum(z) + sum(C6/r^6*z_a*z_b). "
            "C6 = 865723.02 rad/us*um^6 from Pulser. The fitted scale and scaled constant "
            "are saved. Global detuning fits the mean diagonal only: this is not an exact QUBO.",
            "Pair error is ||J_real - s J_target||_F / ||s J_target||_F over all "
            "off-diagonal pairs. Linear error is ||h_real - s h||_2 / ||s h||_2. "
            "Embedding fidelity is characterized by these errors, not by a claimed exact mapping.",
            f"Calibration: {calibration['selected'] if calibration else 'not run'}. "
            f"Status: {calibration.get('status', 'not recorded') if calibration else 'not run'}; "
            f"fallback: {calibration.get('selection_fallback') if calibration else None}. "
            "Candidate sweep durations, amplitudes and initial detunings are recorded "
            "in calibration trials; the final amplitude is zero. Settings freeze after the first instance. "
            "Calibration uses a benchmark instance, so it is not held-out validation.",
            f"Recorded quantum backends: {dict(backends)}. S uses emu-sv; M uses emu-mps on CPU. "
            f"MPS limits (maximum bond dimension, precision): {mps_limits}. "
            "Time steps, cutoffs, seeds and resource limits are recorded per run. "
            "MPS is approximate; convergence is not established. L, DMM and QPU operation are deferred.",
        ])
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("Observed results and comparison", [
            f"Quantum run statuses: {dict(status)}. All statuses, failure messages, "
            "actual shots, reference values and gaps are preserved in results.csv.",
            f"Additional unbalanced runs: {len(unbalanced_runs)}; "
            f"(lambda1, lambda2): {unbalanced_weights}; assignment penalty range: "
            f"{(min(assignment_weights), max(assignment_weights)) if assignment_weights else 'not run'}. "
            "Compare original feasible costs, not energies across formulations. "
            "Its separate pulse calibration is saved in unbalanced/calibration.json.",
            "Feasibility is measured against the original assignment and capacity "
            "constraints, not against slack-bit residuals. Reference-hit probabilities "
            "use all raw shots as denominator. Gap thresholds: 0.1%, 1%, 5%. "
            "Undefined gaps remain missing; negative best-known gaps are never clipped.",
            "The best compactness gap and best CVRP gap may come from different sampled "
            "partitions. Each receives exact fixed-group depot tours; routing never changes "
            "groups or feeds back into the quantum optimizer.",
            "Classical constrained results are deterministic exact solutions repeated to "
            "the common shot budget, not independent stochastic samples. Classical QUBO "
            "comparisons use independent Metropolis chains with recorded schedules; random baselines use "
            "uniform raw bits. Both use identical decoding and feasibility filtering.",
            "Reference, clustering, embedding, emulation and routing runtimes are separated. "
            "Reference DP is offline. The deterministic clustering baseline independently "
            "enumerates feasible partitions; this comparison measures quality rather than speedup.",
        ])
        fig.canvas.draw()
        text_bottom = min(text.get_window_extent().y0 / fig.bbox.height for text in fig.texts)
        ax = fig.add_axes((0.13, 0.10, 0.76, min(0.28, text_bottom - 0.15)))
        methods = sorted({row["method"] for row in rows})
        for index, method in enumerate(methods):
            values = [row["feasibility_rate"] for row in rows if row["method"] == method
                      and row.get("feasibility_rate") is not None]
            ax.scatter([index] * len(values), values, alpha=0.5)
        ax.set(xticks=range(len(methods)), xticklabels=methods, ylabel="Feasible raw-shot fraction",
               ylim=(-0.03, 1.03), title="Completed runs only")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("What did not work, limits and AI use", [
            "The dense slack QUBO cannot generally be reproduced by 2D repulsive "
            "interactions with one shared detuning. Positive coupling signs alone do not "
            "establish embeddability. Large saved errors are physical-model approximation "
            "failures, not failures of the proven target penalty or exact references.",
            f"Completed quantum runs: {len(quantum)}; runs with no feasible raw shots: "
            f"{sum(row.get('feasible_shots') == 0 for row in quantum)}. "
            "L is intentionally out of scope. Resource-limited or failed runs have "
            "no reported success metrics.",
            "Resource limits bound each isolated emulator by wall time and "
            "address space. MPS bond/truncation convergence, time-step convergence and noise robustness are not established "
            "by this baseline. No scaling or hardware performance claims are made.",
            "Only one geometry is covered. Explicit demand vectors include different load ratios, "
            "but do not replace a seed-based instance generator. Low feasible "
            "rates on uniform bitstrings are expected for one-hot assignment. A small "
            "pulse calibration and local swap layout are baselines, not an exhaustive "
            "embedding/control optimization. There is no classical repair of quantum shots.",
            "Reproduction: run the classical, quantum, then results notebooks with the "
            ".venv kernel, or call the shared Python experiment stages. Saved evidence is "
            "reassembled by the results notebook without rerunning clustering or quantum jobs. "
            "The three notebooks show executable core algorithms with beginner-friendly "
            "explanations and saved-result views.",
            "AI-use disclosure: an AI assistant using Copilot SDK in VS Code implemented "
            "the shared modules, notebooks, derivations and automated report. "
            "The submitted numbers come from actual local solvers/emulators and saved "
            "raw outputs, not generated claims. Human review remains necessary.",
        ])
        pdf.savefig(fig)
        plt.close(fig)
