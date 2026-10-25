"""Shared alternative-QUBO execution and original-constraint comparisons, without repair."""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from .analysis import clustering_metrics, fingerprint, save
from .classical import (
    ANNEALING_STEPS, FINAL_TEMPERATURE_FACTOR, INITIAL_TEMPERATURE_FACTOR, anneal,
)
from .experiments import _run_record, _select_pulse_trial, _validate_run
from .instances import Instance
from .models import (
    VARIABLE_REDUCTION_ENCODING, Qubo, assignment_view, build_unbalanced_qubo, decode, objective,
    unbalanced_penalty_weights,
)
from .quantum import (
    EMBEDDING_SWAPS, EMULATOR_DT_NS, INTERACTION_CUTOFF, MPS_MAX_BOND_DIM, MPS_PRECISION,
    PulseSettings, backend_for, embedding, emulate, sequence_for, validate_emulation_settings,
)


def _configuration(target: Qubo, lambda1: float, lambda2: float) -> dict:
    return {
        "lambda1": lambda1, "lambda2": lambda2, "assignment_penalty": target.penalty,
        "normalization": target.normalization, "warm_start": False, "repair": False,
        "ground_state_feasibility_guaranteed": False,
    }


def _reference(instance: Instance, output: Path) -> dict:
    reference = json.loads((output / "references" / f"{instance.id}.json").read_text())
    if reference["instance_hash"] != fingerprint(asdict(instance)):
        raise ValueError(f"Classical reference does not match {instance.id}")
    return reference


def _sample_evidence(instance: Instance, target: Qubo, run: dict, output: Path) -> None:
    rows = []
    if run["status"] == "completed":
        counts = run["counts"]
        if not counts or sum(counts.values()) != run["shot_budget"]:
            raise ValueError("Completed alternative-QUBO run has missing shots")
        if any(type(count) is not int or count <= 0 for count in counts.values()):
            raise ValueError("Raw counts must be positive integers")
        for bits, count in counts.items():
            labels = decode(instance, bits, encoding=target.encoding)
            loads = [sum(instance.demands[i] * int(bits[i * instance.vehicles + k])
                         for i in range(instance.n)) for k in range(instance.vehicles)]
            rows.append({
                "bitstring": bits, "count": count, "probability": count / run["shot_budget"],
                "target_energy": target.energy(list(map(int, bits))),
                "feasible": labels is not None, **assignment_view(instance, labels),
                "model_cost": objective(instance, labels) if labels is not None else None,
                "assignment_residuals": [
                    bits[i * instance.vehicles:(i + 1) * instance.vehicles].count("1") - 1
                    for i in range(instance.n)],
                "capacity_headroom": [instance.capacity - load for load in loads],
            })
            if target.encoding == VARIABLE_REDUCTION_ENCODING:
                auxiliary = bits[instance.n * instance.vehicles:]
                logical_slack = [1 - int(bit) for bit in auxiliary]
                rows[-1].update({
                    "physical_auxiliary_bits": auxiliary, "logical_binary_slack": logical_slack,
                    "normalized_capacity_residuals": [
                        load / instance.capacity - 1 - bit
                        for load, bit in zip(loads, logical_slack, strict=True)],
                })
        rows.sort(key=lambda row: (row["target_energy"], -row["count"], row["bitstring"]))
    save(output / "samples" / f"{instance.id}-{run['method']}-{run['seed']}.json", {
        "instance_id": instance.id, "instance_hash": run["instance_hash"],
        "source_run_hash": fingerprint(run), "encoding": target.encoding,
        "variable_order": target.variable_order, "status": run["status"],
        "error": run.get("error"), "configuration": run["configuration"],
        "sampled_states": rows, "exact_ground_states_verified": False,
    })


def _classical_formulation_run(instance: Instance, target: Qubo, configuration: dict,
                               output: Path, shots: int, seed: int, method: str,
                               annealing_steps: int) -> None:
    configuration = {
        **configuration, "annealing_steps": annealing_steps,
        "initial_temperature": INITIAL_TEMPERATURE_FACTOR * target.penalty,
        "final_temperature": FINAL_TEMPERATURE_FACTOR * target.penalty,
        "cooling": "geometric", "restarts": shots, "shots": shots, "seed": seed,
    }
    print(f"{method} {instance.id}: {len(target.variable_order)} bits", flush=True)
    start = perf_counter()
    counts = dict(anneal(target, shots, seed, steps=annealing_steps))
    run = {
        **_run_record(instance, target, method, shots, seed, configuration),
        "status": "completed", "counts": counts, "clustering_seconds": perf_counter() - start,
    }
    save(output / "runs" / f"{instance.id}-{method}-{seed}.json", run)
    _sample_evidence(instance, target, run, output)


def classical_unbalanced_runs(instances: list[Instance], output: Path, shots: int, seed: int, *,
                              lambda1: float | None = None, lambda2: float | None = None,
                              assignment_penalty: float | None = None,
                              annealing_steps: int = ANNEALING_STEPS) -> None:
    _validate_run(instances, output, shots)
    if type(annealing_steps) is not int or annealing_steps <= 0:
        raise ValueError("Annealing steps must be a positive integer")
    for instance in instances:
        _reference(instance, output)
        first, second = unbalanced_penalty_weights(instance, lambda1, lambda2)
        target = build_unbalanced_qubo(instance, first, second, assignment_penalty)
        _classical_formulation_run(instance, target, _configuration(target, first, second),
                                   output, shots, seed, "classical-unbalanced-sa", annealing_steps)


def _quantum_formulation_runs(instances: list[Instance], output: Path, shots: int, seed: int,
                            timeout: float, memory_mb: int, *,
                            targets: list[Qubo], model_configurations: list[dict], formulation: str,
                            embedding_swaps: int = EMBEDDING_SWAPS,
                            dt_ns: int = EMULATOR_DT_NS, interaction_cutoff: float = INTERACTION_CUTOFF,
                            mps_max_bond_dim: int = MPS_MAX_BOND_DIM,
                            mps_precision: float = MPS_PRECISION) -> None:
    _validate_run(instances, output, shots)
    backend = backend_for(instances[0])
    if any(backend_for(instance) != backend for instance in instances):
        raise ValueError("One experiment must use a single tier and emulator backend")
    validate_emulation_settings(shots, timeout, memory_mb, dt_ns, interaction_cutoff,
                                backend=backend, mps_max_bond_dim=mps_max_bond_dim,
                                mps_precision=mps_precision)
    if type(embedding_swaps) is not int or embedding_swaps < 0:
        raise ValueError("Embedding swaps must be a nonnegative integer")
    if len(targets) != len(instances) or len(model_configurations) != len(instances):
        raise ValueError("Every instance requires a target QUBO and model configuration")
    references = [_reference(i, output) for i in instances]
    first_layout = embedding(targets[0], seed, swaps=embedding_swaps)
    trials = []
    for pulse in (PulseSettings(1000, 8.0), PulseSettings(2000, 8.0)):
        print(f"Calibrating {formulation} {instances[0].tier.upper()} on emu-{backend}: {pulse}", flush=True)
        result = emulate(first_layout, pulse, shots, seed, timeout_seconds=timeout, memory_mb=memory_mb,
                         dt_ns=dt_ns, interaction_cutoff=interaction_cutoff, backend=backend,
                         mps_max_bond_dim=mps_max_bond_dim, mps_precision=mps_precision)
        measured = clustering_metrics(instances[0], result.get("counts", {}),
                                      references[0]["clustering"]["cost"], encoding=targets[0].encoding)
        trials.append({"pulse": asdict(pulse), "result": result, "metrics": measured})
    selected, calibration_status = _select_pulse_trial(trials)
    pulse = PulseSettings(**selected["pulse"])
    save(output / formulation / "calibration.json", {
        "instance_id": instances[0].id, "selected": asdict(pulse), "trials": trials,
        "configuration": model_configurations[0], **calibration_status,
    })
    for index, (instance, target, model_configuration) in enumerate(
        zip(instances, targets, model_configurations, strict=True)
    ):
        print(f"Quantum {formulation} {instance.id}: {len(target.variable_order)} atoms", flush=True)
        configuration = {
            **model_configuration, "pulse": asdict(pulse),
            "embedding_seed": seed, "sampling_seed": seed, "shots": shots,
            "backend": f"emu-{backend}", "dt_ns": dt_ns, "interaction_cutoff": interaction_cutoff,
            "memory_mb": memory_mb, "timeout_seconds": timeout,
            "embedding_swap_attempts": embedding_swaps, "postprocessing": False,
        }
        if backend == "mps":
            configuration.update({"max_bond_dim": mps_max_bond_dim, "precision": mps_precision,
                                  "optimize_qubit_ordering": False})
        layout = first_layout if index == 0 else embedding(target, seed, swaps=embedding_swaps)
        save(output / formulation / "embeddings" / f"{instance.id}-{seed}.json", layout)
        sequence = sequence_for(layout, pulse)
        save(output / formulation / "sequences" / f"{instance.id}-{seed}.json",
             json.loads(sequence.to_abstract_repr()))
        result = selected["result"] if index == 0 else emulate(
            layout, pulse, shots, seed, timeout_seconds=timeout, memory_mb=memory_mb,
            dt_ns=dt_ns, interaction_cutoff=interaction_cutoff, backend=backend,
            mps_max_bond_dim=mps_max_bond_dim, mps_precision=mps_precision)
        run = {
            **_run_record(instance, target, f"quantum-{formulation}", shots, seed, configuration),
            **result, "embedding_seconds": layout["seconds"], "emulation_seconds": result["seconds"],
            "pair_relative_error": layout["pair_relative_error"],
            "linear_relative_error": layout["linear_relative_error"], "scale": layout["scale"],
        }
        save(output / "runs" / f"{instance.id}-{run['method']}-{seed}.json", run)
        _sample_evidence(instance, target, run, output)


def quantum_unbalanced_runs(instances: list[Instance], output: Path, shots: int, seed: int,
                            timeout: float, memory_mb: int, *,
                            lambda1: float | None = None, lambda2: float | None = None,
                            assignment_penalty: float | None = None,
                            embedding_swaps: int = EMBEDDING_SWAPS,
                            dt_ns: int = EMULATOR_DT_NS, interaction_cutoff: float = INTERACTION_CUTOFF,
                            mps_max_bond_dim: int = MPS_MAX_BOND_DIM,
                            mps_precision: float = MPS_PRECISION) -> None:
    _validate_run(instances, output, shots)
    targets, configurations = [], []
    for instance in instances:
        first, second = unbalanced_penalty_weights(instance, lambda1, lambda2)
        target = build_unbalanced_qubo(instance, first, second, assignment_penalty)
        targets.append(target)
        configurations.append(_configuration(target, first, second))
    _quantum_formulation_runs(
        instances, output, shots, seed, timeout, memory_mb, targets=targets,
        model_configurations=configurations, formulation="unbalanced", embedding_swaps=embedding_swaps,
        dt_ns=dt_ns, interaction_cutoff=interaction_cutoff,
        mps_max_bond_dim=mps_max_bond_dim, mps_precision=mps_precision,
    )


def comparison_rows(instances: list[Instance], output: Path, seed: int) -> list[dict]:
    """Compare feasible clustering costs, never energies of different penalties."""
    rows = []
    for instance in instances:
        reference = _reference(instance, output)
        rows.append({
            "instance": instance.id, "method": "classical-exact", "encoding": "constrained",
            **assignment_view(instance, reference["clustering"]["labels"]),
            "model_cost": reference["clustering"]["cost"], "status": reference["clustering"]["status"],
            "outcome": "Exact constrained optimum",
        })
        for method in ("classical-qubo-sa", "classical-unbalanced-sa",
                       "classical-variable_reduction-sa", "quantum-global",
                       "quantum-unbalanced", "quantum-variable_reduction"):
            path = output / "runs" / f"{instance.id}-{method}-{seed}.json"
            if not path.exists():
                continue
            run = json.loads(path.read_text())
            if run["instance_hash"] != fingerprint(asdict(instance)):
                raise ValueError(f"Saved run does not match {instance.id}")
            measured = {}
            if run["status"] == "completed":
                if not run["counts"] or sum(run["counts"].values()) != run["shot_budget"]:
                    raise ValueError(f"Shot budget mismatch in {path}")
                measured = clustering_metrics(instance, run["counts"], reference["clustering"]["cost"],
                                              encoding=run["encoding"])
            labels = measured.get("best_model_labels")
            rows.append({
                "instance": instance.id, "method": method, "encoding": run["encoding"],
                "bits": run["atom_count"], "status": run["status"],
                "outcome": ("Feasible sample found" if labels is not None else
                            "No feasible sample" if run["status"] == "completed" else run["status"]),
                **assignment_view(instance, labels),
                "model_cost": objective(instance, labels) if labels is not None else None,
                "actual_shots": measured.get("actual_shots"),
                "feasible_shots": measured.get("feasible_shots"),
                "feasibility_rate": measured.get("feasibility_rate"),
                "best_model_gap": measured.get("best_model_gap"), "error": run.get("error"),
                "lambda1": run["configuration"].get("lambda1"),
                "lambda2": run["configuration"].get("lambda2"),
                "alpha": run["configuration"].get("alpha"),
                "beta": run["configuration"].get("beta"),
                "assignment_penalty": run["configuration"].get(
                    "assignment_penalty", run["configuration"].get("penalty")),
            })
    return rows
