"""Callable experiment stages for notebook-first local reproduction."""

from collections.abc import Callable
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .analysis import clustering_metrics, fingerprint, save
from .classical import (
    ANNEALING_STEPS, FINAL_TEMPERATURE_FACTOR, INITIAL_TEMPERATURE_FACTOR,
    anneal, clustering_reference, exact_references, partitions,
)
from .evidence import save_sample_evidence, save_target_evidence
from .instances import INPUT_PATH, Instance, ensure_instances, feasible, instance_record
from .models import Qubo, build_qubo, decode, encode, objective, validate_qubo
from .quantum import (
    EMBEDDING_SWAPS, EMULATOR_DT_NS, INTERACTION_CUTOFF, MPS_MAX_BOND_DIM, MPS_PRECISION,
    PulseSettings, backend_for, device_metadata, embedding, emulate, sequence_for, validate_emulation_settings,
)


def prepare_experiment(output: Path, input_path: Path = INPUT_PATH) -> list[Instance]:
    instances = ensure_instances(input_path)
    records = [instance_record(instance) for instance in instances]
    instance_path = output / "instances.json"
    if instance_path.exists() and fingerprint(json.loads(instance_path.read_text())) != fingerprint(records):
        raise ValueError("Output directory belongs to different inputs/geometry; use a new output path")
    save(instance_path, records)
    save(output / "generation.json", {
        "input_path": str(input_path.resolve()), "numpy": np.__version__,
        "coordinates": "explicit input coordinates, shared across demand vectors",
        "selected_demand_indices": [instance.instance_seed for instance in instances],
        "input": json.loads(input_path.read_text()),
    })
    return instances


def _validate_run(instances: list[Instance], output: Path, shots: int) -> None:
    if not instances:
        raise ValueError("Experiments require at least one instance")
    if type(shots) is not int or not 1 <= shots <= 2000:
        raise ValueError("Require 1-2000 shots")
    path = output / "instances.json"
    if not path.exists() or fingerprint(json.loads(path.read_text())) != fingerprint(
        [instance_record(instance) for instance in instances]
    ):
        raise ValueError("Prepare the output directory with these inputs first")


def _run_record(instance: Instance, qubo: Qubo, method: str, shots: int, seed: int,
                configuration: dict) -> dict:
    record = asdict(instance)
    return {
        "instance": record, "instance_id": instance.id, "instance_hash": fingerprint(record),
        "method": method, "seed": seed, "encoding": qubo.encoding, "shot_budget": shots,
        "atom_count": len(qubo.variable_order), "variable_order": qubo.variable_order,
        "configuration": configuration, "configuration_id": fingerprint(configuration),
    }


def _select_pulse_trial(trials: list[dict]) -> tuple[dict, dict]:
    valid = [trial for trial in trials if trial["result"]["status"] == "completed"]
    selected = max(valid, key=lambda t: (
        -(t["metrics"]["best_model_gap"] if t["metrics"]["best_model_gap"] is not None else float("inf")),
        t["metrics"]["model_hit_probability_0.1pct"] or 0,
        t["metrics"]["feasibility_rate"] or 0,
    )) if valid else trials[0]
    return selected, {
        "status": "completed" if valid else "failed",
        "selection_fallback": None if valid else "first candidate; all calibration trials failed",
        "frozen_before_benchmark": True, "uses_benchmark_instance": True,
        "selection": "best feasible model gap, reference hit probability, feasibility",
    }


def classical_runs(instances: list[Instance], output: Path, shots: int, seed: int, *,
                   annealing_steps: int = ANNEALING_STEPS) -> None:
    _validate_run(instances, output, shots)
    if type(annealing_steps) is not int or annealing_steps <= 0:
        raise ValueError("Annealing steps must be a positive integer")
    for instance in instances:
        print(f"Classical {instance.id}", flush=True)
        record = asdict(instance)
        reference = clustering_reference(instance)
        reference["instance_hash"] = fingerprint(record)
        save(output / "references" / f"{instance.id}.json", reference)
        qubo = build_qubo(instance)
        validation = validate_qubo(instance, reference["clustering"]["cost"])
        crosscheck_key = "installed_qubo_solver" if instance.tier == "s" else "independent_clustering_solver"
        validation[crosscheck_key] = reference["cplex_crosscheck"]["clustering"]
        save(output / "validation" / f"{instance.id}.json", validation)
        for method in ("classical-constrained", "classical-qubo-sa", "random-bits"):
            config = {"method": method, "shots": shots, "seed": seed,
                      "warm_start": False, "repair": False, "time_limit_seconds": None}
            start = perf_counter()
            if method == "classical-constrained":
                labels = tuple(reference["clustering"]["labels"]) if instance.tier == "l" else min(
                    (labels for labels in partitions(instance.n, instance.vehicles)
                     if feasible(instance, labels)),
                    key=lambda labels: objective(instance, labels),
                )
                counts = {encode(instance, labels): shots}
                seconds = perf_counter() - start
            elif method == "classical-qubo-sa":
                config.update({"annealing_steps": annealing_steps,
                               "initial_temperature": INITIAL_TEMPERATURE_FACTOR * qubo.penalty,
                               "final_temperature": FINAL_TEMPERATURE_FACTOR * qubo.penalty,
                               "cooling": "geometric", "restarts": shots, "penalty": qubo.penalty})
                counts = dict(anneal(qubo, shots, seed, steps=annealing_steps))
                seconds = perf_counter() - start
            else:
                rng = np.random.default_rng(seed)
                bits = rng.integers(0, 2, (shots, len(qubo.variable_order)))
                counts = dict(Counter("".join(map(str, row)) for row in bits))
                seconds = perf_counter() - start
            save(output / "runs" / f"{instance.id}-{method}-{seed}.json", {
                **_run_record(instance, qubo, method, shots, seed, config),
                "status": "completed", "counts": counts, "clustering_seconds": seconds,
                "deterministic_solver_result_repeated": method == "classical-constrained",
            })


def quantum_runs(instances: list[Instance], output: Path, shots: int, seed: int,
                 timeout: float, memory_mb: int, *, embedding_swaps: int = EMBEDDING_SWAPS,
                 dt_ns: int = EMULATOR_DT_NS, interaction_cutoff: float = INTERACTION_CUTOFF,
                 mps_max_bond_dim: int = MPS_MAX_BOND_DIM, mps_precision: float = MPS_PRECISION) -> None:
    _validate_run(instances, output, shots)
    backend = backend_for(instances[0])
    if any(backend_for(instance) != backend for instance in instances):
        raise ValueError("One experiment must use a single tier and emulator backend")
    validate_emulation_settings(shots, timeout, memory_mb, dt_ns, interaction_cutoff,
                                backend=backend, mps_max_bond_dim=mps_max_bond_dim,
                                mps_precision=mps_precision)
    if type(embedding_swaps) is not int or embedding_swaps < 0:
        raise ValueError("Embedding swaps must be a nonnegative integer")
    save(output / "device.json", device_metadata())
    validations = {}
    for instance in instances:
        reference = json.loads((output / "references" / f"{instance.id}.json").read_text())
        if reference["instance_hash"] != fingerprint(asdict(instance)):
            raise ValueError(f"Classical reference does not match {instance.id}")
        target = build_qubo(instance)
        validations[instance.id] = validate_qubo(instance, reference["clustering"]["cost"])
        save_target_evidence(instance, target, validations[instance.id], output)
    first = instances[0]
    ref = json.loads((output / "references" / f"{first.id}.json").read_text())
    qubo = build_qubo(first)
    first_layout = embedding(qubo, seed, swaps=embedding_swaps)
    candidates = [PulseSettings(1000, 8.0), PulseSettings(2000, 8.0)]
    trials = []
    for pulse in candidates:
        print(f"Calibrating {first.tier.upper()} on emu-{backend}: {pulse}", flush=True)
        result = emulate(first_layout, pulse, shots, seed, timeout_seconds=timeout, memory_mb=memory_mb,
                         dt_ns=dt_ns, interaction_cutoff=interaction_cutoff, backend=backend,
                         mps_max_bond_dim=mps_max_bond_dim, mps_precision=mps_precision)
        measured = clustering_metrics(first, result.get("counts", {}), ref["clustering"]["cost"])
        trials.append({"pulse": asdict(pulse), "result": result, "metrics": measured})
    selected, calibration_status = _select_pulse_trial(trials)
    pulse = PulseSettings(**selected["pulse"])
    save(output / "calibration.json", {
        "instance_id": first.id, "selected": asdict(pulse), "trials": trials,
        **calibration_status,
    })
    small_ok = all(check["passed"] for check in validations.values())
    for instance in instances:
        print(f"Quantum {instance.id}", flush=True)
        qubo = build_qubo(instance)
        configuration = {"pulse": asdict(pulse), "embedding_seed": seed, "sampling_seed": seed,
                         "backend": f"emu-{backend}", "dt_ns": dt_ns, "interaction_cutoff": interaction_cutoff,
                         "memory_mb": memory_mb, "timeout_seconds": timeout, "shots": shots,
                         "embedding_swap_attempts": embedding_swaps, "penalty": qubo.penalty,
                         "normalization": qubo.normalization, "postprocessing": False}
        if backend == "mps":
            configuration.update({"max_bond_dim": mps_max_bond_dim, "precision": mps_precision,
                                  "optimize_qubit_ordering": False})
        base = _run_record(instance, qubo, "quantum-global", shots, seed, configuration)
        layout = first_layout if instance.id == first.id else embedding(qubo, seed, swaps=embedding_swaps)
        save(output / "embeddings" / f"{instance.id}-{seed}.json", layout)
        sequence = sequence_for(layout, pulse)
        save(output / "sequences" / f"{instance.id}-{seed}.json",
             json.loads(sequence.to_abstract_repr()))
        if instance.id == first.id:
            result = selected["result"]
        else:
            result = emulate(layout, pulse, shots, seed,
                             timeout_seconds=timeout, memory_mb=memory_mb,
                             dt_ns=dt_ns, interaction_cutoff=interaction_cutoff, backend=backend,
                             mps_max_bond_dim=mps_max_bond_dim, mps_precision=mps_precision)
        has_feasible = any(decode(instance, bits) is not None for bits in result.get("counts", {}))
        small_ok = small_ok and result["status"] == "completed" and has_feasible
        run = {
            **base, **result, "variable_order": qubo.variable_order,
            "embedding_seconds": layout["seconds"], "emulation_seconds": result["seconds"],
            "pair_relative_error": layout["pair_relative_error"],
            "linear_relative_error": layout["linear_relative_error"], "scale": layout["scale"],
            "pulse": asdict(pulse),
        }
        save(output / "runs" / f"{instance.id}-quantum-global-{seed}.json", run)
        save_sample_evidence(instance, qubo, validations[instance.id], run, output)
    save(output / "checkpoint.json", {
        "tier": first.tier, "passed": small_ok, f"{first.tier}_passed": small_ok,
        "instance_ids": list(validations),
        "requirements": f"validated QUBO, legal device, feasible raw {first.tier.upper()} shots",
    })


def evaluation_references(instances: list[Instance], output: Path, *,
                          reference_for: Callable[[Instance], dict] = exact_references) -> None:
    """Add missing full-CVRP references only when preparing results."""
    for instance in instances:
        path = output / "references" / f"{instance.id}.json"
        reference = json.loads(path.read_text())
        if reference["instance_hash"] != fingerprint(asdict(instance)):
            raise ValueError(f"Classical reference does not match {instance.id}")
        if "cvrp" in reference:
            continue
        complete = reference_for(instance)
        if not np.isclose(complete["clustering"]["cost"], reference["clustering"]["cost"]):
            raise AssertionError("Evaluation and clustering references disagree")
        save(path, {**complete, "instance_hash": reference["instance_hash"]})
