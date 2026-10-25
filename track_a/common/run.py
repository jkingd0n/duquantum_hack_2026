"""One-command local reproduction: python -m track_a.common.run."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .analysis import clustering_metrics, fingerprint, save, summarize
from .classical import anneal, clustering_reference, exact_references, partitions
from .instances import INPUT_PATH, OUTPUT_PATH, ensure_instances, feasible, instance_record
from .models import ENCODING, build_qubo, decode, encode, objective, validate_small_qubo
from .quantum import PulseSettings, device_metadata, embedding, emulate, sequence_for


def classical_runs(instances, output: Path, shots: int, seed: int) -> None:
    for instance in instances:
        print(f"Classical {instance.id}", flush=True)
        record = asdict(instance)
        reference = clustering_reference(instance)
        reference["instance_hash"] = fingerprint(record)
        save(output / "references" / f"{instance.id}.json", reference)
        qubo = build_qubo(instance)
        if instance.tier == "s":
            validation = validate_small_qubo(instance, reference["clustering"]["cost"])
            validation["installed_qubo_solver"] = reference["cplex_crosscheck"]["clustering"]
            save(output / "validation" / f"{instance.id}.json",
                 validation)
        config = {"shots": shots, "seed": seed, "annealing_steps": 400,
                  "initial_temperature": 10 * qubo.penalty,
                  "final_temperature": 0.01 * qubo.penalty,
                  "cooling": "geometric", "restarts": shots,
                  "warm_start": False, "repair": False, "penalty": qubo.penalty,
                  "time_limit_seconds": None}
        for method in ("classical-constrained", "classical-qubo-sa", "random-bits"):
            start = perf_counter()
            if method == "classical-constrained":
                labels = min(
                    (labels for labels in partitions(instance.n, instance.vehicles)
                     if feasible(instance, labels)),
                    key=lambda labels: objective(instance, labels),
                )
                counts = {encode(instance, labels): shots}
                seconds = perf_counter() - start
            elif method == "classical-qubo-sa":
                counts = dict(anneal(qubo, shots, seed))
                seconds = perf_counter() - start
            else:
                rng = np.random.default_rng(seed)
                bits = rng.integers(0, 2, (shots, len(qubo.variable_order)))
                counts = dict(Counter("".join(map(str, row)) for row in bits))
                seconds = perf_counter() - start
            save(output / "runs" / f"{instance.id}-{method}-{seed}.json", {
                "instance": record, "instance_id": instance.id, "instance_hash": fingerprint(record),
                "method": method, "seed": seed, "encoding": ENCODING,
                "configuration_id": fingerprint({**config, "method": method}),
                "configuration": {**config, "method": method},
                "status": "completed", "counts": counts, "shot_budget": shots,
                "atom_count": len(qubo.variable_order), "clustering_seconds": seconds,
                "deterministic_solver_result_repeated": method == "classical-constrained",
                "variable_order": qubo.variable_order,
            })


def quantum_runs(instances, output: Path, shots: int, seed: int,
                 timeout: float, memory_mb: int) -> None:
    save(output / "device.json", device_metadata())
    validations = {}
    for instance in instances:
        reference = json.loads((output / "references" / f"{instance.id}.json").read_text())
        if reference["instance_hash"] != fingerprint(asdict(instance)):
            raise ValueError(f"Classical reference does not match {instance.id}")
        validations[instance.id] = validate_small_qubo(instance, reference["clustering"]["cost"])
    first = instances[0]
    ref = json.loads((output / "references" / f"{first.id}.json").read_text())
    qubo = build_qubo(first)
    first_layout = embedding(qubo, seed)
    candidates = [PulseSettings(1000, 8.0), PulseSettings(2000, 8.0)]
    trials = []
    for pulse in candidates:
        print(f"Calibrating S: {pulse}", flush=True)
        result = emulate(first_layout, pulse, shots, seed, timeout_seconds=timeout, memory_mb=memory_mb)
        measured = clustering_metrics(first, result.get("counts", {}), ref["clustering"]["cost"])
        trials.append({"pulse": asdict(pulse), "result": result, "metrics": measured})
    valid = [trial for trial in trials if trial["result"]["status"] == "completed"]
    selected = max(valid, key=lambda t: (
        -(t["metrics"]["best_model_gap"] if t["metrics"]["best_model_gap"] is not None else float("inf")),
        t["metrics"]["model_hit_probability_0.1pct"] or 0,
        t["metrics"]["feasibility_rate"] or 0,
    )) if valid else trials[0]
    pulse = PulseSettings(**selected["pulse"])
    save(output / "calibration.json", {
        "instance_id": first.id, "selected": asdict(pulse), "trials": trials,
        "frozen_before_benchmark": True, "uses_benchmark_instance": True,
        "selection": "best feasible model gap, reference hit probability, feasibility",
    })
    small_ok = all(check["passed"] for check in validations.values())
    for instance in instances:
        print(f"Quantum {instance.id}", flush=True)
        record = asdict(instance)
        qubo = build_qubo(instance)
        configuration = {"pulse": asdict(pulse), "embedding_seed": seed, "sampling_seed": seed,
                         "backend": "emu-sv", "dt_ns": 10, "interaction_cutoff": 0.0,
                         "memory_mb": memory_mb, "timeout_seconds": timeout, "shots": shots,
                         "embedding_swap_attempts": 1200, "penalty": qubo.penalty,
                         "normalization": qubo.normalization, "postprocessing": False}
        base = {
            "instance": record, "instance_id": instance.id, "instance_hash": fingerprint(record),
            "method": "quantum-global", "seed": seed, "encoding": ENCODING,
            "shot_budget": shots, "atom_count": len(qubo.variable_order),
            "configuration": configuration, "configuration_id": fingerprint(configuration),
        }
        layout = first_layout if instance.id == first.id else embedding(qubo, seed)
        save(output / "embeddings" / f"{instance.id}-{seed}.json", layout)
        sequence = sequence_for(layout, pulse)
        save(output / "sequences" / f"{instance.id}-{seed}.json",
             json.loads(sequence.to_abstract_repr()))
        if instance.id == first.id:
            result = selected["result"]
        else:
            result = emulate(layout, pulse, shots, seed,
                             timeout_seconds=timeout, memory_mb=memory_mb)
        has_feasible = any(decode(instance, bits) is not None for bits in result.get("counts", {}))
        small_ok = small_ok and result["status"] == "completed" and has_feasible
        save(output / "runs" / f"{instance.id}-quantum-global-{seed}.json", {
            **base, **result, "variable_order": qubo.variable_order,
            "embedding_seconds": layout["seconds"], "emulation_seconds": result["seconds"],
            "pair_relative_error": layout["pair_relative_error"],
            "linear_relative_error": layout["linear_relative_error"], "scale": layout["scale"],
            "pulse": asdict(pulse),
        })
    save(output / "checkpoint.json", {"s_passed": small_ok, "instance_ids": list(validations),
                                     "requirements": "exact QUBO, legal device, feasible raw S shots"})


def evaluation_references(instances, output: Path) -> None:
    """Add missing full-CVRP references only when preparing results."""
    for instance in instances:
        path = output / "references" / f"{instance.id}.json"
        reference = json.loads(path.read_text())
        if reference["instance_hash"] != fingerprint(asdict(instance)):
            raise ValueError(f"Classical reference does not match {instance.id}")
        if "cvrp" in reference:
            continue
        complete = exact_references(instance)
        if not np.isclose(complete["clustering"]["cost"], reference["clustering"]["cost"]):
            raise AssertionError("Evaluation and clustering references disagree")
        save(path, {**complete, "instance_hash": reference["instance_hash"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "classical", "quantum", "results"], default="all")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.shots <= 0 or args.shots > 2000 or args.timeout <= 0 or args.memory_mb < 1024:
        parser.error("Require 1-2000 shots, positive timeout, and at least 1024 MB memory")
    instances = ensure_instances(args.input)
    instance_path = args.output / "instances.json"
    if instance_path.exists():
        previous = json.loads(instance_path.read_text())
        if fingerprint(previous) != fingerprint([instance_record(i) for i in instances]):
            parser.error("Output directory belongs to different inputs/geometry; use a new --output path")
    if args.stage in {"all", "classical"}:
        save(args.output / "instances.json", [instance_record(i) for i in instances])
        save(args.output / "generation.json", {
            "input_path": str(args.input.resolve()), "numpy": np.__version__,
            "coordinates": "explicit input coordinates, shared across demand vectors",
            "selected_demand_indices": [i.instance_seed for i in instances],
            "input": json.loads(args.input.read_text()),
        })
        classical_runs(instances, args.output, args.shots, args.seed)
    if args.stage in {"all", "quantum"}:
        quantum_runs(instances, args.output, args.shots, args.seed, args.timeout, args.memory_mb)
    if args.stage in {"all", "results"}:
        evaluation_references(instances, args.output)
        rows = summarize(args.output)
        from .report import write_report
        write_report(args.output, rows)
        print(f"Wrote {len(rows)} rows to {args.output / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
