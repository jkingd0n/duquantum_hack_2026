"""Shot-weighted, original-constraint metrics and saved-evidence reporting."""

import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .classical import Routing
from .instances import Instance, validate
from .models import ENCODING, decode, objective

GAP_TOLERANCE_PERCENT = 0.1


def fingerprint(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gap(cost: float, reference: float) -> float | None:
    if reference == 0:
        return 0.0 if cost == 0 else None
    return 100 * (cost - reference) / reference


def metrics(instance: Instance, counts: dict[str, int], reference: dict,
            routing: Routing) -> tuple[dict, list[dict]]:
    if any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError("Raw counts must be positive integers")
    total = sum(counts.values())
    feasible_count = 0
    evaluations: dict[tuple[int, ...], dict] = {}
    hits = {f"{name}_hit_probability_{threshold:g}pct": 0
            for name in ("model", "cvrp") for threshold in (0.1, 1.0, 5.0)}
    routing_seconds = 0.0
    for bitstring, count in counts.items():
        labels = decode(instance, bitstring)
        if labels is None:
            continue
        feasible_count += count
        if labels not in evaluations:
            start = perf_counter()
            routed = routing.evaluate(instance, labels)
            routing_seconds += perf_counter() - start
            value = objective(instance, labels)
            evaluations[labels] = {
                "labels": list(labels), "model_cost": value, "route_cost": routed["cost"],
                "model_gap": gap(value, reference["clustering"]["cost"]),
                "cvrp_gap": gap(routed["cost"], reference["cvrp"]["cost"]),
                "tours": routed["tours"], "shots": 0,
            }
        item = evaluations[labels]
        item["shots"] += count
        for name in ("model", "cvrp"):
            for threshold in (0.1, 1.0, 5.0):
                if item[f"{name}_gap"] is not None and item[f"{name}_gap"] <= threshold:
                    hits[f"{name}_hit_probability_{threshold:g}pct"] += count
    best_model = min(evaluations.values(), key=lambda e: e["model_cost"], default=None)
    best_route = min(evaluations.values(), key=lambda e: e["route_cost"], default=None)
    return {
        "actual_shots": total, "feasible_shots": feasible_count,
        "feasibility_rate": feasible_count / total if total else None,
        "best_model_gap": best_model["model_gap"] if best_model else None,
        "best_cvrp_gap": best_route["cvrp_gap"] if best_route else None,
        "best_model_labels": best_model["labels"] if best_model else None,
        "best_cvrp_labels": best_route["labels"] if best_route else None,
        "routing_seconds": routing_seconds,
        **{key: value / total if total else None for key, value in hits.items()},
    }, list(evaluations.values())


def summarize(output: Path) -> list[dict]:
    """Read saved evidence only: never invoke an optimizer or emulator."""
    rows = []
    for path in sorted((output / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        instance = Instance(
            **{**run["instance"], "coordinates": tuple(map(tuple, run["instance"]["coordinates"])),
               "demands": tuple(run["instance"]["demands"]),
               "certificate": tuple(run["instance"]["certificate"]),
               "depot": tuple(run["instance"]["depot"])}
        )
        validate(instance)
        ref = json.loads((output / "references" / f"{instance.id}.json").read_text())
        row = {key: run.get(key) for key in (
            "instance_id", "method", "seed", "encoding", "configuration_id", "instance_hash",
            "status", "shot_budget", "atom_count", "clustering_seconds",
            "embedding_seconds", "emulation_seconds", "pair_relative_error",
            "linear_relative_error", "scale", "error",
        )}
        row.update(instance.ratios)
        row.update({"certificate_valid": True, "model_reference": ref["clustering"]["cost"],
                    "cvrp_reference": ref["cvrp"]["cost"],
                    "reference_status": ref["status"], "reference_seconds": ref["seconds"],
                    "model_reference_status": ref["clustering"]["status"],
                    "cvrp_reference_status": ref["cvrp"]["status"],
                    "gap_tolerance_percent": GAP_TOLERANCE_PERCENT})
        if run["encoding"] != ENCODING or run["instance_hash"] != fingerprint(run["instance"]):
            raise ValueError(f"Incompatible encoding or modified instance in {path}")
        if ref["instance_hash"] != run["instance_hash"]:
            raise ValueError(f"Reference-instance mismatch in {path}")
        cache_path = output / "evaluations" / path.name
        if run["status"] == "completed":
            if sum(run["counts"].values()) != run["shot_budget"]:
                raise ValueError(f"Shot budget mismatch in {path}")
            cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
            if cached is not None and cached["run_hash"] == fingerprint(run):
                measured = cached["metrics"]
            else:
                measured, evaluations = metrics(instance, run["counts"], ref, _routing_for(instance))
                save(cache_path, {"run_hash": fingerprint(run), "metrics": measured,
                                  "partitions": evaluations})
            row.update(measured)
        rows.append(row)
    if not rows:
        raise ValueError(f"No saved runs found in {output / 'runs'}")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save(output / "summary.json", rows)
    return rows


_ROUTING: dict[tuple, Routing] = {}


def _routing_for(instance: Instance) -> Routing:
    key = instance.coordinates, instance.depot
    if key not in _ROUTING:
        _ROUTING[key] = Routing(instance)
    return _ROUTING[key]


def plot_evidence(output: Path, instance_id: str = "s-0", seed: int = 0):
    import matplotlib.pyplot as plt

    layout = json.loads((output / "embeddings" / f"{instance_id}-{seed}.json").read_text())
    run = json.loads((output / "runs" / f"{instance_id}-quantum-global-{seed}.json").read_text())
    pulse = run["pulse"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    coordinates = layout["coordinates_um"]
    axes[0].scatter(*zip(*coordinates))
    for name, (x, y) in zip(layout["variable_order"], coordinates, strict=True):
        axes[0].annotate(name, (x, y), fontsize=7)
    axes[0].set(xlabel="x (um)", ylabel="y (um)", title="Legal atom layout", aspect="equal")
    from pulser.waveforms import InterpolatedWaveform, RampWaveform
    amplitude = InterpolatedWaveform(pulse["duration_ns"], [0, pulse["amplitude"], 0])
    detuning = RampWaveform(pulse["duration_ns"], pulse["initial_detuning"], layout["final_detuning"])
    axes[1].plot(amplitude.samples.as_array(detach=True))
    axes[1].set(xlabel="time (ns)", ylabel="rad/us", title="Rabi amplitude")
    axes[2].plot(detuning.samples.as_array(detach=True))
    axes[2].set(xlabel="time (ns)", ylabel="rad/us", title="Global detuning")
    fig.tight_layout()
    return fig
