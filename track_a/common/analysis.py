"""Shot-weighted, original-constraint metrics and saved-evidence reporting."""

import csv
from collections.abc import Callable, Iterable
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .classical import Routing
from .instances import Instance, instance_from_record
from .models import ENCODING, UNBALANCED_ENCODING, VARIABLE_REDUCTION_ENCODING, decode, objective

GAP_TOLERANCE_PERCENT = 0.1
HIT_THRESHOLDS_PERCENT = (GAP_TOLERANCE_PERCENT, 1.0, 5.0)
EVALUATION_VERSION = 2


class RoutingEvaluator(Protocol):
    def evaluate(self, instance: Instance, labels: tuple[int, ...]) -> dict: ...


def fingerprint(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gap(cost: float, reference: float) -> float | None:
    if reference == 0:
        return 0.0 if cost == 0 else None
    return 100 * (cost - reference) / reference


def _samples(instance: Instance, counts: dict[str, int], *,
             encoding: str = ENCODING) -> tuple[int, list[dict]]:
    if any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError("Raw counts must be positive integers")
    partitions: dict[tuple[int, ...], dict] = {}
    for bitstring, count in counts.items():
        labels = decode(instance, bitstring, encoding=encoding)
        if labels is not None:
            if labels not in partitions:
                partitions[labels] = {"labels": list(labels),
                                      "model_cost": objective(instance, labels), "shots": 0}
            partitions[labels]["shots"] += count
    return sum(counts.values()), list(partitions.values())


def _hit_probabilities(values: Iterable[tuple[float | None, int]],
                       total: int, name: str) -> dict:
    values = list(values)
    return {
        f"{name}_hit_probability_{threshold:g}pct": (
            sum(count for value, count in values if value is not None and value <= threshold)
            / total if total else None)
        for threshold in HIT_THRESHOLDS_PERCENT
    }


def _model_metrics(samples: list[dict], total: int, reference: float) -> dict:
    best = min(samples, key=lambda item: item["model_cost"], default=None)
    feasible_shots = sum(item["shots"] for item in samples)
    return {
        "actual_shots": total, "feasible_shots": feasible_shots,
        "feasibility_rate": feasible_shots / total if total else None,
        "best_model_gap": gap(best["model_cost"], reference) if best else None,
        "best_model_labels": best["labels"] if best else None,
        **_hit_probabilities(((gap(item["model_cost"], reference), item["shots"])
                             for item in samples), total, "model"),
    }


def clustering_metrics(instance: Instance, counts: dict[str, int], reference: float, *,
                       encoding: str = ENCODING) -> dict:
    """Clustering-only raw-shot statistics for pulse selection, without routing."""
    total, samples = _samples(instance, counts, encoding=encoding)
    return _model_metrics(samples, total, reference)


def metrics(instance: Instance, counts: dict[str, int], reference: dict,
            routing: RoutingEvaluator, *, encoding: str = ENCODING) -> tuple[dict, list[dict]]:
    total, evaluations = _samples(instance, counts, encoding=encoding)
    measured = _model_metrics(evaluations, total, reference["clustering"]["cost"])
    routing_seconds = 0.0
    for item in evaluations:
        start = perf_counter()
        routed = routing.evaluate(instance, tuple(item["labels"]))
        routing_seconds += perf_counter() - start
        item.update({"route_cost": routed["cost"], "tours": routed["tours"],
                     "model_gap": gap(item["model_cost"], reference["clustering"]["cost"]),
                     "cvrp_gap": gap(routed["cost"], reference["cvrp"]["cost"])})
    best_route = min(evaluations, key=lambda item: item["route_cost"], default=None)
    return {
        **measured,
        "best_cvrp_gap": best_route["cvrp_gap"] if best_route else None,
        "best_cvrp_labels": best_route["labels"] if best_route else None,
        "routing_seconds": routing_seconds,
        **_hit_probabilities(((item["cvrp_gap"], item["shots"]) for item in evaluations),
                             total, "cvrp"),
    }, evaluations


def summarize(output: Path, *, routing_factory: Callable[[Instance], RoutingEvaluator] = Routing,
              routing_algorithm: str = "held-karp-v1") -> list[dict]:
    """Read saved evidence only: never invoke an optimizer or emulator."""
    rows = []
    routing_by_geometry: dict[tuple, RoutingEvaluator] = {}
    for path in sorted((output / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        instance = instance_from_record(run["instance"])
        ref = json.loads((output / "references" / f"{instance.id}.json").read_text())
        row = {key: run.get(key) for key in (
            "instance_id", "method", "seed", "encoding", "configuration_id", "instance_hash",
            "status", "shot_budget", "atom_count", "clustering_seconds",
            "embedding_seconds", "emulation_seconds", "pair_relative_error",
            "linear_relative_error", "scale", "error",
        )}
        row.update({key: run["configuration"].get(key)
                    for key in ("backend", "dt_ns", "interaction_cutoff", "max_bond_dim",
                                "precision", "optimize_qubit_ordering", "memory_mb", "timeout_seconds")})
        row.update(instance.ratios)
        row.update({"certificate_valid": True, "model_reference": ref["clustering"]["cost"],
                    "cvrp_reference": ref["cvrp"]["cost"],
                    "reference_status": ref["status"], "reference_seconds": ref["seconds"],
                    "model_reference_status": ref["clustering"]["status"],
                    "cvrp_reference_status": ref["cvrp"]["status"],
                    "gap_tolerance_percent": GAP_TOLERANCE_PERCENT})
        if (run["encoding"] not in (ENCODING, UNBALANCED_ENCODING, VARIABLE_REDUCTION_ENCODING)
                or run["instance_hash"] != fingerprint(run["instance"])):
            raise ValueError(f"Incompatible encoding or modified instance in {path}")
        if ref["instance_hash"] != run["instance_hash"]:
            raise ValueError(f"Reference-instance mismatch in {path}")
        cache_path = output / "evaluations" / path.name
        if run["status"] == "completed":
            if sum(run["counts"].values()) != run["shot_budget"]:
                raise ValueError(f"Shot budget mismatch in {path}")
            cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
            cache_key = {
                "run_hash": fingerprint(run), "reference_hash": fingerprint(ref),
                "evaluation_version": EVALUATION_VERSION, "routing_algorithm": routing_algorithm,
                "hit_thresholds_percent": list(HIT_THRESHOLDS_PERCENT),
            }
            if cached is not None and all(cached.get(key) == value for key, value in cache_key.items()):
                measured = cached["metrics"]
            else:
                geometry = instance.coordinates, instance.depot
                if geometry not in routing_by_geometry:
                    routing_by_geometry[geometry] = routing_factory(instance)
                measured, evaluations = metrics(instance, run["counts"], ref,
                                                routing_by_geometry[geometry],
                                                encoding=run["encoding"])
                save(cache_path, {**cache_key, "metrics": measured,
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
