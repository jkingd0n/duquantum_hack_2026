"""Ground-state and raw-bitstring evidence shared by notebooks and Python calls."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import fingerprint, save
from .instances import Instance
from .models import ENCODING, Qubo, assignment_view, bit_layout, bitstring_breakdown, decode, objective


def save_target_evidence(instance: Instance, target: Qubo, check: dict, output: Path) -> None:
    save(output / "groundstates" / f"{instance.id}-exact.json", {
        "instance_id": instance.id, "instance_hash": fingerprint(asdict(instance)),
        "variable_order": target.variable_order, "bit_layout": bit_layout(instance, target), **check,
    })


def save_sample_evidence(instance: Instance, target: Qubo, exact: dict,
                         run: dict, output: Path) -> dict:
    if run["instance_hash"] != fingerprint(asdict(instance)):
        raise ValueError("Saved quantum shots do not match the current instance")
    if run["encoding"] != ENCODING or run["variable_order"] != target.variable_order:
        raise ValueError("Saved quantum bit order or encoding does not match the target QUBO")
    evidence = {
        "instance_id": instance.id, "instance_hash": run["instance_hash"],
        "source_run_hash": fingerprint(run), "variable_order": target.variable_order,
        "bit_layout": bit_layout(instance, target),
        "exact_target_ground_energy": exact["minimum_energy"],
        "exact_target_ground_bitstrings": exact["ground_state_bitstrings"],
        "status": run["status"],
    }
    if run["status"] != "completed":
        evidence["error"] = run["error"]
        evidence["summary"] = {
            "instance": instance.id, "status": run["status"], "error": run["error"],
            **assignment_view(instance, None), "model_cost": None,
        }
    else:
        counts = run["counts"]
        if not counts or any(type(count) is not int or count <= 0 for count in counts.values()):
            raise ValueError("A completed quantum run must contain positive raw shot counts")
        total = sum(counts.values())
        if total != run["shot_budget"]:
            raise ValueError("Saved quantum shot counts do not match the shot budget")
        ground_strings = set(exact["ground_state_bitstrings"])
        measured = []
        for bitstring, count in counts.items():
            labels = decode(instance, bitstring)
            measured.append({
                **assignment_view(instance, labels),
                "bitstring": bitstring, **bitstring_breakdown(instance, bitstring),
                "count": count, "probability": count / total,
                "target_energy": target.energy(list(map(int, bitstring))),
                "feasible": labels is not None, "customer_groups": labels,
                "model_cost": objective(instance, labels) if labels is not None else None,
                "exact_target_ground_state": bitstring in ground_strings,
            })
        measured.sort(key=lambda row: (row["target_energy"], -row["count"], row["bitstring"]))
        observed_ground = [row for row in measured if row["exact_target_ground_state"]]
        lowest_energy = measured[0]["target_energy"]
        lowest_samples = [row for row in measured
                          if np.isclose(row["target_energy"], lowest_energy, atol=1e-8, rtol=0)]
        best_feasible = min((row for row in measured if row["feasible"]),
                            key=lambda row: row["model_cost"], default=None)
        evidence.update({
            "observed_ground_states": observed_ground,
            "lowest_sampled_energy": lowest_energy, "lowest_sampled_states": lowest_samples,
            "sampled_states": measured,
            "summary": {
                "instance": instance.id, "status": run["status"], "actual_shots": total,
                **assignment_view(instance, best_feasible["customer_labels"] if best_feasible else None),
                "model_cost": best_feasible["model_cost"] if best_feasible else None,
                "feasible_shots": sum(row["count"] for row in measured if row["feasible"]),
                "exact_ground_shots": sum(row["count"] for row in observed_ground),
                "lowest_sampled_target_energy": lowest_energy,
            },
        })
    save(output / "groundstates" / f"{instance.id}-quantum-global-{run['seed']}.json", evidence)
    return evidence
