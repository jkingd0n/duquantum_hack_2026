"""Read-only comparisons of previously saved classical and quantum QUBO shots."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import clustering_metrics, fingerprint
from .instances import ROOT, feasible, instance_from_record
from .models import (
    ENCODING, UNBALANCED_ENCODING, VARIABLE_REDUCTION_ENCODING, assignment_view, objective,
)

QUBO_METHODS = {
    "classical-qubo-sa": "Classical slack",
    "classical-unbalanced-sa": "Classical unbalanced",
    "classical-variable_reduction-sa": "Classical variable reduction",
    "quantum-global": "Quantum slack",
    "quantum-unbalanced": "Quantum unbalanced",
    "quantum-variable_reduction": "Quantum variable reduction",
}
FORMULATIONS = {
    "slack": ENCODING,
    "unbalanced_penalty": UNBALANCED_ENCODING,
    "variable_reduction": VARIABLE_REDUCTION_ENCODING,
}


def saved_comparisons(directories: dict[str, str | Path], seed: int = 0) -> pd.DataFrame:
    """Decode saved counts only; never solve, emulate, route, or write evidence."""
    if type(seed) is not int or seed < 0:
        raise ValueError("The saved sampling seed must be a nonnegative integer")
    rows = []
    for tier, directory in directories.items():
        if tier.upper() not in ("S", "M"):
            raise ValueError(f"Unsupported comparison tier: {tier}")
        output = Path(directory).expanduser()
        if not output.is_absolute():
            output = ROOT.parent / output
        output = output.resolve()
        manifest = output / "instances.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"No saved instance manifest at {manifest}; select a completed run directory")
        instances = [instance_from_record(record) for record in json.loads(manifest.read_text())]
        if not instances or any(instance.tier.upper() != tier.upper() for instance in instances):
            raise ValueError(f"Saved manifest does not belong to tier {tier}: {manifest}")
        if len({instance.id for instance in instances}) != len(instances):
            raise ValueError(f"Duplicate instance IDs in {manifest}")
        for instance in instances:
            reference = json.loads((output / "references" / f"{instance.id}.json").read_text())
            instance_hash = fingerprint(asdict(instance))
            if reference["instance_hash"] != instance_hash:
                raise ValueError(f"Saved reference does not match {instance.id} in {output}")
            for method, label in QUBO_METHODS.items():
                formulation = ("variable_reduction" if "variable_reduction" in method else
                               "unbalanced_penalty" if "unbalanced" in method else "slack")
                expected_encoding = FORMULATIONS[formulation]
                path = output / "runs" / f"{instance.id}-{method}-{seed}.json"
                row = {
                    "tier": tier.upper(), "instance_id": instance.id, "method": method,
                    "solver": label, "formulation": formulation,
                    "seed": seed, "source_directory": str(output), "run_file": str(path),
                    "reference_cost": reference["clustering"]["cost"],
                    "reference_status": reference["clustering"]["status"],
                    "reference_labels": assignment_view(
                        instance, reference["clustering"]["labels"])["customer_labels"],
                    "coordinates": instance.coordinates, "demands": instance.demands,
                    "depot": instance.depot, "capacity": instance.capacity,
                    "vehicles": instance.vehicles,
                    "status": "not_saved", "outcome": "Not saved",
                    "bits": None, "shot_budget": None, "actual_shots": None, "feasible_shots": None,
                    "feasibility_rate": None, "optimal_hit_probability": None,
                    "optimal_shots": None, "best_model_gap": None, "model_cost": None,
                    **assignment_view(instance, None),
                    "lambda1": None, "lambda2": None, "assignment_penalty": None,
                    "alpha": None, "beta": None, "auxiliary_mapping": None,
                    "backend": None, "solver_seconds": None, "error": None,
                    "saved_cvrp_gap": None, "cvrp_evidence": "Not saved",
                    "configuration_id": None,
                }
                if not path.is_file():
                    rows.append(row)
                    continue
                run = json.loads(path.read_text())
                if (run["instance_hash"] != instance_hash or fingerprint(run["instance"]) != instance_hash
                        or run["method"] != method or run["seed"] != seed
                        or run["encoding"] != expected_encoding):
                    raise ValueError(f"Incompatible saved run: {path}")
                configuration = run["configuration"]
                row.update({
                    "status": run["status"], "outcome": run["status"],
                    "bits": run["atom_count"], "shot_budget": run["shot_budget"],
                    "lambda1": configuration.get("lambda1"), "lambda2": configuration.get("lambda2"),
                    "alpha": configuration.get("alpha"), "beta": configuration.get("beta"),
                    "auxiliary_mapping": configuration.get("auxiliary_mapping"),
                    "assignment_penalty": configuration.get("assignment_penalty",
                                                             configuration.get("penalty")),
                    "backend": configuration.get("backend"),
                    "configuration_id": run.get("configuration_id"), "error": run.get("error"),
                })
                runtime_keys = (("embedding_seconds", "emulation_seconds") if method.startswith("quantum")
                                else ("clustering_seconds",))
                times = [run.get(key) for key in runtime_keys]
                row["solver_seconds"] = sum(times) if all(t is not None for t in times) else None
                if run["status"] == "completed":
                    counts = run["counts"]
                    if not counts or sum(counts.values()) != run["shot_budget"]:
                        raise ValueError(f"Saved shot budget mismatch: {path}")
                    if any(len(bits) != run["atom_count"] for bits in counts):
                        raise ValueError(f"Saved bit lengths do not match atom count: {path}")
                    measured = clustering_metrics(instance, counts, reference["clustering"]["cost"],
                                                  encoding=run["encoding"])
                    labels = measured["best_model_labels"]
                    row.update({
                        "actual_shots": measured["actual_shots"],
                        "feasible_shots": measured["feasible_shots"],
                        "feasibility_rate": measured["feasibility_rate"],
                        "optimal_hit_probability": measured["model_hit_probability_0.1pct"],
                        "optimal_shots": round(measured["model_hit_probability_0.1pct"]
                                               * measured["actual_shots"]),
                        "best_model_gap": measured["best_model_gap"],
                        **assignment_view(instance, labels),
                        "model_cost": objective(instance, labels) if labels is not None else None,
                        "outcome": ("Feasible sample found" if labels is not None else
                                    f"No feasible sample (0/{measured['actual_shots']})"),
                    })
                cache_path = output / "evaluations" / path.name
                if cache_path.is_file():
                    cached = json.loads(cache_path.read_text())
                    if (cached.get("run_hash") == fingerprint(run)
                            and cached.get("reference_hash") == fingerprint(reference)):
                        row["saved_cvrp_gap"] = cached["metrics"].get("best_cvrp_gap")
                        row["cvrp_evidence"] = "Saved" if row["saved_cvrp_gap"] is not None else "No saved gap"
                    else:
                        row["cvrp_evidence"] = "Stale cache; not used"
                rows.append(row)
    if not rows:
        raise ValueError("Select at least one saved S or M output directory")
    return pd.DataFrame(rows)


def comparison_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Pool shot probabilities; average best gaps only where a feasible result exists."""
    rows = []
    for (tier, method, label), group in runs.groupby(["tier", "method", "solver"], sort=False):
        completed = group[group["status"] == "completed"]
        shots = completed["actual_shots"].sum()
        gaps = completed["best_model_gap"].dropna()
        rows.append({
            "tier": tier, "method": method, "solver": label,
            "expected_runs": len(group), "completed_runs": len(completed),
            "failed_runs": int((~group["status"].isin(["completed", "not_saved"])).sum()),
            "missing_runs": int((group["status"] == "not_saved").sum()),
            "feasible_result_runs": int((completed["feasible_shots"] > 0).sum()),
            "actual_shots": int(shots),
            "feasibility_percent": 100 * completed["feasible_shots"].sum() / shots if shots else None,
            "optimal_hit_percent": 100 * completed["optimal_shots"].sum() / shots if shots else None,
            "mean_best_gap_percent": gaps.mean() if len(gaps) else None,
            "worst_best_gap_percent": gaps.max() if len(gaps) else None,
            "median_solver_seconds": completed["solver_seconds"].median()
            if completed["solver_seconds"].notna().any() else None,
        })
    return pd.DataFrame(rows)


def plot_saved_comparisons(runs: pd.DataFrame, summary: pd.DataFrame):
    """Show missing/failed-only methods explicitly rather than plotting them as zero."""
    import matplotlib.pyplot as plt

    tiers = list(dict.fromkeys(runs["tier"]))
    figure, axes = plt.subplots(1, len(tiers), figsize=(9 * len(tiers), 4.5), squeeze=False,
                                constrained_layout=True)
    short_labels = [label.replace(" ", "\n", 1).replace("variable reduction", "variable\nreduction")
                    for label in QUBO_METHODS.values()]
    for ax, tier in zip(axes[0], tiers, strict=True):
        table = summary[summary["tier"] == tier].set_index("method").reindex(QUBO_METHODS)
        x = np.arange(len(QUBO_METHODS))
        for offset, column, color, label in (
            (-.18, "feasibility_percent", "#4477aa", "Feasible shots"),
            (.18, "optimal_hit_percent", "#228833", "Within 0.1% of optimum"),
        ):
            values = table[column]
            bars = ax.bar(x + offset, values.fillna(0), .34, color=color, label=label)
            for bar, value in zip(bars, values, strict=True):
                if pd.notna(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                            f"{value:.1f}", ha="center", fontsize=8)
        for index, (_, row) in enumerate(table.iterrows()):
            if row["completed_runs"] == 0:
                note = "Not saved" if row["missing_runs"] == row["expected_runs"] else "No completed runs"
                ax.text(index, 7, note, ha="center", fontsize=8, rotation=25, color="#777777")
        ax.set(title=f"{tier}: raw-shot probabilities", ylabel="Percent of completed raw shots",
               xticks=x, xticklabels=short_labels, ylim=(0, 115))
        ax.legend(fontsize=8, loc="upper right")
    figure.set_label("shot_probabilities")
    yield figure


    figure, axes = plt.subplots(1, len(tiers), figsize=(9 * len(tiers), 4.5), squeeze=False,
                                constrained_layout=True)
    for ax, tier in zip(axes[0], tiers, strict=True):
        for index, method in enumerate(QUBO_METHODS):
            group = runs[(runs["tier"] == tier) & (runs["method"] == method)]
            values = group["best_model_gap"].dropna()
            if len(values):
                offsets = np.linspace(-.12, .12, len(values)) if len(values) > 1 else [0]
                ax.scatter(index + np.asarray(offsets), values, color="#4477aa", alpha=.7)
                ax.scatter(index, values.mean(), color="black", marker="D", s=40,
                           label="Mean" if index == 0 else None)
            else:
                note = "Not saved" if (group["status"] == "not_saved").all() else "No feasible result"
                ax.text(index, .05, note, transform=ax.get_xaxis_transform(), ha="center",
                        rotation=25, fontsize=8, color="#777777")
        ax.axhline(0, color="#228833", linestyle="--", linewidth=1, label="Saved exact optimum")
        ax.set(title=f"{tier}: best feasible clustering gap", ylabel="Gap to saved reference (%)",
               xticks=range(len(QUBO_METHODS)), xticklabels=short_labels)
        ax.legend(fontsize=8)
    figure.set_label("best_clustering_gaps")
    yield figure


def plot_saved_clusters(runs: pd.DataFrame, selections: dict[str, list[str]] | None = None):
    """Map the saved reference and best feasible sample per QUBO method, without solving."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if selections is None:
        requested = list(runs[["tier", "instance_id"]].drop_duplicates().itertuples(index=False, name=None))
    else:
        requested = []
        for tier, identifiers in selections.items():
            if not isinstance(identifiers, list) or not identifiers or not all(
                isinstance(identifier, str) for identifier in identifiers
            ):
                raise ValueError("Cluster selections require a nonempty list of instance IDs per tier")
            requested.extend((tier.upper(), identifier) for identifier in identifiers)
    if not requested:
        raise ValueError("Select at least one saved instance for cluster maps")
    colors = ("#4477aa", "#ee6677", "#228833")
    for tier, identifier in requested:
        group = runs[(runs["tier"] == tier) & (runs["instance_id"] == identifier)]
        if len(group) != len(QUBO_METHODS) or set(group["method"]) != set(QUBO_METHODS):
            raise ValueError(f"Select a saved instance with one row per QUBO method: {tier}/{identifier}")
        first = group.iloc[0]
        instance = instance_from_record({
            "tier": tier.lower(), "instance_seed": int(identifier.split("-")[1]),
            "coordinates": first["coordinates"], "demands": first["demands"],
            "depot": first["depot"], "vehicles": first["vehicles"], "capacity": first["capacity"],
            "certificate": first["reference_labels"],
        })
        coordinates = np.asarray(instance.coordinates)
        all_points = np.vstack([coordinates, instance.depot])
        margin = max(float(np.ptp(all_points, axis=0).max()) * .15, 2.)
        lower, upper = all_points.min(axis=0) - margin, all_points.max(axis=0) + margin
        figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True,
                                    constrained_layout=True)
        panels = [{
            "solver": "Saved exact reference", "customer_labels": first["reference_labels"],
            "model_cost": first["reference_cost"], "best_model_gap": 0.,
            "outcome": first["reference_status"],
        }, *[group[group["method"] == method].iloc[0].to_dict() for method in QUBO_METHODS]]
        for ax, panel in zip(axes.flat, panels):
            labels = panel["customer_labels"]
            if labels is None:
                ax.scatter(*coordinates.T, c="#bbbbbb", s=65, edgecolors="white")
                ax.text(.5, .95, panel["outcome"], transform=ax.transAxes,
                        ha="center", va="top", color="#555555", fontsize=10)
                title = panel["solver"] + "\nNo saved feasible assignment"
            else:
                if not feasible(instance, labels):
                    raise ValueError(f"Cannot plot infeasible saved labels: {tier}/{identifier}/{panel['solver']}")
                loads = []
                for vehicle in range(instance.vehicles):
                    members = np.array(labels) == vehicle
                    ax.scatter(*coordinates[members].T, c=colors[vehicle], s=80, edgecolors="white")
                    load = sum(demand for demand, assigned in zip(instance.demands, labels, strict=True)
                               if assigned == vehicle)
                    loads.append(f"V{vehicle}: {load}/{instance.capacity}")
                gap = panel["best_model_gap"]
                gap_text = f"{gap:.2f}%" if pd.notna(gap) else "N/A"
                title = (f"{panel['solver']}\nCost {panel['model_cost']:.2f} | gap {gap_text}\n"
                         + "; ".join(loads))
            ax.scatter(*instance.depot, marker="*", s=150, c="black", zorder=4)
            for customer, (point, demand) in enumerate(zip(coordinates, instance.demands, strict=True)):
                ax.annotate(f"{customer} (d={demand})", point, xytext=(4, 5),
                            textcoords="offset points", fontsize=8)
            ax.set(title=title, xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                   xlabel="Customer x coordinate", ylabel="Customer y coordinate")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.2)
        legend_ax = axes.flat[-1]
        legend_ax.axis("off")
        handles = [Line2D([], [], marker="o", color=color, linestyle="None",
                          label=f"Vehicle {vehicle} cluster", markersize=8)
                   for vehicle, color in enumerate(colors[:instance.vehicles])]
        handles.append(Line2D([], [], marker="*", color="black", linestyle="None",
                              label="Saved depot (not part of clustering cost)", markersize=12))
        legend_ax.legend(handles=handles, loc="upper left", fontsize=10)
        legend_ax.text(0, .5, "Labels: customer ID (d=demand)\nV loads: assigned demand / capacity\n"
                       "Colors use canonical vehicle labels.\nGray customers: no feasible result available.\n\n"
                       "Each solver shows its best feasible saved sample.\n"
                       "No routes, repair, or new optimization are drawn.",
                       transform=legend_ax.transAxes, va="top", fontsize=10, linespacing=1.6)
        figure.suptitle(f"{tier} / {identifier}: saved clusters, sampling seed {first['seed']}", fontsize=14)
        figure.set_label(f"clusters_{identifier}")
        yield figure
