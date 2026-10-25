"""Selectable S/M/L inputs and independently checked, reproducible geometry."""

from dataclasses import asdict, dataclass, fields
from itertools import product
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "seeds_s.json"
OUTPUT_PATH = ROOT / "outputs" / "all_demands"
TIER_CONFIGS = {"s": (4, 2, 7, 4), "m": (10, 3, 20, 8), "l": (15, 4, 30, 10)}


def _tier_for(n: int, vehicles: int, capacity: int) -> str:
    for tier, (customers, fleet, limit, _) in TIER_CONFIGS.items():
        if (n, vehicles, capacity) == (customers, fleet, limit):
            return tier
    raise ValueError("Supported tiers: S (4,2,7), M (10,3,20), L (15,4,30)")


@dataclass(frozen=True)
class Instance:
    tier: str
    instance_seed: int
    coordinates: tuple[tuple[float, float], ...]
    demands: tuple[int, ...]
    certificate: tuple[int, ...]
    vehicles: int
    capacity: int
    depot: tuple[float, float] = (50.0, 50.0)

    @property
    def id(self) -> str:
        return f"{self.tier}-{self.instance_seed}"

    @property
    def n(self) -> int:
        return len(self.demands)

    @property
    def distances(self) -> NDArray[np.float64]:
        points = np.asarray(self.coordinates)
        return np.linalg.norm(points[:, None] - points[None, :], axis=-1)

    @property
    def ratios(self) -> dict[str, float]:
        return {"rho": sum(self.demands) / (self.vehicles * self.capacity),
                "largest_demand_ratio": max(self.demands) / self.capacity}


def feasible(instance: Instance, labels: tuple[int, ...] | list[int]) -> bool:
    if len(labels) != instance.n or any(
        type(k) is not int or not 0 <= k < instance.vehicles for k in labels
    ):
        return False
    loads = [0] * instance.vehicles
    for demand, k in zip(instance.demands, labels, strict=True):
        loads[k] += demand
    return max(loads) <= instance.capacity


def validate(instance: Instance) -> None:
    tier = _tier_for(instance.n, instance.vehicles, instance.capacity)
    if instance.tier != tier:
        raise ValueError("Instance tier does not match N, K and Q")
    points = np.asarray(instance.coordinates)
    if points.shape != (instance.n, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid customer coordinates")
    if np.any((points < 0) | (points > 100)) or instance.depot != (50.0, 50.0):
        raise ValueError("Invalid coordinates or depot")
    maximum = TIER_CONFIGS[tier][3]
    if any(type(d) is not int or not 1 <= d <= maximum for d in instance.demands):
        raise ValueError(f"{tier.upper()} demands must be integers in 1..{maximum}")
    if not feasible(instance, instance.certificate):
        raise ValueError("Invalid internally computed feasibility witness")
    distances = instance.distances
    if not np.allclose(distances, distances.T):
        raise ValueError("Asymmetric distances")
    if np.any(distances[:, :, None] > distances[:, None, :] + distances[None, :, :] + 1e-10):
        raise ValueError("Triangle inequality violation")


def _feasibility_witness(instance: Instance) -> tuple[int, ...] | None:
    if instance.tier == "s":
        return next((labels for labels in product(range(instance.vehicles), repeat=instance.n)
                     if feasible(instance, labels)), None)
    order = sorted(range(instance.n), key=lambda i: instance.demands[i], reverse=True)
    loads = [0] * instance.vehicles
    labels = [0] * instance.n

    def place(index: int) -> bool:
        if index == instance.n:
            return True
        customer = order[index]
        seen = set()
        for vehicle in range(instance.vehicles):
            if loads[vehicle] in seen or loads[vehicle] + instance.demands[customer] > instance.capacity:
                continue
            seen.add(loads[vehicle])
            loads[vehicle] += instance.demands[customer]
            labels[customer] = vehicle
            if place(index + 1):
                return True
            loads[vehicle] -= instance.demands[customer]
        return False

    return tuple(labels) if place(0) else None


def load_tier(path: Path) -> list[Instance]:
    """Load and validate every supplied demand vector without changing the input."""
    data = json.loads(path.read_text())
    expected = {"N", "K", "Q", "demand_range", "demands", "coordinates", "depot"}
    if set(data) != expected:
        raise ValueError(f"Input must contain exactly {sorted(expected)}")
    if any(type(data[key]) is not int for key in ("N", "K", "Q")):
        raise ValueError("N, K, Q must be integers")
    tier = _tier_for(data["N"], data["K"], data["Q"])
    maximum = TIER_CONFIGS[tier][3]
    demand_range = data["demand_range"]
    if (not isinstance(demand_range, list) or len(demand_range) != 2
            or any(type(v) is not int for v in demand_range)
            or not 1 <= demand_range[0] <= demand_range[1] <= maximum):
        raise ValueError(f"demand_range must be [minimum, maximum] within {tier.upper()} range 1..{maximum}")
    vectors = data["demands"]
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("demands must be a nonempty array of explicit demand vectors")
    points = data["coordinates"]
    if (not isinstance(points, list) or len(points) != data["N"]
            or any(not isinstance(point, list) or len(point) != 2
                   or any(type(value) not in (int, float) or not np.isfinite(value)
                          or not 0 <= value <= 100 for value in point)
                   for point in points)):
        raise ValueError("coordinates must contain N finite numeric [x, y] pairs in [0, 100]")
    if data["depot"] != [50, 50]:
        raise ValueError("Depot must be [50, 50]")
    coordinates = tuple((float(x), float(y)) for x, y in points)
    depot = (float(data["depot"][0]), float(data["depot"][1]))
    instances = []
    for index, vector in enumerate(vectors):
        if (not isinstance(vector, list) or len(vector) != data["N"]
                or any(type(d) is not int or not demand_range[0] <= d <= demand_range[1]
                       for d in vector)):
            raise ValueError(f"Invalid demand vector at index {index}")
        instance = Instance(tier, index, coordinates, tuple(vector), (), data["K"], data["Q"], depot)
        witness = _feasibility_witness(instance)
        if witness is None:
            raise ValueError(f"Demand vector {index} is capacity-infeasible: {vector}")
        instance = Instance(tier, index, coordinates, tuple(vector), witness, data["K"], data["Q"], depot)
        validate(instance)
        instances.append(instance)
    return instances


def ensure_instances(input_path: Path = INPUT_PATH) -> list[Instance]:
    """Load all demand vectors without regenerating or rewriting input."""
    return load_tier(input_path)


def instance_record(instance: Instance) -> dict:
    return {**asdict(instance), **instance.ratios, "certificate_valid": True}


def instance_from_record(record: dict) -> Instance:
    """Restore and validate a saved instance, including JSON tuple conversions."""
    names = {field.name for field in fields(Instance)}
    if set(record) - names - {"rho", "largest_demand_ratio", "certificate_valid"}:
        raise ValueError("Unrecognized saved instance fields")
    data = {name: record[name] for name in names}
    instance = Instance(
        **{**data, "coordinates": tuple(map(tuple, record["coordinates"])),
           "demands": tuple(record["demands"]), "certificate": tuple(record["certificate"]),
           "depot": tuple(record["depot"])}
    )
    validate(instance)
    expected = {**instance.ratios, "certificate_valid": True}
    if any(record[key] != value for key, value in expected.items() if key in record):
        raise ValueError("Saved instance metadata does not match its validated values")
    return instance


def notebook_inputs(input_file: str | Path = "seeds_s.json",
                    output_directory: str | Path | None = None) -> tuple[Path, Path, list[Instance]]:
    """Resolve notebook paths independently of the kernel's working directory."""
    input_path = Path(input_file).expanduser()
    if not input_path.is_absolute():
        input_path = (ROOT / "inputs" if len(input_path.parts) == 1 else ROOT.parent) / input_path
    input_path = input_path.resolve()
    instances = ensure_instances(input_path)
    if output_directory is None:
        output = OUTPUT_PATH if input_path == INPUT_PATH.resolve() else ROOT / "outputs" / input_path.stem
    else:
        output = Path(output_directory).expanduser()
        if not output.is_absolute():
            output = ROOT.parent / output
    output = output.resolve()
    manifest = output / "instances.json"
    if manifest.exists():
        saved = [instance_from_record(record) for record in json.loads(manifest.read_text())]
        if saved != instances:
            raise ValueError("Output directory belongs to different inputs/geometry; choose another output_directory")
    return input_path, output, instances
