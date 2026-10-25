"""Small-tier inputs and independently checked, reproducible geometry."""

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "seeds_s.json"
OUTPUT_PATH = ROOT / "outputs" / "all_demands"


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
    if (instance.tier, instance.n, instance.vehicles, instance.capacity) != ("s", 4, 2, 7):
        raise ValueError("Only prescribed S instances (N=4, K=2, Q=7) are supported")
    points = np.asarray(instance.coordinates)
    if points.shape != (instance.n, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid customer coordinates")
    if np.any((points < 0) | (points > 100)) or instance.depot != (50.0, 50.0):
        raise ValueError("Invalid coordinates or depot")
    if any(type(d) is not int or not 1 <= d <= 4 for d in instance.demands):
        raise ValueError("S demands must be integers in 1..4")
    if not feasible(instance, instance.certificate):
        raise ValueError("Invalid internally computed feasibility witness")
    distances = instance.distances
    if not np.allclose(distances, distances.T):
        raise ValueError("Asymmetric distances")
    if np.any(distances[:, :, None] > distances[:, None, :] + distances[None, :, :] + 1e-10):
        raise ValueError("Triangle inequality violation")


def load_tier(path: Path) -> list[Instance]:
    """Load and validate every supplied demand vector without changing the input."""
    data = json.loads(path.read_text())
    expected = {"N", "K", "Q", "demand_range", "demands", "coordinates", "depot"}
    if set(data) != expected:
        raise ValueError(f"Input must contain exactly {sorted(expected)}")
    if any(type(data[key]) is not int for key in ("N", "K", "Q")):
        raise ValueError("N, K, Q must be integers")
    if (data["N"], data["K"], data["Q"]) != (4, 2, 7):
        raise ValueError("Only S (N=4, K=2, Q=7) is implemented")
    demand_range = data["demand_range"]
    if (not isinstance(demand_range, list) or len(demand_range) != 2
            or any(type(v) is not int for v in demand_range)
            or not 1 <= demand_range[0] <= demand_range[1] <= 4):
        raise ValueError("demand_range must be [minimum, maximum] within S range 1..4")
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
        raise ValueError("S depot must be [50, 50]")
    coordinates = tuple((float(x), float(y)) for x, y in points)
    depot = (float(data["depot"][0]), float(data["depot"][1]))
    instances = []
    for index, vector in enumerate(vectors):
        if (not isinstance(vector, list) or len(vector) != data["N"]
                or any(type(d) is not int or not demand_range[0] <= d <= demand_range[1]
                       for d in vector)):
            raise ValueError(f"Invalid demand vector at index {index}")
        instance = Instance("s", index, coordinates, tuple(vector), (), data["K"], data["Q"], depot)
        witness = next((labels for labels in product(range(data["K"]), repeat=data["N"])
                        if feasible(instance, labels)), None)
        if witness is None:
            raise ValueError(f"Demand vector {index} is capacity-infeasible: {vector}")
        instance = Instance("s", index, coordinates, tuple(vector), witness, data["K"], data["Q"], depot)
        validate(instance)
        instances.append(instance)
    return instances


def ensure_instances(input_path: Path = INPUT_PATH) -> list[Instance]:
    """Load all demand vectors without regenerating or rewriting input."""
    return load_tier(input_path)


def instance_record(instance: Instance) -> dict:
    return {**asdict(instance), **instance.ratios, "certificate_valid": True}
