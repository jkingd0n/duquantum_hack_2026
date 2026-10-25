"""A symmetric x.T Q x QUBO with an explicit constant offset."""

from dataclasses import dataclass
from itertools import permutations
import json
from math import ceil, log2

import numpy as np
from numpy.typing import NDArray

from .instances import ROOT, Instance, feasible

ENCODING = "binary-slack-v1"
UNBALANCED_ENCODING = "unbalanced-v1"
VARIABLE_REDUCTION_ENCODING = "variable-reduction-complemented-v1"


@dataclass(frozen=True)
class EncodingLayout:
    assignment_size: int
    slack_width: int
    vehicles: int

    @property
    def size(self) -> int:
        return self.assignment_size + self.vehicles * self.slack_width

    def slack_slice(self, vehicle: int) -> slice:
        start = self.assignment_size + vehicle * self.slack_width
        return slice(start, start + self.slack_width)


def encoding_layout(instance: Instance) -> EncodingLayout:
    return EncodingLayout(instance.n * instance.vehicles,
                          ceil(log2(instance.capacity + 1)), instance.vehicles)


def canonical(labels: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    return tuple(mapping.setdefault(k, len(mapping)) for k in labels)


def _assignment_bits(instance: Instance, labels: tuple[int, ...] | list[int]) -> str:
    return "".join(str(int(label == vehicle))
                   for label in labels for vehicle in range(instance.vehicles))


def assignment_view(instance: Instance, labels: tuple[int, ...] | list[int] | None) -> dict:
    """Use the same vehicle-label-independent display for classical and quantum results."""
    if labels is None:
        return {"customer_labels": None, "canonical_assignment_bits": None}
    if len(labels) != instance.n or any(
        type(vehicle) is not int or not 0 <= vehicle < instance.vehicles for vehicle in labels
    ):
        raise ValueError("Customer labels do not match the instance")
    normalized = canonical(labels)
    return {"customer_labels": list(normalized),
            "canonical_assignment_bits": _assignment_bits(instance, normalized)}


def objective(instance: Instance, labels: tuple[int, ...] | list[int]) -> float:
    distances = instance.distances
    return float(sum(distances[i, j] for i in range(instance.n)
                     for j in range(i) if labels[i] == labels[j]))


@dataclass
class Qubo:
    matrix: NDArray[np.float64]
    constant: float
    penalty: float
    normalization: float
    slack_bits: int
    variable_order: list[str]
    encoding: str = ENCODING

    def energy(self, bits: NDArray | list[int]) -> float:
        z = np.asarray(bits)
        return float(z @ self.matrix @ z + self.constant)


def _objective_matrix(instance: Instance, size: int) -> tuple[NDArray[np.float64], float]:
    normalization = max(float(instance.distances.max()), 1.0)
    matrix = np.zeros((size, size))
    for i in range(instance.n):
        for j in range(i):
            for vehicle in range(instance.vehicles):
                a, c = i * instance.vehicles + vehicle, j * instance.vehicles + vehicle
                matrix[a, c] += instance.distances[i, j] / normalization / 2
                matrix[c, a] = matrix[a, c]
    return matrix, normalization


def _add_square(matrix: NDArray[np.float64], coefficients: NDArray[np.float64],
                rhs: int, weight: float) -> float:
    matrix += weight * np.outer(coefficients, coefficients)
    matrix[np.diag_indices_from(matrix)] -= 2 * weight * rhs * coefficients
    return weight * rhs**2


def build_qubo(instance: Instance, penalty: float | None = None) -> Qubo:
    n, k, q = instance.n, instance.vehicles, instance.capacity
    layout = encoding_layout(instance)
    b, size = layout.slack_width, layout.size
    matrix, normalization = _objective_matrix(instance, size)
    upper = objective(instance, instance.certificate) / normalization
    weight = upper + 1 if penalty is None else penalty
    if not np.isfinite(weight) or weight <= upper:
        raise ValueError("Penalty must exceed feasible-certificate normalized objective")
    constant = 0.0
    for i in range(n):
        coefficients = np.zeros(size)
        coefficients[i * k:(i + 1) * k] = 1
        constant += _add_square(matrix, coefficients, 1, weight)
    for vehicle in range(k):
        coefficients = np.zeros(size)
        coefficients[vehicle:n * k:k] = instance.demands
        coefficients[layout.slack_slice(vehicle)] = 2.0 ** np.arange(b)
        constant += _add_square(matrix, coefficients, q, weight)
    names = [f"x_{i}_{v}" for i in range(n) for v in range(k)]
    names += [f"y_{v}_{bit}" for v in range(k) for bit in range(b)]
    return Qubo(matrix, constant, weight, normalization, b, names)


def unbalanced_penalty_weights(instance: Instance, lambda1: float | None = None,
                               lambda2: float | None = None) -> tuple[float, float]:
    """Read saved per-tier benchmark presets; explicit weights override each independently."""
    presets = json.loads((ROOT / "inputs" / "unbalanced_penalties.json").read_text())
    if not isinstance(presets, dict) or not isinstance(presets.get(instance.tier), dict):
        raise ValueError(f"No unbalanced penalty defaults for tier {instance.tier}")
    defaults = presets[instance.tier]
    first = defaults.get("lambda1") if lambda1 is None else lambda1
    second = defaults.get("lambda2") if lambda2 is None else lambda2
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value <= 0 for value in (first, second)):
        raise ValueError("Unbalanced lambda1 and lambda2 must be finite and positive")
    return float(first), float(second)


def build_unbalanced_qubo(instance: Instance, lambda1: float | None = None,
                          lambda2: float | None = None,
                          assignment_penalty: float | None = None) -> Qubo:
    """H = F/normalization + beta*assignment_squares + sum(-lambda1*h + lambda2*h^2)."""
    lambda1, lambda2 = unbalanced_penalty_weights(instance, lambda1, lambda2)
    n, k, q = instance.n, instance.vehicles, instance.capacity
    size = n * k
    matrix, normalization = _objective_matrix(instance, size)
    weight = (objective(instance, instance.certificate) / normalization + 1
              if assignment_penalty is None else assignment_penalty)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("Unbalanced assignment penalty must be finite and positive")
    constant = 0.0
    for i in range(n):
        coefficients = np.zeros(size)
        coefficients[i * k:(i + 1) * k] = 1
        constant += _add_square(matrix, coefficients, 1, weight)
    for vehicle in range(k):
        coefficients = np.zeros(size)
        coefficients[vehicle:size:k] = instance.demands
        constant += _add_square(matrix, coefficients, q, lambda2)
        matrix[np.diag_indices(size)] += lambda1 * coefficients
        constant -= lambda1 * q
    return Qubo(matrix, constant, weight, normalization, 0,
                [f"x_{i}_{v}" for i in range(n) for v in range(k)], UNBALANCED_ENCODING)


def variable_reduction_penalties(instance: Instance, alpha: float | None = None,
                                 beta: float | None = None) -> tuple[float, float]:
    default = objective(instance, instance.certificate) / max(float(instance.distances.max()), 1.) + 1
    first = default if alpha is None else alpha
    second = default if beta is None else beta
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not np.isfinite(value) or value <= 0 for value in (first, second)):
        raise ValueError("Variable-reduction alpha and beta must be finite and positive")
    return first, second


def build_variable_reduction_qubo(instance: Instance, alpha: float | None = None,
                                  beta: float | None = None) -> Qubo:
    """Photo: alpha*(load/Q - 1 - b)^2, represented exactly with b = 1 - z."""
    alpha, beta = variable_reduction_penalties(instance, alpha, beta)
    n, k = instance.n, instance.vehicles
    size = n * k + k
    matrix, normalization = _objective_matrix(instance, size)
    constant = 0.
    for customer in range(n):
        coefficients = np.zeros(size)
        coefficients[customer * k:(customer + 1) * k] = 1
        constant += _add_square(matrix, coefficients, 1, beta)
    for vehicle in range(k):
        coefficients = np.zeros(size)
        coefficients[vehicle:n * k:k] = np.asarray(instance.demands) / instance.capacity
        coefficients[n * k + vehicle] = 1
        constant += _add_square(matrix, coefficients, 2, alpha)
    names = [f"x_{i}_{v}" for i in range(n) for v in range(k)] + [f"z_{v}" for v in range(k)]
    return Qubo(matrix, constant, beta, normalization, 1, names, VARIABLE_REDUCTION_ENCODING)


def decode_assignment(instance: Instance, bitstring: str) -> tuple[int, ...] | None:
    if len(bitstring) != instance.n * instance.vehicles:
        raise ValueError("Assignment bitstring length does not match N*K")
    if set(bitstring) - {"0", "1"}:
        raise ValueError("Nonbinary candidate")
    x = np.array([int(c) for c in bitstring])
    x = x.reshape(instance.n, instance.vehicles)
    if not np.all(x.sum(axis=1) == 1):
        return None
    labels = tuple(int(v) for v in x.argmax(axis=1))
    return canonical(labels) if feasible(instance, labels) else None


def decode(instance: Instance, bitstring: str, *, encoding: str = ENCODING) -> tuple[int, ...] | None:
    if encoding == UNBALANCED_ENCODING:
        return decode_assignment(instance, bitstring)
    if encoding == VARIABLE_REDUCTION_ENCODING:
        expected_size = instance.n * instance.vehicles + instance.vehicles
    elif encoding == ENCODING:
        expected_size = encoding_layout(instance).size
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    if len(bitstring) != expected_size:
        raise ValueError("Bitstring length does not match encoding")
    if set(bitstring) - {"0", "1"}:
        raise ValueError("Nonbinary candidate")
    return decode_assignment(instance, bitstring[:instance.n * instance.vehicles])


def encode(instance: Instance, labels: tuple[int, ...]) -> str:
    if not feasible(instance, labels):
        raise ValueError("Cannot encode infeasible assignment")
    b = encoding_layout(instance).slack_width
    bits = list(map(int, _assignment_bits(instance, labels)))
    for v in range(instance.vehicles):
        slack = instance.capacity - sum(d for i, d in enumerate(instance.demands) if labels[i] == v)
        bits.extend((slack >> bit) & 1 for bit in range(b))
    return "".join(map(str, bits))


def residuals(instance: Instance, bitstring: str) -> tuple[list[int], list[int]]:
    decode(instance, bitstring)
    layout = encoding_layout(instance)
    bits = np.array(list(map(int, bitstring)))
    x = bits[:layout.assignment_size].reshape(instance.n, instance.vehicles)
    slack = bits[layout.assignment_size:].reshape(instance.vehicles, layout.slack_width)
    slack = slack @ (2 ** np.arange(layout.slack_width))
    return (x.sum(axis=1) - 1).tolist(), (np.array(instance.demands) @ x + slack - instance.capacity).tolist()


def bit_layout(instance: Instance, target: Qubo) -> list[dict]:
    if len(target.variable_order) != encoding_layout(instance).size:
        raise ValueError("Variable order does not match encoding")
    rows = []
    for index, name in enumerate(target.variable_order):
        kind, first, second = name.split("_")
        rows.append({"position_from_left": index + 1, "python_index": index,
                     "variable": name, "kind": "assignment" if kind == "x" else "slack",
                     "customer": int(first) if kind == "x" else None,
                     "vehicle": int(second) if kind == "x" else int(first),
                     "slack_weight": 2 ** int(second) if kind == "y" else None})
    return rows


def bitstring_breakdown(instance: Instance, bitstring: str) -> dict:
    decode(instance, bitstring)
    layout = encoding_layout(instance)
    parts = [bitstring[:layout.assignment_size]]
    result = {"assignment_bits": parts[0], "slack_bits": bitstring[layout.assignment_size:]}
    for vehicle in range(instance.vehicles):
        bits = bitstring[layout.slack_slice(vehicle)]
        result[f"slack_bits_vehicle_{vehicle}"] = bits
        result[f"slack_value_vehicle_{vehicle}"] = sum(int(bit) * 2**r for r, bit in enumerate(bits))
        parts.append(bits)
    return {**result, "grouped_bitstring": " | ".join(parts)}


def validate_small_qubo(instance: Instance, reference: float) -> dict:
    qubo = build_qubo(instance)
    size = len(qubo.variable_order)
    if size > 16:
        raise ValueError("Exhaustive QUBO validation is restricted to small instances")
    bits = ((np.arange(2**size)[:, None] >> np.arange(size)) & 1)
    energies = np.einsum("bi,ij,bj->b", bits, qubo.matrix, bits) + qubo.constant
    ground = np.flatnonzero(np.isclose(energies, energies.min(), atol=1e-8, rtol=0))
    partitions = set()
    for index in ground:
        bitstring = "".join(map(str, bits[index]))
        labels = decode(instance, bitstring)
        assignment, capacity = residuals(instance, bitstring)
        if labels is None or any(assignment + capacity):
            raise AssertionError("Infeasible target QUBO ground state")
        if not np.isclose(objective(instance, labels), reference):
            raise AssertionError("QUBO and constrained optima disagree")
        partitions.add(labels)
    return {"checked_bitstrings": 2**size, "ground_states": len(ground),
            "ground_state_bitstrings": ["".join(map(str, bits[index])) for index in ground],
            "distinct_ground_partitions": len(partitions), "passed": True,
            "minimum_energy": float(energies.min()),
            "validation_method": "exhaustive-bitstrings", "exhaustive_binary_space": True}


def validate_qubo(instance: Instance, reference: float) -> dict:
    """Validate by penalty bound plus partition enumeration or exact L subset DP."""
    qubo = build_qubo(instance)
    if len(qubo.variable_order) <= 16:
        return validate_small_qubo(instance, reference)
    from .classical import partitions, subset_clustering_optima

    bound = objective(instance, instance.certificate) / qubo.normalization
    if not feasible(instance, instance.certificate) or qubo.penalty <= bound:
        raise AssertionError("The feasibility penalty bound is not satisfied")
    checked = 0
    optimum = float("inf")
    winners = []
    canonical_ground_strings = set()
    if instance.tier == "l":
        dp_cost, optimal_partitions = subset_clustering_optima(instance)
        if not np.isclose(dp_cost, reference, atol=1e-8, rtol=0):
            raise AssertionError("Subset DP and constrained reference disagree")
        candidates = optimal_partitions
    else:
        candidates = partitions(instance.n, instance.vehicles)
    for labels in candidates:
        if not feasible(instance, labels):
            continue
        bitstring = encode(instance, labels)
        cost = objective(instance, labels)
        assignment, capacity = residuals(instance, bitstring)
        if any(assignment + capacity) or not np.isclose(
            qubo.energy(list(map(int, bitstring))), cost / qubo.normalization, atol=1e-8, rtol=0
        ):
            raise AssertionError("Encoded feasible partition has an incorrect QUBO energy")
        checked += 1
        optimum = min(optimum, cost)
        if np.isclose(cost, reference, atol=1e-8, rtol=0):
            winners.append(labels)
            canonical_ground_strings.add(bitstring)
    if not winners or not np.isclose(optimum, reference, atol=1e-8, rtol=0):
        raise AssertionError("QUBO validation and constrained reference disagree")
    ground_strings = {
        encode(instance, tuple(relabel[vehicle] for vehicle in labels))
        for labels in winners for relabel in permutations(range(instance.vehicles))
    }
    minimum = reference / qubo.normalization
    for bitstring in ground_strings - canonical_ground_strings:
        if not np.isclose(qubo.energy(list(map(int, bitstring))), minimum, atol=1e-8, rtol=0):
            raise AssertionError("Vehicle-relabelled ground state has an incorrect energy")
    return {
        "checked_bitstrings": checked + len(ground_strings - canonical_ground_strings),
        "checked_feasible_partitions": checked, "ground_states": len(ground_strings),
        "ground_state_bitstrings": sorted(ground_strings),
        "distinct_ground_partitions": len(winners), "passed": True, "minimum_energy": minimum,
        "validation_method": ("penalty-proof-and-subset-DP" if instance.tier == "l"
                              else "penalty-proof-and-feasible-partitions"),
        "exhaustive_binary_space": False,
        "penalty": qubo.penalty, "feasible_certificate_bound": bound,
    }
