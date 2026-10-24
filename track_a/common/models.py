"""A symmetric x.T Q x QUBO with an explicit constant offset."""

from dataclasses import dataclass
from math import ceil, log2

import numpy as np
from numpy.typing import NDArray

from .instances import Instance, feasible

ENCODING = "binary-slack-v1"


def canonical(labels: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    return tuple(mapping.setdefault(k, len(mapping)) for k in labels)


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

    def energy(self, bits: NDArray | list[int]) -> float:
        z = np.asarray(bits)
        return float(z @ self.matrix @ z + self.constant)


def build_qubo(instance: Instance, penalty: float | None = None) -> Qubo:
    n, k, q = instance.n, instance.vehicles, instance.capacity
    b = ceil(log2(q + 1))
    size = n * k + k * b
    normalization = max(float(instance.distances.max()), 1.0)
    upper = objective(instance, instance.certificate) / normalization
    weight = upper + 1 if penalty is None else penalty
    if not np.isfinite(weight) or weight <= upper:
        raise ValueError("Penalty must exceed feasible-certificate normalized objective")
    matrix = np.zeros((size, size))
    for i in range(n):
        for j in range(i):
            for vehicle in range(k):
                a, c = i * k + vehicle, j * k + vehicle
                matrix[a, c] += instance.distances[i, j] / normalization / 2
                matrix[c, a] = matrix[a, c]
    constant = 0.0

    def square(coefficients: NDArray[np.float64], rhs: int) -> None:
        nonlocal constant
        matrix[:] += weight * np.outer(coefficients, coefficients)
        matrix[np.diag_indices(size)] -= 2 * weight * rhs * coefficients
        constant += weight * rhs**2

    for i in range(n):
        coefficients = np.zeros(size)
        coefficients[i * k:(i + 1) * k] = 1
        square(coefficients, 1)
    for vehicle in range(k):
        coefficients = np.zeros(size)
        coefficients[vehicle:n * k:k] = instance.demands
        coefficients[n * k + vehicle * b:n * k + (vehicle + 1) * b] = 2.0 ** np.arange(b)
        square(coefficients, q)
    names = [f"x_{i}_{v}" for i in range(n) for v in range(k)]
    names += [f"y_{v}_{bit}" for v in range(k) for bit in range(b)]
    return Qubo(matrix, constant, weight, normalization, b, names)


def decode(instance: Instance, bitstring: str) -> tuple[int, ...] | None:
    b = ceil(log2(instance.capacity + 1))
    if len(bitstring) != instance.n * instance.vehicles + instance.vehicles * b:
        raise ValueError("Bitstring length does not match encoding")
    if set(bitstring) - {"0", "1"}:
        raise ValueError("Nonbinary candidate")
    x = np.array([int(c) for c in bitstring[:instance.n * instance.vehicles]])
    x = x.reshape(instance.n, instance.vehicles)
    if not np.all(x.sum(axis=1) == 1):
        return None
    labels = tuple(int(v) for v in x.argmax(axis=1))
    return canonical(labels) if feasible(instance, labels) else None


def encode(instance: Instance, labels: tuple[int, ...]) -> str:
    if not feasible(instance, labels):
        raise ValueError("Cannot encode infeasible assignment")
    b = ceil(log2(instance.capacity + 1))
    bits = [int(v == labels[i]) for i in range(instance.n) for v in range(instance.vehicles)]
    for v in range(instance.vehicles):
        slack = instance.capacity - sum(d for i, d in enumerate(instance.demands) if labels[i] == v)
        bits.extend((slack >> bit) & 1 for bit in range(b))
    return "".join(map(str, bits))


def residuals(instance: Instance, bitstring: str) -> tuple[list[int], list[int]]:
    decode(instance, bitstring)
    k, n = instance.vehicles, instance.n
    bits = np.array(list(map(int, bitstring)))
    x = bits[:n * k].reshape(n, k)
    b = (len(bits) - n * k) // k
    slack = bits[n * k:].reshape(k, b) @ (2 ** np.arange(b))
    return (x.sum(axis=1) - 1).tolist(), (np.array(instance.demands) @ x + slack - instance.capacity).tolist()


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
            "minimum_energy": float(energies.min())}
