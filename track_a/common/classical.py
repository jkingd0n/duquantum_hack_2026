"""Exact subset references, fixed-group Held-Karp routing, and raw QUBO SA."""

from collections import Counter
from functools import lru_cache
from itertools import permutations
from math import inf
from time import perf_counter

import numpy as np

from .instances import Instance, feasible
from .models import Qubo, build_qubo, canonical, objective

ANNEALING_STEPS = 400
INITIAL_TEMPERATURE_FACTOR = 10.0
FINAL_TEMPERATURE_FACTOR = 0.01


def partitions(n: int, vehicles: int):
    """Restricted-growth strings: each unlabeled partition exactly once."""
    labels = [0] * n

    def visit(i: int, maximum: int):
        if i == n:
            yield tuple(labels)
            return
        for k in range(min(maximum + 1, vehicles - 1) + 1):
            labels[i] = k
            yield from visit(i + 1, max(maximum, k))

    yield from visit(1, 0)


@lru_cache(maxsize=16)
def subset_clustering_optima(instance: Instance) -> tuple[float, tuple[tuple[int, ...], ...]]:
    """Exact compactness DP over capacity-feasible subsets, without routing."""
    n = instance.n
    size = 1 << n
    loads = np.zeros(size, dtype=int)
    costs = np.zeros(size)
    distances = instance.distances
    for mask in range(1, size):
        bit = mask & -mask
        customer = bit.bit_length() - 1
        previous = mask ^ bit
        loads[mask] = loads[previous] + instance.demands[customer]
        costs[mask] = costs[previous] + sum(
            distances[customer, j] for j in range(n) if previous & (1 << j))

    @lru_cache(None)
    def solve(mask: int, vehicles: int) -> tuple[float, tuple[int, ...]]:
        if mask == 0:
            return 0., ()
        if vehicles == 0 or loads[mask] > vehicles * instance.capacity:
            return inf, ()
        if vehicles == 1:
            return (float(costs[mask]), (mask,)) if loads[mask] <= instance.capacity else (inf, ())
        first = mask & -mask
        sub = mask
        best = inf
        choices = []
        while sub:
            if sub & first and loads[sub] <= instance.capacity:
                remaining, _ = solve(mask ^ sub, vehicles - 1)
                value = float(costs[sub]) + remaining
                if value < best - 1e-8:
                    best, choices = value, [sub]
                elif np.isfinite(value) and abs(value - best) <= 1e-8:
                    best = min(best, value)
                    choices.append(sub)
            sub = (sub - 1) & mask
        return best, tuple(choices)

    def reconstruct(mask: int, vehicles: int):
        if not mask:
            yield ()
            return
        for sub in solve(mask, vehicles)[1]:
            for rest in reconstruct(mask ^ sub, vehicles - 1):
                yield (sub, *rest)

    full = size - 1
    best = solve(full, instance.vehicles)[0]
    if not np.isfinite(best):
        raise ValueError("No feasible grouping")
    assignments = []
    for groups in reconstruct(full, instance.vehicles):
        labels = [0] * n
        for vehicle, group in enumerate(groups):
            for customer in range(n):
                if group & (1 << customer):
                    labels[customer] = vehicle
        assignments.append(tuple(labels))
    return best, tuple(assignments)


def clustering_reference(instance: Instance) -> dict:
    """Exact clustering only; routing references belong to the results stage."""
    start = perf_counter()
    if instance.tier == "l":
        best, winners = subset_clustering_optima(instance)
        result = {
            "clustering": {"cost": best, "labels": winners[0], "distinct_optima": len(winners),
                           "status": "proven_optimum"},
            "status": "proven_optimum", "algorithm": "exact-clustering-subset-DP",
            "settings": {"maximum_customers": 15, "vehicle_labels_quotiented": True,
                         "equality_tolerance": 1e-8},
            "seconds": perf_counter() - start,
        }
        result["cplex_crosscheck"] = {
            "clustering": {"status": "not_run",
                           "reason": "L uses exact clustering-only subset DP; no routing columns built."},
        }
        return result
    candidates = [labels for labels in partitions(instance.n, instance.vehicles)
                  if feasible(instance, labels)]
    if not candidates:
        raise ValueError("No feasible grouping")
    costs = [objective(instance, labels) for labels in candidates]
    best = min(costs)
    result = {
        "clustering": {"cost": best, "labels": candidates[costs.index(best)],
                       "distinct_optima": sum(abs(cost - best) <= 1e-8 for cost in costs),
                       "status": "proven_optimum"},
        "status": "proven_optimum", "algorithm": "exhaustive-clustering-partitions",
        "settings": {"maximum_customers": instance.n, "vehicle_labels_quotiented": True,
                     "equality_tolerance": 1e-8},
        "feasible_partitions": len(candidates), "seconds": perf_counter() - start,
    }
    if instance.tier == "s":
        result["cplex_crosscheck"] = {
            "clustering": installed_qubo_crosscheck(instance, build_qubo(instance), best),
        }
    else:
        result["cplex_crosscheck"] = cplex_crosscheck(instance, result)
    return result


class Routing:
    """Exact tours shared across demand permutations with the same geometry."""

    def __init__(self, instance: Instance):
        self.coordinates = instance.coordinates
        self.depot = instance.depot
        self.distances = instance.distances
        self.to_depot = np.linalg.norm(np.asarray(instance.coordinates) - instance.depot, axis=1)
        self.tours: dict[tuple[int, ...], tuple[float, tuple[int, ...]]] = {(): (0.0, ())}

    def tour(self, group: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        group = tuple(sorted(group))
        if group in self.tours:
            return self.tours[group]

        @lru_cache(None)
        def path(mask: int, last: int) -> tuple[float, tuple[int, ...]]:
            previous = mask ^ (1 << last)
            customer = group[last]
            if previous == 0:
                return float(self.to_depot[customer]), (customer,)
            cost, order = min(
                (path(previous, j)[0] + self.distances[group[j], customer],
                 path(previous, j)[1])
                for j in range(len(group)) if previous & (1 << j)
            )
            return float(cost), order + (customer,)

        mask = (1 << len(group)) - 1
        cost, order = min(
            (path(mask, last)[0] + self.to_depot[group[last]], path(mask, last)[1])
            for last in range(len(group))
        )
        result = float(cost), order
        self.tours[group] = result
        return result

    def evaluate(self, instance: Instance, labels: tuple[int, ...]) -> dict:
        if (instance.coordinates != self.coordinates or instance.depot != self.depot
                or not feasible(instance, labels)):
            raise ValueError("Invalid fixed-group routing request")
        tours, cost = [], 0.0
        covered = []
        for k in range(instance.vehicles):
            group = tuple(i for i, v in enumerate(labels) if v == k)
            length, order = self.tour(group)
            covered.extend(order)
            if set(order) != set(group):
                raise AssertionError("Routing changed customer groups")
            tours.append({"vehicle": k, "customers": list(order),
                          "route": ["depot", *order, "depot"], "cost": length,
                          "load": sum(instance.demands[i] for i in group)})
            cost += length
        if sorted(covered) != list(range(instance.n)):
            raise AssertionError("Route coverage failure")
        return {"cost": cost, "tours": tours}


@lru_cache(maxsize=3)
def subset_costs(coordinates: tuple[tuple[float, float], ...],
                 depot: tuple[float, float]):
    """All group clustering costs and exact depot tour costs, independent of demand."""
    n = len(coordinates)
    distances = np.linalg.norm(np.asarray(coordinates)[:, None] - np.asarray(coordinates)[None, :], axis=-1)
    to_depot = np.linalg.norm(np.asarray(coordinates) - depot, axis=1)
    compactness = np.zeros(1 << n)
    paths = np.full((1 << n, n), np.inf)
    tours = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        bit = mask & -mask
        i = bit.bit_length() - 1
        previous = mask ^ bit
        compactness[mask] = compactness[previous] + sum(
            distances[i, j] for j in range(n) if previous & (1 << j))
        members = [j for j in range(n) if mask & (1 << j)]
        for last in members:
            rest = mask ^ (1 << last)
            paths[mask, last] = (to_depot[last] if rest == 0 else min(
                paths[rest, j] + distances[j, last] for j in members if j != last))
        tours[mask] = min(paths[mask, j] + to_depot[j] for j in members)
    return compactness, tours


def exact_references(instance: Instance) -> dict:
    start = perf_counter()
    compactness, tour_cost = subset_costs(instance.coordinates, instance.depot)
    loads = np.zeros(1 << instance.n, dtype=int)
    for mask in range(1, len(loads)):
        bit = mask & -mask
        loads[mask] = loads[mask ^ bit] + instance.demands[bit.bit_length() - 1]

    @lru_cache(None)
    def solve(mask: int, vehicles: int) -> tuple[float, float, int, int, int, int]:
        if mask == 0:
            return 0.0, 0.0, 1, 1, 0, 0
        if vehicles == 0 or loads[mask] > vehicles * instance.capacity:
            return inf, inf, 0, 0, 0, 0
        if vehicles == 1:
            if loads[mask] <= instance.capacity:
                return float(compactness[mask]), float(tour_cost[mask]), 1, 1, mask, mask
            return inf, inf, 0, 0, 0, 0
        first = mask & -mask
        sub = mask
        best_f = best_c = inf
        count_f = count_c = choice_f = choice_c = 0
        while sub:
            if sub & first and loads[sub] <= instance.capacity:
                f, c, nf, nc, _, _ = solve(mask ^ sub, vehicles - 1)
                f += compactness[sub]
                c += tour_cost[sub]
                if nf:
                    if f < best_f - 1e-8:
                        best_f, count_f, choice_f = float(f), nf, sub
                    elif abs(f - best_f) <= 1e-8:
                        count_f += nf
                if nc:
                    if c < best_c - 1e-8:
                        best_c, count_c, choice_c = float(c), nc, sub
                    elif abs(c - best_c) <= 1e-8:
                        count_c += nc
            sub = (sub - 1) & mask
        return best_f, best_c, count_f, count_c, choice_f, choice_c

    full = (1 << instance.n) - 1
    f, c, nf, nc, _, _ = solve(full, instance.vehicles)

    def assignment(which: int) -> tuple[int, ...]:
        labels = [0] * instance.n
        mask, remaining, vehicle = full, instance.vehicles, 0
        while mask:
            sub = solve(mask, remaining)[which]
            if not sub:
                raise AssertionError("Exact reference reconstruction failed")
            for i in range(instance.n):
                if sub & (1 << i):
                    labels[i] = vehicle
            mask ^= sub
            remaining -= 1
            vehicle += 1
        return canonical(labels)

    result = {
        "clustering": {"cost": f, "labels": assignment(4), "distinct_optima": nf,
                       "status": "proven_optimum"},
        "cvrp": {"cost": c, "labels": assignment(5), "optimal_partitions": nc,
                 "status": "proven_optimum"},
        "status": "proven_optimum", "algorithm": "exhaustive-subset-dynamic-programming",
        "settings": {"maximum_customers": 15, "vehicle_labels_quotiented": True,
                     "equality_tolerance": 1e-8},
        "seconds": perf_counter() - start,
    }
    if instance.tier == "s":
        routing = Routing(instance)
        route_optima = 0
        feasible_partitions = 0
        for labels in partitions(instance.n, instance.vehicles):
            if not feasible(instance, labels):
                continue
            feasible_partitions += 1
            if abs(routing.evaluate(instance, labels)["cost"] - c) > 1e-8:
                continue
            multiplicity = 1
            for k in range(instance.vehicles):
                group = tuple(i for i, v in enumerate(labels) if v == k)
                cost, _ = routing.tour(group)
                orders = set()
                for order in permutations(group):
                    length = (0.0 if not order else routing.to_depot[order[0]] +
                              sum(routing.distances[a, b] for a, b in zip(order, order[1:])) +
                              routing.to_depot[order[-1]])
                    if abs(length - cost) <= 1e-8:
                        orders.add(min(order, order[::-1]))
                multiplicity *= len(orders)
            route_optima += multiplicity
        result["cvrp"]["distinct_optima_modulo_reversal"] = route_optima
        result["feasible_partitions"] = feasible_partitions
    result["cplex_crosscheck"] = cplex_crosscheck(instance, result)
    return result


def cplex_crosscheck(instance: Instance, reference: dict, time_limit: float = 30.0) -> dict:
    """Independent set-partition MILPs on S, with exact group-tour columns."""
    from docplex.mp.model import Model
    from docplex.mp.utils import DOcplexException

    start = perf_counter()
    compactness, tours = subset_costs(instance.coordinates, instance.depot)
    groups = [mask for mask in range(1, 1 << instance.n) if sum(
        instance.demands[i] for i in range(instance.n) if mask & (1 << i)
    ) <= instance.capacity]
    outputs = {}
    for name, costs in (("clustering", compactness), ("cvrp", tours)):
        if name not in reference:
            continue
        try:
            with Model(name=f"track_a_{name}") as model:
                variables = model.binary_var_list(len(groups))
                for i in range(instance.n):
                    model.add_constraint(model.sum(variables[j] for j, mask in enumerate(groups)
                                                   if mask & (1 << i)) == 1)
                model.add_constraint(model.sum(variables) <= instance.vehicles)
                model.minimize(model.sum(float(costs[mask]) * variables[j]
                                         for j, mask in enumerate(groups)))
                model.parameters.timelimit = time_limit
                model.parameters.threads = 1
                model.parameters.randomseed = 0
                solution = model.solve(log_output=False)
                if solution is None:
                    outputs[name] = {"status": model.solve_details.status, "cost": None}
                else:
                    value = solution.objective_value
                    if abs(value - reference[name]["cost"]) > 1e-6:
                        raise AssertionError("CPLEX and exhaustive reference disagree")
                    outputs[name] = {"status": model.solve_details.status, "cost": value,
                                     "agrees": True}
        except DOcplexException as error:
            outputs[name] = {"status": "solver_error", "error": str(error)}
    return {"settings": {"time_limit_seconds": time_limit, "threads": 1, "seed": 0},
            "seconds": perf_counter() - start,
            **outputs}


def installed_qubo_crosscheck(instance: Instance, qubo: Qubo, reference: float) -> dict:
    """Exercise the installed qubo-solver API without preprocessing or repair."""
    from qubosolver import ClassicalSolvingConfig, Instance as SolverInstance
    from qubosolver import Solver, SolverConfig, matrix
    from .models import decode

    solution = Solver(
        SolverInstance(matrix.tensor(qubo.matrix)),
        SolverConfig(solving=ClassicalSolvingConfig(
            algorithm="cplex", time_limit=30, max_bitstrings=1,
        ), preprocessing=False, postprocessing=False),
    ).solve()
    bits = "".join(map(str, solution.bitstrings[0].tolist()))
    labels = decode(instance, bits)
    if labels is None or not np.isclose(objective(instance, labels), reference):
        raise AssertionError("Installed qubo-solver disagrees with the constrained reference")
    return {"agrees": True, "bitstring": bits, "energy": qubo.energy(list(map(int, bits))),
            "preprocessing": False, "postprocessing": False, "algorithm": "cplex",
            "time_limit_seconds": 30}


def anneal(qubo: Qubo, shots: int, seed: int, steps: int = ANNEALING_STEPS) -> Counter[str]:
    """Independent Metropolis chains, no repair or certificate warm start."""
    rng = np.random.default_rng(seed)
    size = len(qubo.variable_order)
    z = rng.integers(0, 2, size=(shots, size))
    diagonal = np.diag(qubo.matrix)
    pairs = 2 * (qubo.matrix - np.diag(diagonal))
    fields = z @ pairs + diagonal
    temperatures = np.geomspace(INITIAL_TEMPERATURE_FACTOR * qubo.penalty,
                                FINAL_TEMPERATURE_FACTOR * qubo.penalty, steps)
    rows = np.arange(shots)
    for temperature in temperatures:
        columns = rng.integers(size, size=shots)
        change = 1 - 2 * z[rows, columns]
        delta = change * fields[rows, columns]
        accepted = (delta <= 0) | (np.log(rng.random(shots)) < -delta / temperature)
        selected_rows, selected_cols = rows[accepted], columns[accepted]
        fields[selected_rows] += change[accepted, None] * pairs[selected_cols]
        z[selected_rows, selected_cols] = 1 - z[selected_rows, selected_cols]
    return Counter("".join(map(str, row)) for row in z)
