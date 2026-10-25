"""Photographed normalized-capacity QUBO with one complemented slack bit per vehicle."""

from pathlib import Path

from .classical import ANNEALING_STEPS
from .experiments import _validate_run
from .instances import Instance
from .models import Qubo, build_variable_reduction_qubo, variable_reduction_penalties
from .quantum import (
    EMBEDDING_SWAPS, EMULATOR_DT_NS, INTERACTION_CUTOFF, MPS_MAX_BOND_DIM, MPS_PRECISION,
)
from .unbalanced import _classical_formulation_run, _quantum_formulation_runs, _reference


def _configuration(target: Qubo, alpha: float, beta: float) -> dict:
    return {
        "alpha": alpha, "beta": beta, "capacity_penalty": alpha, "assignment_penalty": beta,
        "normalization": target.normalization, "logical_capacity_residual": "load/Q - 1 - b",
        "auxiliary_mapping": "b_k = 1 - z_k", "binary_slack_bits_per_vehicle": 1,
        "ground_state_feasibility_guaranteed": False, "warm_start": False, "repair": False,
    }


def classical_variable_reduction_runs(instances: list[Instance], output: Path, shots: int, seed: int, *,
                                      alpha: float | None = None, beta: float | None = None,
                                      annealing_steps: int = ANNEALING_STEPS) -> None:
    _validate_run(instances, output, shots)
    if type(annealing_steps) is not int or annealing_steps <= 0:
        raise ValueError("Annealing steps must be a positive integer")
    for instance in instances:
        _reference(instance, output)
        first, second = variable_reduction_penalties(instance, alpha, beta)
        target = build_variable_reduction_qubo(instance, first, second)
        _classical_formulation_run(instance, target, _configuration(target, first, second),
                                   output, shots, seed, "classical-variable_reduction-sa", annealing_steps)


def quantum_variable_reduction_runs(instances: list[Instance], output: Path, shots: int, seed: int,
                                    timeout: float, memory_mb: int, *,
                                    alpha: float | None = None, beta: float | None = None,
                                    embedding_swaps: int = EMBEDDING_SWAPS,
                                    dt_ns: int = EMULATOR_DT_NS,
                                    interaction_cutoff: float = INTERACTION_CUTOFF,
                                    mps_max_bond_dim: int = MPS_MAX_BOND_DIM,
                                    mps_precision: float = MPS_PRECISION) -> None:
    _validate_run(instances, output, shots)
    targets, configurations = [], []
    for instance in instances:
        first, second = variable_reduction_penalties(instance, alpha, beta)
        target = build_variable_reduction_qubo(instance, first, second)
        targets.append(target)
        configurations.append(_configuration(target, first, second))
    _quantum_formulation_runs(
        instances, output, shots, seed, timeout, memory_mb, targets=targets,
        model_configurations=configurations, formulation="variable_reduction",
        embedding_swaps=embedding_swaps, dt_ns=dt_ns, interaction_cutoff=interaction_cutoff,
        mps_max_bond_dim=mps_max_bond_dim, mps_precision=mps_precision,
    )
