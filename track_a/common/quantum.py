"""Explicit global-drive baseline. No QUBO-solver repair or cloud execution."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from pulser import AnalogDevice, Pulse, Register, Sequence
from pulser.waveforms import InterpolatedWaveform, RampWaveform

from .models import Qubo


@dataclass(frozen=True)
class PulseSettings:
    duration_ns: int = 1000
    amplitude: float = 8.0
    initial_detuning: float = -8.0


def device_metadata() -> dict:
    channel = AnalogDevice.channels["rydberg_global"]
    actual = {
        "max_atoms": AnalogDevice.max_atom_num,
        "min_separation_um": AnalogDevice.min_atom_distance,
        "max_radius_um": AnalogDevice.max_radial_distance,
        "max_amplitude": channel.max_amp,
        "max_abs_detuning": channel.max_abs_detuning,
        "max_duration_ns": AnalogDevice.max_sequence_duration,
        "interaction_coeff": AnalogDevice.interaction_coeff,
    }
    guide = [80, 5, 38, 12.57, 125.66, 6000, 8.657e5]
    if not all(np.isclose(a, b, rtol=5e-4) for a, b in zip(actual.values(), guide, strict=True)):
        raise RuntimeError(f"Installed AnalogDevice differs from challenge guide: {actual}")
    return {**actual, "guide_discrepancy_beyond_rounding": False,
            "abstract_device": json.loads(AnalogDevice.to_abstract_repr())}


def embedding(qubo: Qubo, seed: int, swaps: int = 1200) -> dict:
    """Fit atom permutation on a legal triangular grid, then a positive scale."""
    start = perf_counter()
    device_metadata()
    size = len(qubo.variable_order)
    if size > AnalogDevice.max_atom_num:
        raise ValueError("QUBO exceeds AnalogDevice atom count")
    pairs = 2 * (qubo.matrix - np.diag(np.diag(qubo.matrix)))
    if np.any(pairs < 0):
        raise ValueError("Negative pair couplings are unsupported by repulsive Rydberg interactions")
    candidates = np.array([(i + j / 2, np.sqrt(3) * j / 2)
                           for i in range(-size, size + 1)
                           for j in range(-size, size + 1)])
    candidates = candidates[np.argsort(np.linalg.norm(candidates, axis=1), kind="stable")[:size]]
    coordinates = (candidates - candidates.mean(axis=0)) * 8.0
    radius = np.linalg.norm(coordinates, axis=1).max()
    if radius > 37.9:
        coordinates *= 37.9 / radius
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    interactions = AnalogDevice.interaction_coeff / distances**6
    rng = np.random.default_rng(seed)
    order = rng.permutation(size)
    norm = np.linalg.norm(pairs)
    if norm == 0:
        raise ValueError("Cannot fit a zero interaction target")

    def score(permutation: np.ndarray) -> float:
        realized = interactions[np.ix_(permutation, permutation)]
        return float(np.sum(realized * pairs))

    best = score(order)
    for _ in range(swaps):
        a, b = rng.choice(size, 2, replace=False)
        order[a], order[b] = order[b], order[a]
        value = score(order)
        if value > best:
            best = value
        else:
            order[a], order[b] = order[b], order[a]
    coordinates = coordinates[order]
    realized = interactions[np.ix_(order, order)]
    scale = float(np.sum(realized * pairs) / norm**2)
    linear = np.diag(qubo.matrix)
    common = float(scale * linear.mean())
    channel = AnalogDevice.channels["rydberg_global"]
    if scale <= 0 or abs(common) > channel.max_abs_detuning:
        raise ValueError("Fitted scale or final global detuning is illegal")
    register = Register(dict(zip(qubo.variable_order, coordinates, strict=True)))
    AnalogDevice.validate_register(register)
    return {
        "variable_order": qubo.variable_order, "coordinates_um": coordinates.tolist(),
        "target_pairs": pairs.tolist(), "realized_pairs": realized.tolist(),
        "target_linear": linear.tolist(), "realized_linear": [common] * size,
        "scale": scale, "constant_offset_scaled": scale * qubo.constant,
        "final_detuning": -common,
        "pair_relative_error": float(np.linalg.norm(realized - scale * pairs) / (scale * norm)),
        "linear_relative_error": float(np.linalg.norm(common - scale * linear) /
                                       np.linalg.norm(scale * linear)),
        "unsupported_coupling_signs": False, "nonquadratic_terms": False,
        "unequal_linear_terms": not np.allclose(linear, linear[0]),
        "seed": seed, "settings": {"algorithm": "triangular-grid-greedy-swaps",
                                   "swap_attempts": swaps, "initial_spacing_um": 8.0},
        "legal": True, "seconds": perf_counter() - start,
    }


def sequence_for(layout: dict, settings: PulseSettings) -> Sequence:
    channel = AnalogDevice.channels["rydberg_global"]
    if (settings.duration_ns % channel.clock_period
            or not channel.min_duration <= settings.duration_ns <= AnalogDevice.max_sequence_duration
            or not 0 < settings.amplitude <= channel.max_amp
            or not -channel.max_abs_detuning <= settings.initial_detuning < 0):
        raise ValueError("Illegal pulse settings")
    register = Register(dict(zip(layout["variable_order"], layout["coordinates_um"], strict=True)))
    AnalogDevice.validate_register(register)
    sequence = Sequence(register, AnalogDevice)
    sequence.declare_channel("drive", "rydberg_global")
    amplitude = InterpolatedWaveform(settings.duration_ns, [0, settings.amplitude, 0])
    detuning = RampWaveform(settings.duration_ns, settings.initial_detuning, layout["final_detuning"])
    sequence.add(Pulse(amplitude, detuning, 0), "drive")
    sequence.measure("ground-rydberg")
    if sequence.get_duration(include_fall_time=True) > AnalogDevice.max_sequence_duration:
        raise ValueError("Pulse including modulation fall time exceeds device duration")
    return sequence


def emulate(layout: dict, settings: PulseSettings, shots: int, seed: int,
            timeout_seconds: float = 180, memory_mb: int = 4096) -> dict:
    result = emulate_sequence(sequence_for(layout, settings), shots, seed,
                              timeout_seconds=timeout_seconds, memory_mb=memory_mb)
    result["settings"] = {**asdict(settings), **result["settings"]}
    return result


def emulate_sequence(sequence: Sequence, shots: int, seed: int,
                     timeout_seconds: float = 180, memory_mb: int = 4096) -> dict:
    """Isolate emulators to bound wall time and address-space use."""
    job = {
        "sequence": sequence.to_abstract_repr(), "backend": "sv",
        "shots": shots, "seed": seed,
        "memory_mb": memory_mb, "dt_ns": 10,
    }
    start = perf_counter()
    with TemporaryDirectory(prefix="track-a-emulation-") as directory:
        input_path, output_path = Path(directory) / "job.json", Path(directory) / "result.json"
        input_path.write_text(json.dumps(job))
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "track_a.common.quantum", str(input_path), str(output_path)],
                capture_output=True, text=True, timeout=timeout_seconds, env=env,
                cwd=Path(__file__).resolve().parents[2],
            )
        except subprocess.TimeoutExpired as error:
            return {"status": "resource_limited", "error": str(error),
                    "seconds": perf_counter() - start, "settings": job | {"sequence": None}}
        if completed.returncode:
            return {"status": "emulator_error", "error": completed.stderr[-6000:],
                    "stdout": completed.stdout[-2000:], "returncode": completed.returncode,
                    "seconds": perf_counter() - start, "settings": job | {"sequence": None}}
        result = json.loads(output_path.read_text())
    return {**result, "seconds": perf_counter() - start,
            "settings": {**job, "sequence": None,
                         "interaction_cutoff": 0.0, "with_modulation": False,
                         "timeout_seconds": timeout_seconds, "noise": "noiseless"}}


def _worker(input_path: Path, output_path: Path) -> None:
    import resource
    import torch
    from pulser.backend import BitStrings
    from emu_sv import SVBackend, SVConfig

    job = json.loads(input_path.read_text())
    limit = job["memory_mb"] * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    torch.set_num_threads(1)
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    sequence = Sequence.from_abstract_repr(job["sequence"])
    options = {"dt": job["dt_ns"], "gpu": False, "interaction_cutoff": 0.0,
               "log_level": 40, "observables": [BitStrings(num_shots=job["shots"])],
               "with_modulation": False}
    results = SVBackend(sequence, config=SVConfig(**options)).run()
    if isinstance(results, list):
        raise RuntimeError("Expected a single noiseless emulator result")
    counts = Counter(results.final_bitstrings)
    if sum(counts.values()) != job["shots"]:
        raise AssertionError("Emulator shot count mismatch")
    output_path.write_text(json.dumps({"status": "completed", "counts": dict(counts)}))


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
