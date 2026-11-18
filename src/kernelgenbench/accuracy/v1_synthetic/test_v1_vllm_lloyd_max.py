"""Accuracy and performance for v1_vllm_lloyd_max.

Host-side (CPU) Lloyd-Max quantizer initialization for N(0, 1/d) in float64.
Input: vector dimension, bit width (levels = 2**bits), iteration bound and
convergence tolerance. Output: (centroids [2**bits], boundaries [2**bits - 1])
as float32 CPU tensors.

Key assertions: tuple length, exact shapes (including the empty boundary tensor
for bits=0), float32 CPU dtype, strict monotonic ordering, symmetry of the
Gaussian quantizer, the documented "break before assigning" semantics for
max_iter=0 / tol=1e9, and numerical agreement with the official solver.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_lloyd_max"

# (vector dim, bits, max iterations, convergence tolerance)
COMBOS = [
    (64, 1, 0, 1e-10),
    (64, 2, 3, 1e-10),
    (128, 3, 8, 1e-10),
    (32, 1, 25, 1e-10),
    (64, 2, 3, 1e9),
    (256, 0, 2, 1e-10),
    (64, 4, 200, 1e-10),
]


def input_build(config):
    d, bits, max_iter, tol = config
    return dict(d=d, bits=bits, max_iter=max_iter, tol=tol)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_lloyd_max(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_lloyd_max(**inputs)


def _initial_centroids(d, bits):
    import math
    levels = 2 ** bits
    sigma = math.sqrt(1.0 / d)
    lo, hi = -3.5 * sigma, 3.5 * sigma
    return torch.tensor([lo + (hi - lo) * (i + 0.5) / levels for i in range(levels)],
                        dtype=torch.float32)


def compare(ref_out, res_out, config):
    d, bits, max_iter, tol = config
    levels = 2 ** bits
    for out in (ref_out, res_out):
        assert isinstance(out, tuple) and len(out) == 2
    for expected, actual in zip(ref_out, res_out):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype == torch.float32
        assert actual.device.type == expected.device.type == "cpu"
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    centroids, boundaries = ref_out
    assert centroids.shape == (levels,)
    assert boundaries.shape == (max(levels - 1, 0),)
    if levels > 1:
        assert torch.all(centroids[1:] > centroids[:-1]), "centroids must be sorted"
        assert torch.all(boundaries[1:] > boundaries[:-1]) if levels > 2 else True
        assert torch.allclose(centroids, -torch.flip(centroids, [0]), atol=1e-6), \
            "Gaussian quantizer centroids must be symmetric"
        torch.testing.assert_close(boundaries, (centroids[:-1] + centroids[1:]) / 2,
                                   rtol=1e-6, atol=1e-7)
    if max_iter == 0 or tol >= 1e9:
        torch.testing.assert_close(centroids, _initial_centroids(d, bits), rtol=0, atol=1e-7)
        torch.testing.assert_close(boundaries, (centroids[:-1] + centroids[1:]) / 2,
                                   rtol=0, atol=0) if levels > 1 else None


def _bench(fn, config):
    fn(**config)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        fn(**config)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def compute_ref_performance(config):
    return _bench(compute_ref, config)


def compute_res_performance(config):
    return _bench(compute_res, config)


@label("v1_vllm_lloyd_max")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_lloyd_max(config):
    inputs = input_build(config)
    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out, config)
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
