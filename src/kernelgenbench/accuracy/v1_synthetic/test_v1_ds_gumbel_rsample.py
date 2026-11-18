"""Accuracy and performance for v1_ds_gumbel_rsample.

Both sides run under the same manual seed, so the fresh device samples must be
identical. The distribution assertions are property checks on the Solution, not
a second RNG stream: a Gumbel(0, 1) draw has mean Euler-Mascheroni (0.5772) and
standard deviation pi/sqrt(6) (1.2825).
"""
import math

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.sampling_test_utils import measure_call
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

SHAPES = [(), (0,), (7,), (3, 13), (2, 3, 5), (4096,)]
COMBOS = [(shape,) for shape in SHAPES]


def input_build(config):
    (shape,) = config
    return {"shape": shape, "device": torch.device("cpu")}


def _on_device(kwargs):
    return {**kwargs, "device": kernelgenbench.device}


def compute_ref(shape, device):
    return kernelgenbench.baseline.v1_ds_gumbel_rsample(shape, device)


def compute_res(shape, device):
    return kernelgenbench.solution.v1_ds_gumbel_rsample(shape, device)


def compare(ref_out, res_out):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.shape == ref_out.shape
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(shape, device):
    return measure_call(compute_ref, {"shape": shape, "device": device}, device)


def compute_res_performance(shape, device):
    return measure_call(compute_res, {"shape": shape, "device": device}, device)


@label("v1_ds_gumbel_rsample")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_gumbel_rsample(config):
    (shape,) = config
    inputs = _on_device(input_build(config))
    torch.manual_seed(20261110)
    ref_out = compute_ref(**inputs)
    torch.manual_seed(20261110)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out)
    assert ref_out.device.type == "cuda"

    if math.prod(shape) >= 4096:
        # Property check on the real Solution: mean/std of a standard Gumbel.
        mean = float(res_out.mean())
        std = float(res_out.std())
        assert abs(mean - 0.5772156649) < 0.06, mean
        assert abs(std - math.pi / math.sqrt(6)) < 0.08, std
        # Successive draws advance the device RNG instead of replaying a cache.
        again = compute_res(**inputs)
        assert not torch.equal(again, res_out)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
