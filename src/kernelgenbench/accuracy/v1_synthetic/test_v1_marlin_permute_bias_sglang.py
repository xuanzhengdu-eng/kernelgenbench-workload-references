"""Check the pinned v1_marlin_permute_bias_sglang layout, input preservation, allocation and timing."""
import math
import torch
import pytest
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from kernelgenbench.accuracy.v1_synthetic.rows352_361_utils import measure


COMBOS = [(shape, group_size, dtype, strided, device)
          for shape in [(64,), (2, 2, 32)]
          for group_size in (None,)
          for dtype in (torch.float16, torch.bfloat16, torch.float32)
          for strided in (False, True)
          for device in ("cpu", "cuda")]


def input_build(config):
    shape, group_size, dtype, strided, device = config
    factor = 2 if strided else 1
    values = torch.arange(math.prod(shape) * factor, dtype=torch.float32, device=device)
    s = values.to(dtype).reshape(*shape[:-1], shape[-1] * factor)
    if strided:
        s = s[..., ::2]
    return {"s": s}


def compute_ref(s):
    return kernelgenbench.baseline.v1_marlin_permute_bias_sglang(s)


def compute_res(s):
    return kernelgenbench.solution.v1_marlin_permute_bias_sglang(s)


@label("v1_marlin_permute_bias_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_marlin_permute_bias_sglang(config):
    reference_inputs = input_build(config)
    solution_inputs = input_build(config)
    before = solution_inputs["s"].clone()
    expected = compute_ref(**reference_inputs)
    actual = compute_res(**solution_inputs)
    assert torch.equal(expected, actual)
    assert actual.is_contiguous() and actual.shape == before.shape
    assert actual.dtype == before.dtype and actual.device == before.device
    assert not torch._C._is_alias_of(actual, solution_inputs["s"])
    assert torch.equal(solution_inputs["s"], before)
    reference_ms = measure(compute_ref, reference_inputs)
    solution_ms = measure(compute_res, solution_inputs)
    return CustomBenchmarkResult(ref_time=reference_ms, res_time=solution_ms, speedup=reference_ms / solution_ms)
