# Synthetic workload; no production capture or production frequency claim.
"""OP-dca6e8846283, official sub-contract: Megatron get_updated_expert_bias after a one-rank identity all-reduce (NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84, megatron/core/transformer/moe/moe_utils.py:1136-1159). Float32 counts/bias, returns a fresh no-grad tensor; multi-rank aggregation is not represented."""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.synthetic_inputs import build_case
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# tokens_per_expert=expert_bias=(16,) float32; update rate 1e-3; one rank.
# (shape, dtype) only selects the synthetic token count.
COMBOS = [(shape, dtype)
          for shape in [(2, 16, 128), (2, 128, 1024)]
          for dtype in (torch.float32,)]


def input_build(config):
    shape, dtype = config
    return build_case('expert_bias_update', shape, dtype, 'cpu')


def to_device(inputs):
    return {k: (v.to(kernelgenbench.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


def compute_ref(tokens_per_expert, expert_bias, expert_bias_update_rate):
    return kernelgenbench.baseline.v1_expert_bias_update_single_rank(
        tokens_per_expert, expert_bias, expert_bias_update_rate)


def compute_res(tokens_per_expert, expert_bias, expert_bias_update_rate):
    return kernelgenbench.solution.v1_expert_bias_update_single_rank(
        tokens_per_expert, expert_bias, expert_bias_update_rate)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    # updated = expert_bias + sign(average_tokens - tokens_per_expert) * rate
    torch.testing.assert_close(res_out, ref_out, rtol=1e-5, atol=1e-6)


def measure(fn, inputs):
    import triton.testing
    return triton.testing.do_bench(lambda: fn(**inputs), warmup=25, rep=100)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label('v1_expert_bias_update_single_rank')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_expert_bias_update_single_rank(config):
    dtype = config[1]
    inputs = to_device(input_build(config))
    inputs['expert_bias'].requires_grad_(True)
    ref_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    res_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    expected = compute_ref(**ref_inputs)
    actual = compute_res(**res_inputs)
    compare(expected, actual, dtype)
    # The official update runs inside torch.no_grad() and returns a new tensor.
    assert not actual.requires_grad, 'bias update must run without autograd'
    for value in res_inputs.values():
        if isinstance(value, torch.Tensor):
            assert (actual.untyped_storage().data_ptr()
                    != value.untyped_storage().data_ptr()), 'output must not alias an input'
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, res_inputs[key]), f'candidate modified input: {key}'
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float('inf'))
