# Synthetic workload; no production capture or production frequency claim.
"""OP-81914edec43a, official sub-contract: Megatron get_tokens_per_expert_and_token_count with a one-rank identity reduction (NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84, megatron/core/transformer/moe/moe_utils.py:227-246). With a padding mask both token counts are zero-dimensional float tensors; without one they are Python ints. Multi-rank aggregation is not represented."""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.synthetic_inputs import build_case
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# routing_map=(B*T,16) bool, topk=4, optional padding rows (every 5th row is
# zeroed), one rank. (shape, dtype, with_padding_mask)
COMBOS = [(shape, dtype, padding)
          for shape in [(2, 16, 128), (2, 128, 1024)]
          for dtype in (torch.float32,)
          for padding in (False, True)]


def input_build(config):
    shape, dtype, with_padding_mask = config
    inputs = build_case('expert_token_counts', shape, dtype, 'cpu')
    inputs['with_padding_mask'] = with_padding_mask
    if with_padding_mask:
        inputs['routing_map'][::5] = False
    return inputs


def to_device(inputs):
    return {k: (v.to(kernelgenbench.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


def compute_ref(routing_map, topk, with_padding_mask):
    return kernelgenbench.baseline.v1_expert_token_counts_single_rank(
        routing_map, topk, with_padding_mask)


def compute_res(routing_map, topk, with_padding_mask):
    return kernelgenbench.solution.v1_expert_token_counts_single_rank(
        routing_map, topk, with_padding_mask)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    counts_ref, local_ref, total_ref = ref_out
    counts_res, local_res, total_res = res_out
    # Per-expert counts are bool sums, i.e. int64 on both sides.
    assert counts_res.shape == counts_ref.shape
    assert counts_res.dtype == counts_ref.dtype == torch.int64
    assert torch.equal(counts_res, counts_ref), 'per-expert counts differ'
    if isinstance(total_ref, torch.Tensor):
        # Padding branch: both token totals are zero-dimensional float tensors.
        for got, want, name in ((local_res, local_ref, 'local_num_tokens'),
                                (total_res, total_ref, 'total_num_tokens')):
            assert isinstance(got, torch.Tensor), f'{name} must stay a tensor'
            assert got.shape == want.shape and got.dtype == want.dtype
            assert got.device == want.device
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    else:
        # No padding: both totals are Python ints, not tensors.
        assert not isinstance(local_res, torch.Tensor)
        assert not isinstance(total_res, torch.Tensor)
        assert (local_res, total_res) == (local_ref, total_ref)


def measure(fn, inputs):
    import triton.testing
    return triton.testing.do_bench(lambda: fn(**inputs), warmup=25, rep=100)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label('v1_expert_token_counts_single_rank')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_expert_token_counts_single_rank(config):
    dtype = config[1]
    inputs = to_device(input_build(config))
    ref_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    res_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    expected = compute_ref(**ref_inputs)
    actual = compute_res(**res_inputs)
    compare(expected, actual, dtype)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, res_inputs[key]), f'candidate modified input: {key}'
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float('inf'))
