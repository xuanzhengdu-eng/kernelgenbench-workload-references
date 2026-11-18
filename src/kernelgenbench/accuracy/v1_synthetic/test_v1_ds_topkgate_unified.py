"""Artificial dense DeepSpeed TopKGate workloads for k=1, k=2 and k>2.

Official source: deepspeedai/DeepSpeed@8af59b2e7a91c227081617240e98aa282eadc5f9,
deepspeed/moe/sharded_moe.py:216-596. Only the dense default path is covered:
no Jitter/RSample noise, no random token selection (use_rts=False),
drop_tokens=True, top2_2nd_expert_sampling=False. k=1 keeps raw gate
probabilities and can restrict routing with used_token; k=2 prioritizes the
top-1 expert for positional capacity; k>2 drops by score capacity.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# tokens, hidden, experts, k, capacity factor, min capacity, dtype, strided x,
# seed, tie (zero x). k=2 and k>2 cases.
CASES = [
    (7, 5, 5, 2, .55, 1, torch.float32, False, 400, False),
    (8, 6, 7, 2, 1.1, 1, torch.float32, True, 401, False),
    (5, 4, 4, 2, .25, 2, torch.float16, True, 402, False),
    (9, 8, 6, 2, .38, 1, torch.bfloat16, False, 403, False),
    (4, 3, 6, 2, .7, 8, torch.float32, False, 404, False),
    (11, 7, 8, 2, .43, 2, torch.float16, False, 405, False),
    (3, 4, 2, 2, .6, 1, torch.bfloat16, True, 406, False),
    (1, 5, 5, 2, .7, 1, torch.float32, True, 407, False),
    (4, 5, 4, 2, .50000001, 1, torch.float32, False, 408, False),
    (6, 4, 5, 2, .5, 1, torch.float32, False, 409, True),
    (7, 5, 5, 3, .55, 1, torch.float32, False, 200, False),
    (8, 6, 7, 4, 1.1, 1, torch.float32, True, 201, False),
    (5, 4, 4, 3, .25, 2, torch.float16, True, 202, False),
    (9, 8, 6, 5, .38, 1, torch.bfloat16, False, 203, False),
    (4, 3, 6, 3, .7, 8, torch.float32, False, 204, False),
    (11, 7, 8, 4, .43, 2, torch.float16, False, 205, False),
    (3, 4, 4, 4, .6, 1, torch.bfloat16, True, 206, False),
    (1, 5, 5, 3, .7, 1, torch.float32, True, 207, False),
    (2, 5, 4, 4, .50000001, 1, torch.float32, False, 208, False),
]
# k=1 cases include tied overflow; the seed parity toggles used_token.
CASES += [
    (7, 5, 5, 1, 5., 1, torch.float32, False, 500, False),
    (8, 6, 7, 1, 7., 1, torch.float32, True, 501, False),
    (5, 4, 4, 1, 4., 2, torch.float16, True, 502, False),
    (9, 8, 6, 1, 6., 1, torch.bfloat16, False, 503, False),
    (4, 3, 6, 1, 7., 1, torch.float32, False, 504, False),
    (11, 7, 8, 1, 8., 2, torch.float16, False, 505, False),
    (1, 5, 5, 1, 5., 1, torch.bfloat16, True, 506, False),
    (6, 4, 5, 1, 5., 1, torch.float32, False, 507, True),
    (8, 5, 4, 1, .5, 1, torch.float32, False, 509, True),
    (8, 5, 4, 1, 1., 1, torch.bfloat16, True, 510, True),
]
COMBOS = list(CASES)


def input_build(config):
    """Build x=[S,H] and weight=[E,H] on CPU; a padded case is twice as wide
    and is narrowed to a strided view after the move to the device."""
    (tokens, hidden, experts, k, factor, min_capacity, dtype, pad, seed,
     tie) = config
    generator = torch.Generator(device='cpu').manual_seed(seed)
    width = hidden * 2 if pad else hidden
    x = torch.randn((tokens, width), dtype=dtype, generator=generator)
    if tie:
        x.zero_()
    weight = torch.randn((experts, hidden), dtype=dtype, generator=generator)
    return dict(x=x, weight=weight, k=k, capacity_factor=factor,
                min_capacity=min_capacity, used_token=None, pad=pad,
                seed=seed)


def to_device(inputs):
    """Move to the accelerator, then create the strided view on the device so
    the case really exercises non-contiguous input (moving a CPU view would
    silently make it contiguous)."""
    inputs = {k: (v.to(kernelgenbench.device) if isinstance(v, torch.Tensor) else v)
              for k, v in inputs.items()}
    if inputs.pop('pad'):
        inputs['x'] = inputs['x'][..., ::2]
    seed = inputs.pop('seed')
    if inputs['k'] == 1 and seed % 2:
        tokens = inputs['x'].shape[0]
        inputs['used_token'] = (
            torch.arange(tokens, device=kernelgenbench.device, dtype=torch.int64) % 3 != 1
        ).to(torch.int64)
    return inputs


def compute_ref(x, weight, k, capacity_factor, min_capacity, used_token=None):
    return kernelgenbench.baseline.v1_ds_topkgate_unified(
        x, weight, k, capacity_factor, min_capacity, used_token)


def compute_res(x, weight, k, capacity_factor, min_capacity, used_token=None):
    return kernelgenbench.solution.v1_ds_topkgate_unified(
        x, weight, k, capacity_factor, min_capacity, used_token)


def compare(ref_out, res_out, case):
    assert isinstance(ref_out, (tuple, list)) and len(ref_out) == 4
    assert isinstance(res_out, (tuple, list)) and len(res_out) == 4
    for got, want in zip(ref_out, res_out):
        assert got.shape == want.shape
        assert got.dtype == want.dtype
        assert got.device == want.device
    k = case[3]
    if k != 1:
        for index, (want, got) in enumerate(zip(ref_out, res_out)):
            if index < 2:  # aux loss and combine weights
                torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
            else:  # dispatch mask and pre-drop counts are exact
                assert torch.equal(got, want)
        return
    # k=1: the official topk over an equal-valued 0/1 mask may keep any of the
    # tied tokens, so validate capacity/order/weight rules instead.
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=2e-5, atol=2e-5)
    assert torch.equal(res_out[3], ref_out[3])
    combine = res_out[1]
    assert res_out[2].dtype == torch.bool and torch.equal(res_out[2], combine.bool())
    data = to_device(input_build(case))
    p = torch.softmax(torch.nn.functional.linear(
        data['x'].float(), data['weight'].float()), dim=1)
    chosen = p.argmax(dim=1)
    used = data['used_token']
    capacity = combine.shape[2]
    assert (combine != 0).sum((1, 2)).max().item() <= 1
    for expert in range(p.shape[1]):
        nz = combine[:, expert, :].nonzero()
        expected = min(int(res_out[3][expert].item()), capacity)
        assert nz.shape[0] == expected, 'per-expert slot count differs'
        if expected:
            assert torch.equal(nz[:, 1], torch.arange(expected, device=nz.device))
            rows = nz[:, 0]
            assert torch.equal(chosen[rows], torch.full_like(rows, expert))
            if used is not None:
                assert bool((used[rows] == 1).all())
            torch.testing.assert_close(combine[rows, expert, nz[:, 1]],
                                       p[rows, expert], rtol=2e-5, atol=2e-5)


def measure(fn, inputs):
    import triton.testing
    return triton.testing.do_bench(lambda: fn(**inputs), warmup=25, rep=100)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label('v1_ds_topkgate_unified')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_ds_topkgate_unified(config):
    case = config
    inputs = to_device(input_build(case))
    ref_inputs = dict(inputs)
    res_inputs = dict(inputs)
    # The strided x is a view: keep the base storage to prove it is untouched.
    guard = inputs['x']._base.clone() if case[7] else None
    saved = {k: v.clone() for k, v in res_inputs.items() if isinstance(v, torch.Tensor)}
    expected = compute_ref(**ref_inputs)
    actual = compute_res(**res_inputs)
    compare(expected, actual, case)
    for key, value in saved.items():
        assert torch.equal(value, res_inputs[key]), f'candidate modified input: {key}'
        for out in actual:
            assert not torch._C._is_alias_of(out, res_inputs[key]), f'output aliases {key}'
    if guard is not None:
        assert torch.equal(guard, inputs['x']._base), 'candidate wrote through the strided view'
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float('inf'))
