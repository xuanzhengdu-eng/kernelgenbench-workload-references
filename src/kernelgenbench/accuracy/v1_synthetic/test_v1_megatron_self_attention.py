import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


def fresh(inputs):
    """Independent device copy of every tensor input (layout preserved)."""
    return {name: (value.to(kernelgenbench.device) if torch.is_tensor(value) else value)
            for name, value in inputs.items()}


def measure(fn, inputs, warmup=5, rep=20):
    """CUDA-event timing of the real entry point, warmup outside the interval."""
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(**inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


# (S, B, I, H, G, D, O): sequence, batch, hidden width, heads, query groups, head width,
# output width. Megatron's SelfAttention output projection always maps back to the hidden
# width, so O == I here; the old synthetic O != I case is not an upstream contract. G==H
# is plain MHA, G<H repeats K/V per group (GQA).
COMBOS = [(3, 1, 8, 2, 2, 4, 8), (4, 2, 8, 4, 2, 2, 8), (3, 2, 8, 4, 1, 2, 8), (5, 1, 8, 1, 1, 4, 8),
          (2, 3, 8, 4, 4, 2, 8)]


def input_build(config):
    """Deterministic values; qkv_weight is grouped as (Q heads of group, K, V) per group."""
    seq, batch, width, heads, groups, head_dim, out_width = config
    qkv_rows = groups * (heads // groups + 2) * head_dim
    gen = torch.Generator(device='cpu').manual_seed(20261115)

    def tensor(shape):
        count = 1
        for value in shape:
            count *= value
        return (((torch.arange(count, dtype=torch.float32) * 3) % 43) - 21).div(40).reshape(shape)

    mask = torch.ones((batch, 1, seq, seq), dtype=torch.bool).triu(1)
    return {'x': tensor((seq, batch, width)), 'qkv_weight': tensor((qkv_rows, width)),
            'qkv_bias': tensor((qkv_rows,)), 'out_weight': tensor((out_width, heads * head_dim)),
            'out_bias': tensor((out_width,)), 'mask': mask, 'heads': heads, 'groups': groups,
            'head_dim': head_dim}


def compute_ref(x, qkv_weight, qkv_bias, out_weight, out_bias, mask, heads, groups, head_dim):
    return kernelgenbench.baseline.v1_megatron_self_attention(
        x, qkv_weight, qkv_bias, out_weight, out_bias, mask, heads, groups, head_dim)


def compute_res(x, qkv_weight, qkv_bias, out_weight, out_bias, mask, heads, groups, head_dim):
    return kernelgenbench.solution.v1_megatron_self_attention(
        x, qkv_weight, qkv_bias, out_weight, out_bias, mask, heads, groups, head_dim)


def compare(ref_out, res_out, config):
    """Two-tensor result; fp32 eager math on both sides uses a tight tolerance."""
    seq, batch, _, heads, _, head_dim, out_width = config
    assert isinstance(res_out, (tuple, list)) and len(res_out) == 2
    assert ref_out[0].shape == res_out[0].shape == (seq, batch, out_width)
    assert ref_out[0].dtype == res_out[0].dtype == torch.float32
    assert ref_out[1].shape == res_out[1].shape == (out_width,)
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_out[1], ref_out[1], rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label('v1_megatron_self_attention')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_megatron_self_attention(config):
    seq, batch, _, heads, _, head_dim, out_width = config
    inputs = input_build(config)
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    saved = {name: value.clone() for name, value in res_inputs.items() if torch.is_tensor(value)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # Megatron's skip_bias_add contract: the reference hands back the very bias it was
    # given; the official layer hands back its own parameter. Only values are shared.
    assert ref_out[1] is ref_inputs['out_bias']
    for name, value in saved.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    for name in ('x', 'qkv_weight', 'qkv_bias', 'out_weight', 'out_bias', 'mask'):
        assert not torch._C._is_alias_of(res_out[0], res_inputs[name]), name
    assert not torch._C._is_alias_of(res_out[1], res_inputs['out_bias'])

    perf_inputs = fresh(inputs)
    ref_time = compute_ref_performance(**perf_inputs)
    res_time = compute_res_performance(**perf_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
