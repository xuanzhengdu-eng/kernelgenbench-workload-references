import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


def fresh(inputs):
    return {name: (value.to(kernelgenbench.device) if torch.is_tensor(value) else value)
            for name, value in inputs.items()}


def measure(fn, inputs, warmup=5, rep=20):
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


# (B, S, Hq, Hkv, D, Dv, dtype, segment ends, scale, noncontiguous). cu_seqlens splits
# the sequence into independently attended contiguous segments; None means one segment.
COMBOS = [(1, 7, 2, 2, 16, 16, torch.float32, None, None, False),
          (2, 12, 4, 2, 16, 24, torch.float32, (0, 4, 12), 0.2, True),
          (1, 9, 4, 4, 32, 32, torch.float16, (0, 2, 5, 9), None, True),
          (2, 8, 4, 2, 32, 16, torch.float16, None, 0.125, False),
          (1, 11, 4, 2, 32, 32, torch.bfloat16, (0, 3, 11), None, True),
          (2, 6, 2, 2, 16, 24, torch.bfloat16, None, 0.3, False)]


def input_build(config):
    """Q [B,S,Hq,D], K [B,S,Hkv,D], V [B,S,Hkv,Dv]; GQA repeats K/V per query group."""
    batch, seq, qheads, kvheads, dim, vdim, dtype, segments, scale, strided = config
    gen = torch.Generator(device='cpu').manual_seed(500 + batch + seq + dim)

    def arranged(shape):
        if strided:
            return torch.randn((*shape[:-1], shape[-1] * 2), generator=gen, dtype=dtype)[..., ::2]
        return torch.randn(shape, generator=gen, dtype=dtype)

    cu = None if segments is None else torch.tensor(segments, dtype=torch.int32)
    return {'query': arranged((batch, seq, qheads, dim)), 'key': arranged((batch, seq, kvheads, dim)),
            'value': arranged((batch, seq, kvheads, vdim)), 'scale': scale,
            'cu_seqlens': cu, 'enable_gqa': qheads != kvheads}


def compute_ref(query, key, value, scale, cu_seqlens, enable_gqa):
    return kernelgenbench.baseline.v1_dense_attention_bshd_vit(
        query, key, value, scale, cu_seqlens, enable_gqa)


def compute_res(query, key, value, scale, cu_seqlens, enable_gqa):
    return kernelgenbench.solution.v1_dense_attention_bshd_vit(
        query, key, value, scale, cu_seqlens, enable_gqa)


def compare(ref_out, res_out, config):
    batch, seq, qheads, _, _, vdim, dtype, _, _, _ = config
    assert res_out.shape == ref_out.shape == (batch, seq, qheads, vdim)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.is_contiguous()
    limit = {torch.float32: 2e-5, torch.float16: 8e-3, torch.bfloat16: 2.5e-2}[dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=limit, atol=limit)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label('v1_dense_attention_bshd_vit')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_dense_attention_bshd_vit(config):
    inputs = input_build(config)
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    saved = {name: value.clone() for name, value in res_inputs.items() if torch.is_tensor(value)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    for name, value in saved.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
        assert not torch._C._is_alias_of(res_out, res_inputs[name])

    perf_inputs = fresh(inputs)
    ref_time = compute_ref_performance(**perf_inputs)
    res_time = compute_res_performance(**perf_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
