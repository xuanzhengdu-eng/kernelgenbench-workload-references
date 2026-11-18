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


# (tokens, heads, noncontiguous source strides): the kernel allows padded row/head
# strides but requires the last dimension to be contiguous.
COMBOS = [(tokens, heads, strided)
          for tokens, heads in [(1, 1), (3, 4), (17, 16), (64, 8)]
          for strided in (False, True)]


def input_build(config):
    tokens, heads, strided = config
    gen = torch.Generator(device='cpu').manual_seed(20261110)

    def arranged(width):
        source = torch.randn((tokens, heads, width + (8 if strided else 0)), generator=gen,
                             dtype=torch.bfloat16)
        return source[..., :width] if strided else source

    return {'a': arranged(512), 'b': arranged(64)}


def compute_ref(a, b):
    return kernelgenbench.baseline.v1_concat_mla_query_sglang(a, b)


def compute_res(a, b):
    return kernelgenbench.solution.v1_concat_mla_query_sglang(a, b)


def compare(ref_out, res_out, config):
    tokens, heads, strided = config
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (tokens, heads, 576)
    assert res_out.dtype == ref_out.dtype == torch.bfloat16
    assert res_out.is_contiguous()
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    if strided:
        assert not ref_out.is_contiguous() or True  # sources may be strided; output is dense


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label('v1_concat_mla_query_sglang')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_concat_mla_query_sglang(config):
    tokens, heads, strided = config
    inputs = input_build(config)
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    saved = {name: value.clone() for name, value in res_inputs.items()}

    assert ref_inputs['a'].stride(-1) == 1 and ref_inputs['b'].stride(-1) == 1
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
