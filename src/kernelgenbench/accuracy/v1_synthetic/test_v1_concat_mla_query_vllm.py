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


# (tokens, heads, dtype, noncontiguous): padded source strides and a padded output base
# exercise the guard column after nope_dim + rope_dim.
COMBOS = [(tokens, heads, dtype, strided)
          for tokens, heads in [(1, 1), (3, 4), (17, 16)]
          for dtype in (torch.float16, torch.bfloat16)
          for strided in (False, True)]


def input_build(config):
    tokens, heads, dtype, strided = config
    gen = torch.Generator(device='cpu').manual_seed(20261110)

    def arranged(width):
        source = torch.randn((tokens, heads, width + (8 if strided else 0)), generator=gen,
                             dtype=dtype)
        return source[..., :width] if strided else source

    ql_nope, q_pe = arranged(512), arranged(64)
    if strided:
        base = torch.full((tokens, heads, 584), -9, dtype=dtype)
        q_out = base[..., :576]
    else:
        base = None
        q_out = torch.full((tokens, heads, 576), -9, dtype=dtype)
    return {'ql_nope': ql_nope, 'q_pe': q_pe, 'q_out': q_out}


def compute_ref(ql_nope, q_pe, q_out):
    return kernelgenbench.baseline.v1_concat_mla_query_vllm(ql_nope, q_pe, q_out)


def compute_res(ql_nope, q_pe, q_out):
    return kernelgenbench.solution.v1_concat_mla_query_vllm(ql_nope, q_pe, q_out)


def padded(base):
    """Re-create the padded [tokens, heads, 584] base so q_out keeps a guard column."""
    buffer = torch.full((base.shape[0], base.shape[1], 584), -9, dtype=base.dtype,
                        device=base.device)
    buffer[..., :576] = base
    return buffer


def compare(ref_out, res_out, config):
    tokens, heads, dtype, _ = config
    assert ref_out is None and res_out is None, 'the writeback entry returns None'
    return None


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label('v1_concat_mla_query_vllm')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_concat_mla_query_vllm(config):
    tokens, heads, dtype, strided = config
    inputs = input_build(config)
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    if strided:
        # .to(device) drops the view relationship, so rebuild the padded base on device.
        for side in (ref_inputs, res_inputs):
            side['q_out'] = padded(side['q_out'])[..., :576]
    saved = {name: value.clone() for name, value in res_inputs.items()}
    base_ref = ref_inputs['q_out']._base if strided else None
    pointer = res_inputs['q_out'].data_ptr()
    guard_ref = base_ref[..., 576:].clone() if strided else None

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    compare(None, None, config)

    assert res_inputs['q_out'].shape == (tokens, heads, 576) and res_inputs['q_out'].dtype == dtype
    assert res_inputs['q_out'].data_ptr() == pointer
    torch.testing.assert_close(res_inputs['q_out'], ref_inputs['q_out'], rtol=0, atol=0)
    for name in ('ql_nope', 'q_pe'):
        torch.testing.assert_close(res_inputs[name], saved[name], rtol=0, atol=0)
    if strided:
        out_res = res_inputs['q_out']._base
        torch.testing.assert_close(out_res[..., 576:], guard_ref, rtol=0, atol=0)

    perf_inputs = fresh(inputs)
    ref_time = compute_ref_performance(**perf_inputs)
    res_time = compute_res_performance(**perf_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
