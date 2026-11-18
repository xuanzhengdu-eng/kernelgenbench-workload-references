"""Exercise offsets, dtype promotion, partial rotation and non-mutation."""
# Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/rotary_embedding/rope_variant.py#L472-L505
import statistics
from types import SimpleNamespace

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (input dtype, cache dtype, query heads, key heads, rotary width, NeoX pairing, offsets)
COMBOS = [
    (torch.float32, torch.float32, 4, 1, 8, True, False),
    (torch.float16, torch.float16, 2, 2, 4, False, True),
    (torch.bfloat16, torch.float32, 3, 1, 4, True, True),
    (torch.float16, torch.float32, 4, 2, 8, False, False),
]


def input_build(config):
    dtype, cache_dtype, qheads, kheads, rotary_dim, neox, has_offsets = config
    device = kernelgenbench.device
    generator = torch.Generator().manual_seed(141)
    def rand(heads):
        return torch.randn((2, 3, heads, 8), generator=generator).to(device=device, dtype=dtype)
    positions = torch.tensor([[0, 2, 4], [1, 0, 3]], device=device)
    offsets = torch.tensor([[1, 2, 1], [2, 1, 1]], device=device) if has_offsets else None
    angle = torch.arange(9, device=device)[:, None] * torch.linspace(.15, .9, rotary_dim // 2, device=device)
    cache = torch.cat((angle.cos(), angle.sin()), dim=-1).to(cache_dtype)
    state = dict(head_size=8, rotary_dim=rotary_dim, is_neox_style=neox, cos_sin_cache=cache)
    return dict(positions=positions, query=rand(qheads), key=rand(kheads), offsets=offsets, rotary=SimpleNamespace(**state))


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_deepseek_scaling_rope_native(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_deepseek_scaling_rope_native(**inputs)


def compare(result, reference, inputs, config):
    dtype, cache_dtype, _, _, rotary_dim, _, _ = config
    assert isinstance(result, tuple) and len(result) == 2
    for actual, expected, original in zip(result, reference, (inputs["query"], inputs["key"])):
        assert actual.shape == expected.shape == original.shape
        assert actual.dtype == expected.dtype == dtype
        assert not torch._C._is_alias_of(actual, original)
        torch.testing.assert_close(actual, expected)
        if rotary_dim < original.size(-1):
            torch.testing.assert_close(actual[..., rotary_dim:], original[..., rotary_dim:].to(actual.dtype), rtol=0, atol=0)


def _bench(fn, inputs):
    for _ in range(3):
        fn(**inputs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(7):
        start.record()
        fn(**inputs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@label("v1_sgl_deepseek_scaling_rope_native")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_deepseek_scaling_rope_native(config):
    ref_inputs, res_inputs = input_build(config), input_build(config)
    original = {key: value.clone() for key, value in res_inputs.items() if isinstance(value, torch.Tensor)}
    ref_cache, result_cache = ref_inputs["rotary"].cos_sin_cache, res_inputs["rotary"].cos_sin_cache
    cache_before = result_cache.clone()
    expected = compute_ref(**ref_inputs)
    output = compute_res(**res_inputs)
    compare(output, expected, res_inputs, config)
    for name, value in original.items():
        torch.testing.assert_close(ref_inputs[name], value, rtol=0, atol=0)
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    assert torch.equal(ref_cache, cache_before) and torch.equal(result_cache, cache_before)
    rt, st = _bench(compute_ref, ref_inputs), _bench(compute_res, res_inputs)
    return CustomBenchmarkResult(ref_time=rt, res_time=st, speedup=rt / st if st else float("inf"))
