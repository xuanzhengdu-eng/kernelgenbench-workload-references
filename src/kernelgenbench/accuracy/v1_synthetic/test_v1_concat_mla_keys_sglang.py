"""Accuracy and performance for the SGLang concat_mla_keys entry.

Broadcasts the shared [T, 1, 64] rotary key to 128 heads and writes
[out | nope | rope] in place: out [T, 128, 192].  The SGLang CUDA kernel is
bfloat16-only, so the cases cover bfloat16 with contiguous and row-padded
(`stride(0) > 192`) outputs plus padded input rows.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# tokens, dtype, pad the outer row strides, pad the output rows
COMBOS = [
    (1, torch.bfloat16, False, False),
    (3, torch.bfloat16, True, False),
    (3, torch.bfloat16, True, True),
    (17, torch.bfloat16, False, True),
]


def input_build(case):
    n, dtype, strided_in, strided_out = case
    gen = torch.Generator().manual_seed(20261110)
    pad = 8 if strided_in else 0
    k_nope = torch.randn((n, 128, 128 + pad), dtype=dtype, generator=gen)
    if strided_in:
        k_nope = k_nope[..., :128]
    k_rope = torch.randn((n, 1, 64 + pad), dtype=dtype, generator=gen)
    if strided_in:
        k_rope = k_rope[..., :64]
    out_pad = 8 if strided_out else 0
    guard = torch.full((n, 128, 192 + out_pad), -9, dtype=dtype)
    out = guard[..., :192] if out_pad else guard
    return dict(k_nope=k_nope, k_rope=k_rope, out=out, guard=guard)


def measure(fn, inputs):
    """Median single-call ms; input copies and warmup stay outside the timed region."""
    samples = []
    for _ in range(6):
        arguments = {name: value.to(device=kernelgenbench.device, copy=True)
                     if isinstance(value, torch.Tensor) else value
                     for name, value in inputs.items()}
        if kernelgenbench.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**arguments)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn(**arguments)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples[1:])


def _call(fn, kw):
    return fn(kw["k_nope"], kw["k_rope"], kw["out"])


def compute_ref(**kw):
    return _call(kernelgenbench.baseline.v1_concat_mla_keys_sglang, kw)


def compute_res(**kw):
    candidate = getattr(kernelgenbench.triton, "v1_concat_mla_keys_sglang", None)
    if candidate is None:
        candidate = kernelgenbench.triton.v1_concat_mla_keys_
    return _call(candidate, kw)


def compute_ref_performance(**kw):
    return measure(compute_ref, kw)


def compute_res_performance(**kw):
    return measure(compute_res, kw)


@label("v1_concat_mla_keys_")
@label("v1_concat_mla_keys_::sglang_bf16")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_concat_mla_keys_sglang(case):
    n, dtype, strided_in, strided_out = case
    cpu = input_build(case)
    width = cpu["out"].shape[-1]
    guard = cpu["guard"].clone()
    expected = torch.cat((cpu["k_nope"], cpu["k_rope"].expand(n, 128, 64)), dim=-1)
    expected = expected.to(kernelgenbench.device)

    def to_side():
        side = {}
        for name, value in cpu.items():
            if name == "guard":
                side[name] = value.to(kernelgenbench.device)
            elif name == "out" and value._base is not None:
                base = value._base.to(kernelgenbench.device)
                side[name] = base[..., :width]
            else:
                side[name] = value.to(kernelgenbench.device)
        return side

    ref_side, res_side = to_side(), to_side()
    saved_nope = res_side["k_nope"].clone()
    saved_rope = res_side["k_rope"].clone()
    ptrs = {id(side): side["out"].data_ptr() for side in (ref_side, res_side)}

    ref_out = compute_ref(**ref_side)
    res_out = compute_res(**res_side)
    torch.cuda.synchronize()

    assert ref_out is None and res_out is None
    for side in (ref_side, res_side):
        assert torch.equal(side["out"], expected)
        if strided_out:
            assert torch.equal(side["out"]._base[..., width:],
                               ref_side["guard"][..., width:]), "guard columns written"
        assert side["out"].data_ptr() == ptrs[id(side)]
        assert torch.equal(side["k_nope"], saved_nope)
        assert torch.equal(side["k_rope"], saved_rope)
        assert side["out"].shape == (n, 128, 192)

    ref_time = compute_ref_performance(**cpu)
    res_time = compute_res_performance(**cpu)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
