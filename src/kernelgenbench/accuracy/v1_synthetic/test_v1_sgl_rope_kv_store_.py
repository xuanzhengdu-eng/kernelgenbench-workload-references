"""Accuracy and performance for v1_sgl_rope_kv_store_.

Reference: local SGLang-contract reimplementation.
Solution: pinned SGLang apply_rope_inplace_with_kvcache (fused JIT kernel).
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.rows492_501_utils import close, timed
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# tokens, Q heads, KV heads, head/rope dim, NeoX, q dtype, k/v dtype,
# index dtype, padded row stride.  The padded rows check that the fused kernel
# accepts a non-contiguous [tokens, heads, dim] view for q/k/v.
COMBOS = [
    (3, 4, 2, 64, True, torch.bfloat16, torch.bfloat16, torch.int64, False),
    (4, 3, 1, 64, False, torch.float16, torch.float16, torch.int32, True),
    (2, 2, 2, 128, True, torch.float16, torch.bfloat16, torch.int64, False),
    (5, 4, 1, 128, False, torch.bfloat16, torch.bfloat16, torch.int32, True),
]


def input_build(config):
    n, qh, kh, d, neox, qd, kd, idtype, padded = config
    gen = torch.Generator(device="cpu").manual_seed(n * 1000 + qh * 100 + d)

    def rows(heads, dtype):
        x = torch.randn(n * (2 if padded else 1), heads, d,
                        generator=gen, dtype=torch.float32).to(dtype)
        return x[::2] if padded else x

    q, k, v = rows(qh, qd), rows(kh, kd), rows(kh, kd)
    k_cache = torch.randn(n + 4, kh * d, generator=gen).to(kd)
    v_cache = torch.randn(n + 4, kh * d, generator=gen).to(kd)
    # SGLang requires a float32 cos/sin table; first half cos, second half sin.
    angle = torch.randn(17, d // 2, generator=gen)
    table = torch.cat((angle.cos(), angle.sin()), -1).to(torch.float32)
    positions = (torch.arange(n) * 3 % 17).to(idtype)
    out_loc = torch.arange(n - 1, -1, -1, dtype=idtype) + 2
    return dict(q=q, k=k, v=v, k_cache=k_cache, v_cache=v_cache,
                cos_sin_cache=table, positions=positions, out_loc=out_loc,
                is_neox=neox, rope_dim=d)


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_sgl_rope_kv_store_(**kw)


def compute_res(**kw):
    return kernelgenbench.solution.v1_sgl_rope_kv_store_(**kw)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    assert ref_out is None and res_out is None, "the fused entry writes in place and returns None"
    for name in ("q", "k", "k_cache", "v_cache"):
        close(res_inputs[name], ref_inputs[name], ref_inputs[name].dtype)
    # v, the cos/sin table, positions and out_loc are read-only inputs.
    for name in ("v", "cos_sin_cache", "positions", "out_loc"):
        torch.testing.assert_close(res_inputs[name], ref_inputs[name], rtol=0, atol=0)
    # Slots outside out_loc must retain their pre-call contents on both sides.
    written = set(ref_inputs["out_loc"].tolist())
    for name in ("k_cache", "v_cache"):
        for slot in set(range(ref_inputs[name].shape[0])) - written:
            torch.testing.assert_close(res_inputs[name][slot], ref_inputs[name][slot],
                                       rtol=0, atol=0)


def compute_ref_performance(config):
    return timed(lambda: input_build(config), compute_ref)


def compute_res_performance(config):
    return timed(lambda: input_build(config), compute_res)


@label("v1_sgl_rope_kv_store_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_rope_kv_store_(config):
    ref_inputs = input_build(config)
    res_inputs = input_build(config)
    from kernelgenbench.rows492_501_utils import to_device
    ref_inputs = to_device(ref_inputs)
    res_inputs = to_device(res_inputs)

    n, qh, kh, d, neox, qd, kd, _, padded = config
    for side in (ref_inputs, res_inputs):
        assert side["q"].shape == (n, qh, d) and side["q"].dtype == qd
        assert side["k"].shape == (n, kh, d) and side["k"].dtype == kd
        assert side["q"].stride(0) == (2 * qh * d if padded else qh * d)
        assert side["k"].stride(0) == (2 * kh * d if padded else kh * d)
        assert side["q"].is_contiguous() is (not padded)
        assert side["k"].is_contiguous() is (not padded)
        assert side["k_cache"].shape[1] == kh * d
    q_ptr, k_ptr = ref_inputs["q"].data_ptr(), ref_inputs["k"].data_ptr()
    kc_ptr, vc_ptr = ref_inputs["k_cache"].data_ptr(), ref_inputs["v_cache"].data_ptr()
    v_before = ref_inputs["v"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compare(ref_out, res_out, ref_inputs, res_inputs, config)

    # In-place contract: q/k and both caches keep their storage; v is untouched.
    assert ref_inputs["q"].data_ptr() == q_ptr and ref_inputs["k"].data_ptr() == k_ptr
    assert ref_inputs["k_cache"].data_ptr() == kc_ptr
    assert ref_inputs["v_cache"].data_ptr() == vc_ptr
    torch.testing.assert_close(ref_inputs["v"], v_before, rtol=0, atol=0)
    # The rotated k and the original v are visible through the selected slots.
    assert torch.equal(ref_inputs["k_cache"][ref_inputs["out_loc"].long()],
                       ref_inputs["k"].flatten(1))
    assert torch.equal(ref_inputs["v_cache"][ref_inputs["out_loc"].long()],
                       ref_inputs["v"].flatten(1))

    ref_time = compute_ref_performance(config)
    res_time = compute_res_performance(config)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
