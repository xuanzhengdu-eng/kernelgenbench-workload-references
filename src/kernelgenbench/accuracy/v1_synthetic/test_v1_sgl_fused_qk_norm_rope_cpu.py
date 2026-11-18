"""Accuracy and performance for v1_sgl_fused_qk_norm_rope_cpu.

Operator: SGLang CPU ``fused_qk_norm_rope_cpu`` (per-head Q/K RMSNorm plus partial
RoPE, done in place by the ``sgl_kernel`` CPU extension), pinned commit
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/norm.cpp#L1371-L1436

Contract under test
-------------------
* ``q`` ``[n, q_heads*head_dim]`` and ``k`` ``[n, kv_heads*head_dim]``, 2-D,
  contiguous, FP16/BF16, same dtype as the two weight vectors and the cache.
* Per-head FP32 RMSNorm over ``head_dim`` with ``weight``, rounded to the input
  dtype, then partial RoPE over the first ``rotary_dim`` features.
* ``is_neox=True`` rotates the split halves; ``is_neox=False`` the interleaved
  pairs.  Both layouts are covered and must differ.
* In place: same tensor objects updated, ``None`` returned.
* ``positions`` accepts int32 and int64.
* No gate output at all: the gate split belongs to the vLLM item of row 125.

Item: v1_sgl_fused_qk_norm_rope_cpu (723-row 125, SGLang CPU branch).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# These entries are the pinned SGLang *CPU* kernels; the tensors stay on the host.
_CPU = torch.device("cpu")


# Axes: (n, q_heads, kv_heads, head_dim, rotary_dim, dtype, is_neox, position dtype).
# rotary_dim < head_dim leaves an unrotated RMSNorm-only tail.
COMBOS = [
    (5, 3, 1, 64, 32, torch.bfloat16, True, torch.int64),
    (4, 2, 2, 64, 64, torch.float16, False, torch.int32),
    (2, 1, 1, 16, 8, torch.bfloat16, False, torch.int64),
    (3, 4, 2, 32, 16, torch.float16, True, torch.int32),
    (0, 2, 1, 64, 32, torch.bfloat16, True, torch.int64),
]

TOLERANCE = {
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (8e-3, 4e-3),
}


def input_build(config):
    """q: [n, q_heads*head_dim]; k: [n, kv_heads*head_dim]; weights [head_dim];
    cos_sin_cache: [24, rotary_dim] packed [cos | sin]; positions [n] int32/int64.
    """
    n, q_heads, kv_heads, head_dim, rotary_dim, dtype, is_neox, pos_dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261227)
    q = torch.randn((n, q_heads * head_dim), dtype=dtype, generator=generator)
    k = torch.randn((n, kv_heads * head_dim), dtype=dtype, generator=generator)
    q_weight = torch.rand((head_dim,), dtype=dtype, generator=generator) * 0.2 + 0.8
    k_weight = torch.rand((head_dim,), dtype=dtype, generator=generator) * 0.2 + 0.8
    angle = (torch.arange(24, dtype=torch.float32)[:, None]
             * torch.linspace(0.02, 0.4, rotary_dim // 2)[None, :])
    cache = torch.cat((angle.cos(), angle.sin()), -1).to(dtype)
    positions = (torch.arange(n) % 20).to(pos_dtype)
    return {
        "q": q,
        "k": k,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "eps": 1e-5,
        "is_neox": is_neox,
        "positions": positions,
        "cos_sin_cache": cache,
        "rotary_dim": rotary_dim,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_fused_qk_norm_rope_cpu(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_fused_qk_norm_rope_cpu(**inputs)


def compare(ref_out, res_out, dtype=None):
    """Both sides return None; the verdict is the written-back state, checked by the
    caller against the pre-call snapshot.
    """
    assert ref_out is None and res_out is None, \
        f"the in-place op must return None, got {ref_out!r} and {res_out!r}"


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_sgl_fused_qk_norm_rope_cpu, inputs,
                          _CPU)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_sgl_fused_qk_norm_rope_cpu, inputs,
                          _CPU)


@label("v1_sgl_fused_qk_norm_rope_cpu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_fused_qk_norm_rope_cpu(config):
    n, q_heads, kv_heads, head_dim, rotary_dim, dtype, is_neox, pos_dtype = config
    inputs = input_build(config)
    # Two independent copies keep the same dtype/layout; the op is in place, so the
    # pre-call snapshot is the only record of the inputs.
    ref_inputs = checks.fresh(inputs, _CPU)
    res_inputs = checks.fresh(inputs, _CPU)
    before = {name: tensor.clone() for name, tensor in res_inputs.items()
              if isinstance(tensor, torch.Tensor)}
    q_ptr, k_ptr = res_inputs["q"].data_ptr(), res_inputs["k"].data_ptr()
    ref_q_ptr, ref_k_ptr = ref_inputs["q"].data_ptr(), ref_inputs["k"].data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    # Write-back identity: same storage, same shape.
    if n:
        assert res_inputs["q"].data_ptr() == q_ptr and res_inputs["k"].data_ptr() == k_ptr
        assert ref_inputs["q"].data_ptr() == ref_q_ptr and ref_inputs["k"].data_ptr() == ref_k_ptr
    rtol, atol = TOLERANCE[dtype]
    for name in ("q", "k"):
        assert res_inputs[name].shape == inputs[name].shape
        assert res_inputs[name].dtype == dtype
        torch.testing.assert_close(res_inputs[name].float(), ref_inputs[name].float(),
                                   rtol=rtol, atol=atol)
        # The op really wrote something (trivially vacuous for the empty case).
        if n:
            assert not torch.equal(res_inputs[name], before[name]), f"{name} was not updated"

    # The unrotated tail is exactly the normed value: compare with a plain RMSNorm.
    if n and rotary_dim < head_dim:
        for name, weight_name, heads in (("q", "q_weight", q_heads), ("k", "k_weight", kv_heads)):
            source = before[name].reshape(n, heads, head_dim).float()
            expected = (source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-5)
                        * res_inputs[weight_name].float()).to(dtype)
            tail = res_inputs[name].reshape(n, heads, head_dim)[..., rotary_dim:]
            torch.testing.assert_close(tail, expected[..., rotary_dim:], rtol=rtol, atol=atol)

    # is_neox selects a different layout: the two variants must not agree.
    if n:
        neox = dict(inputs, is_neox=True)
        interleaved = dict(inputs, is_neox=False)
        a = checks.fresh(neox, _CPU)
        b = checks.fresh(interleaved, _CPU)
        compute_ref(**a)
        compute_ref(**b)
        assert not torch.equal(a["q"], b["q"]), \
            "is_neox must change the rotated layout"

    # weights, cache and positions are read-only.
    for name in ("q_weight", "k_weight", "cos_sin_cache", "positions"):
        torch.testing.assert_close(res_inputs[name].cpu(), before[name].cpu(), rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
