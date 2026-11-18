"""Accuracy and performance for v1_megatron_mla_rope_backward_.

Operator: Megatron MLA YaRN RoPE backward for the query
(``ApplyMLARotaryEmbQ.backward``, SBHD layout, CP size one).  Official source
(pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L270-L326

Contract under test
-------------------
* ``grad`` is ``[S, B, H, qk_head_dim + emb_dim]``, last-dim contiguous.
* The op is in place on the last ``emb_dim`` columns: it reads the split halves
  ``[left, right]`` written by the forward and writes the interleaved pairs
  ``[left*cos_l + right*sin_r, -left*sin_l + right*cos_r]``.
* The first ``qk_head_dim`` columns are untouched, and the returned first element
  is the *same object* as ``grad``.
* Return value: a 9-tuple ``(grad, None, None, ...)``: this is the autograd
  backward of the 9-input forward (q, cos, sin, qk_head_dim, emb_dim,
  cu_seqlens_q, cp_rank, cp_size, rotary_interleaved).
* The fork-only ``inverse`` flag and interleaved-output switch have no official
  counterpart and are not part of this item.

Item: v1_megatron_mla_rope_backward_ (723-row 123).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# Axes: (S, B, H, qk_head_dim, emb_dim, dtype).  emb_dim // 2 and qk_head_dim must
# be powers of two for the Triton block layout; the rotation only touches the tail.
COMBOS = [
    (3, 2, 4, 8, 16, torch.float32),
    (4, 1, 4, 16, 16, torch.float32),
    (2, 2, 8, 8, 16, torch.bfloat16),
    (3, 1, 2, 16, 32, torch.bfloat16),
    (1, 2, 4, 8, 16, torch.float16),
]

TOLERANCE = {
    torch.float32: (2e-6, 1e-6),
    torch.float16: (3e-3, 2e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


def input_build(config):
    """grad: [S, B, H, qk_head_dim + emb_dim]; cos/sin: [S, 1, 1, emb_dim]."""
    seq, batch, heads, qk_head_dim, emb_dim, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261223)
    grad = torch.randn((seq, batch, heads, qk_head_dim + emb_dim), dtype=dtype,
                       generator=generator)
    angle = torch.randn((seq, 1, 1, emb_dim), dtype=torch.float32, generator=generator)
    return {
        "grad": grad,
        "cos": angle.cos().contiguous(),
        "sin": angle.sin().contiguous(),
        "qk_head_dim": qk_head_dim,
        "emb_dim": emb_dim,
    }


def compute_ref(grad, cos, sin, qk_head_dim, emb_dim):
    return kernelgenbench.baseline.v1_megatron_mla_rope_backward_(
        grad, cos, sin, qk_head_dim, emb_dim)


def compute_res(grad, cos, sin, qk_head_dim, emb_dim):
    return kernelgenbench.solution.v1_megatron_mla_rope_backward_(
        grad, cos, sin, qk_head_dim, emb_dim)


def compare(ref_out, res_out, dtype=None):
    """Nine entries, eight trailing ``None``; the gradient is written in place."""
    assert isinstance(res_out, tuple) and len(res_out) == 9, \
        f"expected the 9-entry autograd backward tuple, got {type(res_out).__name__}"
    assert all(entry is None for entry in res_out[1:]), \
        "every non-gradient entry of the backward tuple must be None"
    assert isinstance(res_out[0], torch.Tensor)
    assert res_out[0].shape == ref_out[0].shape
    assert res_out[0].dtype == ref_out[0].dtype
    rtol, atol = TOLERANCE[dtype or ref_out[0].dtype]
    torch.testing.assert_close(res_out[0].float(), ref_out[0].float(), rtol=rtol, atol=atol)


def compute_ref_performance(grad, cos, sin, qk_head_dim, emb_dim):
    return checks.measure(
        kernelgenbench.baseline.v1_megatron_mla_rope_backward_,
        {"grad": grad, "cos": cos, "sin": sin,
         "qk_head_dim": qk_head_dim, "emb_dim": emb_dim},
        kernelgenbench.device,
    )


def compute_res_performance(grad, cos, sin, qk_head_dim, emb_dim):
    return checks.measure(
        kernelgenbench.solution.v1_megatron_mla_rope_backward_,
        {"grad": grad, "cos": cos, "sin": sin,
         "qk_head_dim": qk_head_dim, "emb_dim": emb_dim},
        kernelgenbench.device,
    )


@label("v1_megatron_mla_rope_backward_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_mla_rope_backward_(config):
    seq, batch, heads, qk_head_dim, emb_dim, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    nope_before = res_inputs["grad"][..., :qk_head_dim].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    # Write-back object and untouched prefix, on both sides.
    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        written = out[0]
        assert written.data_ptr() == side["grad"].data_ptr(), \
            "the backward must write into the incoming gradient tensor"
        torch.testing.assert_close(written[..., :qk_head_dim], nope_before, rtol=0, atol=0)

    # cos/sin are read-only inputs.
    for name in ("cos", "sin"):
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
