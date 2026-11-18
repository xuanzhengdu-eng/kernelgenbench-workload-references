"""Accuracy and performance for v1_sgl_mla_qkv_rope.

Operator: SGLang CPU ``qkv_proj_with_rope_fused_weight`` (the fused MLA QKV
projection, RMSNorm, absorbed bmm and interleaved RoPE used by the DeepSeek MLA
CPU path), pinned commit
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/qkv_proj.cpp#L623-L690

Contract under test
-------------------
* BF16 logical (unpacked) matrices: ``qkv_a_proj_weight``
  ``[q_lora + kv_lora + rope, hidden]``, ``q_b_proj_weight``
  ``[heads * (nope + rope), q_lora]``, ``w_kc`` ``[heads, kv_lora, nope]``.
* FP32 RMSNorm over ``q_lora`` and over ``kv_lora``, rounded back to BF16.
* Interleaved RoPE over the trailing ``rope`` features of Q (per head) and of the
  shared K, with ``rope % 32 == 0`` (kernel requirement).
* Returns ``(q, k, v)`` with ``q`` ``[tokens, heads, kv_lora + rope]``,
  ``k`` ``[tokens, 1, kv_lora + rope]`` and ``v`` a *view* of ``k``'s first
  ``kv_lora`` columns: same storage, same offset, same strides.
* Inputs are read-only.

Item: v1_sgl_mla_qkv_rope (723-row 126).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# These entries are the pinned SGLang *CPU* kernels; the tensors stay on the host.
_CPU = torch.device("cpu")


# Axes: (tokens, heads, eps).  hidden=64, q_lora=32, kv_lora=32, nope=32 and
# rope=32 are fixed so every matrix is small but VNNI-shaped.
COMBOS = [
    (1, 1, 1e-5),
    (2, 2, 1e-5),
    (3, 2, 1e-4),
    (4, 3, 1e-6),
    (5, 4, 1e-5),
]

TOLERANCE = (2.5e-2, 6e-3)


def input_build(config):
    """hidden_states [tokens, 64]; qkv_a_proj_weight [96, 64]; q_b_proj_weight
    [heads*64, 32]; w_kc [heads, 32, 32]; norm weights [32]; positions [tokens];
    cos_sin_cache [7, 32] packed [cos | sin] in BF16.
    """
    tokens, heads, eps = config
    generator = torch.Generator(device="cpu").manual_seed(20261226)

    def rand(*shape):
        return (torch.randn(shape, generator=generator) * 0.1).to(torch.bfloat16)

    phase = torch.randn((7, 16), generator=generator) * 0.5
    return {
        "hidden_states": rand(tokens, 64),
        "qkv_a_proj_weight": rand(96, 64),
        "q_b_proj_weight": rand(heads * 64, 32),
        "w_kc": rand(heads, 32, 32),
        "q_a_layernorm_weight": rand(32) + 1,
        "kv_a_layernorm_weight": rand(32) + 1,
        "positions": (torch.arange(tokens, dtype=torch.int64) * 2 % 7),
        "cos_sin_cache": torch.cat((phase.cos(), phase.sin()), -1).to(torch.bfloat16),
        "eps": eps,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_mla_qkv_rope(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_mla_qkv_rope(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, (tuple, list)) and len(res_out) == 3, \
        f"expected (q, k, v), got {type(res_out).__name__}"
    for index, (expected, actual) in enumerate(zip(ref_out, res_out)):
        assert actual.shape == expected.shape, \
            f"output {index}: {tuple(actual.shape)} != {tuple(expected.shape)}"
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual.float(), expected.float(),
                                   rtol=TOLERANCE[0], atol=TOLERANCE[1])
    # v is the prefix view of k: identical storage, offset and strides.
    key, value = res_out[1], res_out[2]
    assert value.untyped_storage().data_ptr() == key.untyped_storage().data_ptr(), \
        "v must share k's storage"
    assert value.storage_offset() == key.storage_offset()
    assert value.stride() == key.stride()


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_sgl_mla_qkv_rope, inputs,
                          _CPU)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_sgl_mla_qkv_rope, inputs,
                          _CPU)


@label("v1_sgl_mla_qkv_rope")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_mla_qkv_rope(config):
    tokens, heads, eps = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, _CPU)
    res_inputs = checks.fresh(inputs, _CPU)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    q, k, v = res_out
    assert q.shape == (tokens, heads, 64)
    assert k.shape == (tokens, 1, 64)
    assert v.shape == (tokens, 1, 32)
    # v is exactly k[..., :32] element-wise.
    torch.testing.assert_close(v, k[..., :32], rtol=0, atol=0)
    # Only the RoPE tail differs from a plain copy; the latent prefix of k is the
    # RMSNormed compressed KV (non-zero, so a candidate cannot pass by returning zeros).
    assert k[..., :32].abs().sum() > 0
    assert q[..., 32:].abs().sum() > 0

    # Inputs are read-only.
    for name, tensor in res_inputs.items():
        if isinstance(tensor, torch.Tensor):
            torch.testing.assert_close(tensor.cpu(), inputs[name], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
