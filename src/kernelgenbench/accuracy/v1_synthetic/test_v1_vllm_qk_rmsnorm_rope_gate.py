"""Accuracy and performance for v1_vllm_qk_rmsnorm_rope_gate.

Operator: vLLM ``fused_qk_rmsnorm_rope_gate`` (Qwen attn-output-gate path), pinned
commit
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_qk_norm_rope.py#L139-L280

Contract under test
-------------------
* ``q_gate`` is ``[n, q_heads*2*head_dim]`` per head ``[q | gate]``; ``k`` is
  ``[n, kv_heads*head_dim]``.
* Per head the full ``head_dim`` is RMSNormed with ``weight + norm_beta`` and
  rounded back to the input dtype; the first ``rotary_dim`` features get partial
  NeoX RoPE, the tail passes through unrotated.
* Three contiguous fresh outputs ``(q_out, k_out, gate_out)``; ``gate_out`` is
  the raw pre-sigmoid gate copied verbatim, so it must be bit-identical.
* 1-D positions use the same table row for every rotary pair; 3-D positions
  (MRoPE) pick T/H/W per interleaved column.
* ``n == 0`` returns empty tensors; invalid rotary dim, dtype, stride and MRoPE
  arguments raise ``ValueError`` (upstream checks).

Item: v1_vllm_qk_rmsnorm_rope_gate (723-row 125).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# Axes: (n, q_heads, kv_heads, head_dim, rotary_dim, dtype, use_mrope, norm_beta).
# rotary_dim < head_dim exercises the RMSNorm-only pass-through tail; the empty
# case covers the early return.
COMBOS = [
    (5, 3, 1, 64, 32, torch.bfloat16, False, 0.0),
    (4, 2, 2, 64, 64, torch.float16, True, 0.0),
    (1, 4, 1, 128, 64, torch.bfloat16, True, 1.0),
    (7, 2, 1, 64, 32, torch.float16, False, 0.0),
    (3, 2, 2, 32, 16, torch.bfloat16, False, 0.5),
    (0, 2, 1, 64, 32, torch.bfloat16, False, 0.0),
]

TOLERANCE = {
    torch.float16: (4e-3, 2e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


def input_build(config):
    """q_gate: [n, q_heads*2*head_dim]; k: [n, kv_heads*head_dim];
    q_weight/k_weight: [head_dim]; cos_sin_cache: [24, rotary_dim] packed
    [cos | sin]; positions: [n] or [3, n] with T/H/W rows.
    """
    n, q_heads, kv_heads, head_dim, rotary_dim, dtype, use_mrope, norm_beta = config
    generator = torch.Generator(device="cpu").manual_seed(20261225)
    q_gate = torch.randn((n, q_heads * 2 * head_dim), dtype=dtype, generator=generator)
    k = torch.randn((n, kv_heads * head_dim), dtype=dtype, generator=generator)
    q_weight = torch.rand((head_dim,), dtype=dtype, generator=generator) * 0.2 + 0.8
    k_weight = torch.rand((head_dim,), dtype=dtype, generator=generator) * 0.2 + 0.8
    angle = (torch.arange(24, dtype=torch.float32)[:, None]
             * torch.linspace(0.02, 0.4, rotary_dim // 2)[None, :])
    cache = torch.cat((angle.cos(), angle.sin()), -1).to(dtype)
    base = torch.arange(n, dtype=torch.int64) % 20
    mrope_section = None
    if use_mrope:
        positions = torch.stack((base, (base + 3) % 24, (base + 7) % 24))
        half = rotary_dim // 2
        mrope_section = (half // 3 + half % 3, half // 3, half // 3)
    else:
        positions = base
    return {
        "q_gate": q_gate,
        "k": k,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos_sin_cache": cache,
        "positions": positions,
        "eps": 1e-5,
        "num_q_heads": q_heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "mrope_section": mrope_section,
        "norm_beta": norm_beta,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_qk_rmsnorm_rope_gate(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_qk_rmsnorm_rope_gate(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, (tuple, list)) and len(res_out) == 3, \
        f"expected (q, k, gate), got {type(res_out).__name__}"
    rtol, atol = TOLERANCE[dtype or ref_out[0].dtype]
    for index, (expected, actual) in enumerate(zip(ref_out, res_out)):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape, \
            f"output {index}: {tuple(actual.shape)} != {tuple(expected.shape)}"
        assert actual.dtype == expected.dtype == dtype
        assert actual.is_contiguous(), f"output {index} must be contiguous"
        torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    # The gate is a verbatim copy of the pre-sigmoid half: bit-identical.
    assert torch.equal(res_out[2], ref_out[2]), "gate must be copied verbatim"
    # q/k never share storage with the gate output (empty tensors carry no storage).
    if res_out[0].numel():
        assert res_out[0].data_ptr() != res_out[2].data_ptr()
        assert res_out[1].data_ptr() != res_out[2].data_ptr()


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_vllm_qk_rmsnorm_rope_gate, inputs,
                          kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_vllm_qk_rmsnorm_rope_gate, inputs,
                          kernelgenbench.device)


@label("v1_vllm_qk_rmsnorm_rope_gate")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_qk_rmsnorm_rope_gate(config):
    n, q_heads, kv_heads, head_dim, rotary_dim, dtype, use_mrope, norm_beta = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    # Inputs are read-only, and outputs are fresh.
    for name, tensor in res_inputs.items():
        if isinstance(tensor, torch.Tensor):
            torch.testing.assert_close(tensor.cpu(), inputs[name], rtol=0, atol=0)
    for produced in res_out:
        for name, tensor in res_inputs.items():
            if isinstance(tensor, torch.Tensor) and tensor.numel():
                assert not torch._C._is_alias_of(produced, tensor), f"output aliases {name}"

    # Gate output equals the untouched gate half of the input, bit for bit.
    if n:
        gate_half = ref_inputs["q_gate"].reshape(n, q_heads, 2, head_dim)[:, :, 1].reshape(n, -1)
        assert torch.equal(res_out[2], gate_half.contiguous()), \
            "gate output must be the raw gate half of q_gate"

    # The rotary tail beyond rotary_dim is a pure RMSNorm pass-through.
    if n and rotary_dim < head_dim:
        q_source = ref_inputs["q_gate"].reshape(n, q_heads, 2, head_dim)[:, :, 0]
        k_source = ref_inputs["k"].reshape(n, kv_heads, head_dim)
        for res_tensor, weight, source in (
                (res_out[0], ref_inputs["q_weight"], q_source),
                (res_out[1], ref_inputs["k_weight"], k_source)):
            result = res_tensor.reshape(n, source.shape[1], head_dim)[..., rotary_dim:].float()
            xf = source.float()
            expected = ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
                         * (weight.float() + norm_beta)).to(dtype)[..., rotary_dim:].float())
            torch.testing.assert_close(result, expected, rtol=TOLERANCE[dtype][0],
                                       atol=TOLERANCE[dtype][1])

    # Upstream argument checks: both sides must reject the same inputs.
    if n:
        bad = dict(inputs)
        bad["rotary_dim"] = head_dim + 2
        bad["mrope_section"] = None
        bad["positions"] = torch.arange(n, dtype=torch.int64)
        for call in (kernelgenbench.baseline.v1_vllm_qk_rmsnorm_rope_gate,
                     kernelgenbench.solution.v1_vllm_qk_rmsnorm_rope_gate):
            with pytest.raises(ValueError):
                call(**checks.fresh(bad, kernelgenbench.device))
        if use_mrope:
            bad_section = dict(inputs, mrope_section=(1, 1, 1))
            for call in (kernelgenbench.baseline.v1_vllm_qk_rmsnorm_rope_gate,
                         kernelgenbench.solution.v1_vllm_qk_rmsnorm_rope_gate):
                with pytest.raises(ValueError):
                    call(**checks.fresh(bad_section, kernelgenbench.device))

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
