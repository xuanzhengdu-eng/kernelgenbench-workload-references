"""Accuracy and performance for v1_vllm_routed_shared.

Operator: vLLM modular MoE runner, routed experts + separately returned shared
expert (``MoERunner._apply_quant_method``).

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L867-L937

Contract: single-rank FP32, no bias.  Returns ``(shared[T,D], routed[T,D])`` as
two independent tensors; inputs are not modified.  Routing is top-k over
``logits``; ``renormalize=True`` softmaxes the selected logits, ``False`` uses
``exp(value - logsumexp(logits))``.

Precision note: the routed half is computed by the pinned Triton
``fused_moe_kernel``, whose ``tl.dot`` runs fp32 inputs through tf32 tensor
cores.  Measured |ref - op| for these shapes is at most 1.4e-4, so the routed
comparison uses the tf32 allowance below; the shared half (plain fp32 GEMMs) is
bit-identical.

``compute_ref`` loads the transcribed reference from ``kernelgenbench.baseline``;
``compute_res`` calls the operator bound as ``kernelgenbench.solution``, i.e.
``fused_experts_op`` plus the real ``silu_and_mul`` CUDA kernel.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows702_711_checks as checks


# (tokens, hidden, experts, routed intermediate, shared intermediate, top-k,
#  renormalize).  The last two cases exercise topk < experts without renormalize.
COMBOS = [
    (1, 4, 3, 5, 6, 1, True),
    (3, 8, 4, 6, 5, 2, True),
    (5, 6, 3, 4, 7, 3, False),
    (4, 8, 5, 7, 3, 2, False),
]

# tf32 tensor-core rounding inside the pinned Triton fused-MoE kernel.
ROUTED_RTOL = 2e-3
ROUTED_ATOL = 3e-4


def input_build(config):
    """FP32 CPU tensors for one MoE layer; weights follow the reference layout."""
    t, d, e, f, g, topk, renormalize = config
    generator = torch.Generator(device="cpu").manual_seed(492 + t * 17 + e)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * .3

    return {
        "x": rand(t, d),
        "logits": rand(t, e),
        "shared_x": rand(t, d),
        "w13": rand(e, 2 * f, d),
        "w2": rand(e, d, f),
        "shared_w13": rand(2 * g, d),
        "shared_w2": rand(d, g),
        "topk": topk,
        "renormalize": renormalize,
    }


def compute_ref(x, logits, shared_x, w13, w2, shared_w13, shared_w2, topk,
                renormalize):
    return kernelgenbench.baseline.v1_vllm_routed_shared(
        x, logits, shared_x, w13, w2, shared_w13, shared_w2, topk, renormalize)


def compute_res(x, logits, shared_x, w13, w2, shared_w13, shared_w2, topk,
                renormalize):
    return kernelgenbench.solution.v1_vllm_routed_shared(
        x, logits, shared_x, w13, w2, shared_w13, shared_w2, topk, renormalize)


def compare(ref_out, res_out, config):
    """Two tensors in the documented order: shared first, routed second."""
    t, d = config[0], config[1]
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    for name, got, want in (("shared", res_out[0], ref_out[0]),
                            ("routed", res_out[1], ref_out[1])):
        assert isinstance(got, torch.Tensor), f"{name} is not a tensor"
        assert got.shape == want.shape == (t, d)
        assert got.dtype == want.dtype == torch.float32
        assert got.device == want.device
        if name == "shared":
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        else:
            torch.testing.assert_close(got, want, rtol=ROUTED_RTOL,
                                       atol=ROUTED_ATOL)


def compute_ref_performance(x, logits, shared_x, w13, w2, shared_w13, shared_w2,
                            topk, renormalize):
    return checks.measure(
        kernelgenbench.baseline.v1_vllm_routed_shared,
        {"x": x, "logits": logits, "shared_x": shared_x, "w13": w13, "w2": w2,
         "shared_w13": shared_w13, "shared_w2": shared_w2, "topk": topk,
         "renormalize": renormalize},
        kernelgenbench.device,
    )


def compute_res_performance(x, logits, shared_x, w13, w2, shared_w13, shared_w2,
                            topk, renormalize):
    return checks.measure(
        kernelgenbench.solution.v1_vllm_routed_shared,
        {"x": x, "logits": logits, "shared_x": shared_x, "w13": w13, "w2": w2,
         "shared_w13": shared_w13, "shared_w2": shared_w2, "topk": topk,
         "renormalize": renormalize},
        kernelgenbench.device,
    )


@label("v1_vllm_routed_shared")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_routed_shared(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["x"].shape == (config[0], config[1])
        assert side["logits"].shape == (config[0], config[2])
        assert side["w13"].shape == (config[2], 2 * config[3], config[1])

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # The two outputs are distinct fresh buffers and no input is modified.
    assert not torch._C._is_alias_of(res_out[0], res_out[1])
    for produced in res_out:
        for name, tensor in res_inputs.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch._C._is_alias_of(produced, tensor), \
                    f"output aliases {name}"
    for name, tensor in res_inputs.items():
        if not isinstance(tensor, torch.Tensor):
            assert tensor == inputs[name]
            continue
        torch.testing.assert_close(tensor.cpu(), inputs[name], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
