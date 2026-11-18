"""Accuracy and performance for v1_vllm_linear_lora (723-row 101).

Reference: vLLM ``LoRALayerWeights``/``BaseLinearLoRA._apply_sync`` with the
CPU punica wrapper.
Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/layers/base_linear.py#L208-L242

The candidate side calls the official ``PunicaWrapperCPU.add_lora_linear`` on
top of ``F.linear``, which is the exact pair ``_apply_lora_to_output`` performs
(scale 1.0).  Cases cover mixed active/no-adapter rows (``-1``), all ``-1``, all
active, a 3-D input flattened for the adapter math, and ranks that differ from
the output width.  A ``-1`` row must stay bitwise equal to the base linear, and
no input may be modified.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows92_101_checks as checks


TOLERANCE = {
    torch.float32: (2e-5, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}

# leading input shape, input width, output width, rank, adapters, bias, id mode
CASES = [((4,), 8, 6, 3, 2, True, "mixed"), ((2, 3), 5, 7, 2, 3, False, "mixed"),
         ((2,), 4, 8, 2, 1, True, "none"), ((1,), 5, 3, 4, 2, True, "active"),
         ((5,), 6, 4, 3, 2, True, "prefill_mixed"),
         ((3,), 5, 3, 2, 1, False, "prefill_none")]

# ``prefill`` selects the official sgmv_* metadata path; False keeps the decode
# bgmv_* path.  Both are branches of the same ``add_lora_linear`` entry.
COMBOS = [(case, dtype, prefill) for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)
          for prefill in (False, True)]


def input_build(config):
    (lead, kin, kout, rank, adapters, has_bias, mode), dtype, prefill = config
    generator = torch.Generator(device="cpu").manual_seed(20261122)
    tokens = 1
    for size in lead:
        tokens *= size
    ids = torch.arange(tokens, dtype=torch.int64) % adapters
    if mode == "mixed":
        ids[::2] = -1
    elif mode in ("none", "prefill_none"):
        ids.fill_(-1)
    elif mode == "prefill_mixed":
        ids = torch.tensor([-1, -1, 0, 0, 1][:tokens], dtype=torch.int64)

    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=dtype) * 0.2

    return {
        "x": rand(*lead, kin),
        "base_weight": rand(kout, kin),
        "bias": rand(kout) if has_bias else None,
        "lora_a": rand(adapters, 1, rank, kin),
        "lora_b": rand(adapters, 1, kout, rank),
        "adapter_ids": ids,
        "prefill": prefill,
    }


def compute_ref(x, base_weight, bias, lora_a, lora_b, adapter_ids, prefill=False):
    return kernelgenbench.baseline.v1_vllm_linear_lora(
        x, base_weight, bias, lora_a, lora_b, adapter_ids, prefill)


def compute_res(x, base_weight, bias, lora_a, lora_b, adapter_ids, prefill=False):
    return kernelgenbench.solution.v1_vllm_linear_lora(
        x, base_weight, bias, lora_a, lora_b, adapter_ids, prefill)


def compare(ref_out, res_out, dtype):
    checks.assert_tensor(res_out, ref_out, dtype, *TOLERANCE[dtype])


def compute_ref_performance(x, base_weight, bias, lora_a, lora_b, adapter_ids,
                            prefill=False):
    return checks.measure(
        kernelgenbench.baseline.v1_vllm_linear_lora,
        {"x": x, "base_weight": base_weight, "bias": bias, "lora_a": lora_a,
         "lora_b": lora_b, "adapter_ids": adapter_ids, "prefill": prefill},
        x.device)


def compute_res_performance(x, base_weight, bias, lora_a, lora_b, adapter_ids,
                            prefill=False):
    return checks.measure(
        kernelgenbench.solution.v1_vllm_linear_lora,
        {"x": x, "base_weight": base_weight, "bias": bias, "lora_a": lora_a,
         "lora_b": lora_b, "adapter_ids": adapter_ids, "prefill": prefill},
        x.device)


@label("v1_vllm_linear_lora")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_linear_lora(config):
    (lead, kin, kout, rank, adapters, has_bias, mode), dtype, prefill = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {name: value.clone() for name, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}
    ref_before = {name: value.clone() for name, value in ref_inputs.items()
                  if isinstance(value, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert tuple(res_out.shape) == (*lead, kout)
    assert res_out.dtype == dtype
    for name, value in before.items():
        assert torch.equal(res_inputs[name], value), f"solution modified {name}"
    for name, value in ref_before.items():
        assert torch.equal(ref_inputs[name], value), f"reference modified {name}"

    # A "-1" row must equal the plain base linear for that row.
    ids = res_inputs["adapter_ids"].reshape(-1)
    base = torch.nn.functional.linear(res_inputs["x"], res_inputs["base_weight"],
                                      res_inputs["bias"]).reshape(-1, kout)
    no_lora = ids == -1
    if bool(no_lora.any()):
        torch.testing.assert_close(res_out.reshape(-1, kout)[no_lora],
                                   base[no_lora], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**ref_inputs),
                          compute_res_performance(**res_inputs))
