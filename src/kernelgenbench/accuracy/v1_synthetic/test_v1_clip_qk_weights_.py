"""Accuracy and performance for v1_clip_qk_weights_.

Operator: Megatron local Q/K weight clipping and attention-logit reset
(``SelfAttention.clip_qk``); the tensor-parallel all-reduce is outside this
item.  The official method is called on a minimal object carrying a real
``nn.Linear`` QKV projection and a real ``core_attention`` logits buffer.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
  clip_qk           attention.py#L1531-L1578
  _clip_linear_qkv  attention.py#L1580-L1626

Item: v1_clip_qk_weights_ (723-row 104).  ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the official
method that the verifier bound for the same label.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (num_groups, q_size, kv_size, hidden, per-head max logits, alpha,
#  has_master_weight, dtype)
# The logits length is num_groups * heads_per_group and is the per-head max
# buffer the official core_attention records.  Cases cover clipping and
# no-clipping, alpha at both endpoints, and the optional master weight.
COMBOS = [
    (2, 4, 2, 5, (0.5, 1.0, 2.0, 4.0), 0.5, True, torch.float32),
    (2, 4, 2, 8, (0.2, 1.0, 0.5, 1.0), 0.5, False, torch.float32),
    (1, 4, 4, 7, (2.0, 5.0), 0.0, True, torch.float16),
    (2, 8, 4, 16, (2.0, 1.0, 0.5, 4.0), 1.0, False, torch.bfloat16),
    (4, 8, 4, 12, (0.5, 1.0, 2.0, 4.0, 0.25, 0.75, 3.0, 6.0), 0.25, True, torch.float32),
    (2, 4, 2, 5, (0.5, 1.0, 2.0, 4.0), 0.5, False, torch.float16),
]

TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """weight [q+2*kv, hidden] in the group-major QKV layout; main_weight is an
    independent fp32 master copy; state carries the per-head max logits."""
    groups, q_size, kv_size, hidden, logits, alpha, has_master, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    weight = torch.randn((q_size + 2 * kv_size, hidden), dtype=dtype,
                         generator=generator)
    main_weight = weight.float().clone() if has_master else None
    state = {
        "current_max_attn_logits": torch.tensor(logits, dtype=torch.float32),
        "eta": None,
        "eta_extended": None,
    }
    return {
        "weight": weight,
        "main_weight": main_weight,
        "state": state,
        "num_groups": groups,
        "q_size": q_size,
        "kv_size": kv_size,
        "threshold": 1.0,
        "alpha": alpha,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_clip_qk_weights_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_clip_qk_weights_(**inputs)


def compare(ref_out, res_out, dtype):
    assert ref_out is None and res_out is None


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_clip_qk_weights_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_clip_qk_weights_(config):
    groups, q_size, kv_size, hidden, logits, alpha, has_master, dtype = config
    inputs = input_build(config)
    saved = checks.fresh_nested(inputs)
    ref_inputs = checks.fresh_nested(inputs, kernelgenbench.device)
    res_inputs = checks.fresh_nested(inputs, kernelgenbench.device)

    original_ptrs = {}
    for side_name, side in (("ref", ref_inputs), ("res", res_inputs)):
        original_ptrs[side_name] = {
            "weight": side["weight"].data_ptr(),
            "main_weight": None if side["main_weight"] is None else side["main_weight"].data_ptr(),
        }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # In-place contract: the same storage is updated, not replaced.
    for side_name, side in (("ref", ref_inputs), ("res", res_inputs)):
        assert side["weight"].data_ptr() == original_ptrs[side_name]["weight"]
        if side["main_weight"] is not None:
            assert side["main_weight"].data_ptr() == original_ptrs[side_name]["main_weight"]
        assert side["state"]["current_max_attn_logits"] is None

    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_inputs["weight"].cpu(), ref_inputs["weight"].cpu(),
                               rtol=rtol, atol=atol)
    if has_master:
        torch.testing.assert_close(res_inputs["main_weight"].cpu(),
                                   ref_inputs["main_weight"].cpu(), rtol=rtol, atol=atol)
    else:
        assert ref_inputs["main_weight"] is None and res_inputs["main_weight"] is None

    for key in ("eta", "eta_extended"):
        ref_value = ref_inputs["state"][key]
        res_value = res_inputs["state"][key]
        if ref_value is None:
            assert res_value is None
        else:
            assert res_value is not None
            torch.testing.assert_close(res_value.cpu(), ref_value.cpu(), rtol=0, atol=0)

    # No-clipping case: weights must be byte-identical to the input and no eta
    # may be manufactured.
    grouped_max = torch.tensor(logits, dtype=torch.float32).view(groups, -1).max(dim=1).values
    if not bool(torch.any(grouped_max > 1.0)):
        for side in (ref_inputs, res_inputs):
            torch.testing.assert_close(side["weight"].cpu(), saved["weight"], rtol=0, atol=0)
            assert side["state"]["eta"] is None and side["state"]["eta_extended"] is None

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
