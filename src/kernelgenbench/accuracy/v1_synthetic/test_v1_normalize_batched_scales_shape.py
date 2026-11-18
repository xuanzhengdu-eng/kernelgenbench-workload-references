"""Accuracy and performance for v1_normalize_batched_scales_shape (vLLM MoE).

Item: vLLM normalize_batched_scales_shape (row 74).
Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L682-L695

Key contract: 3-D input is returned unchanged; one-element input is
repeat_interleaved into (num_experts, 1, 1) (a COPY, no alias); other inputs
are reshaped to a VIEW of (num_experts, -1, last). ``num_experts`` is required.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows72_81_checks as checks

NAME = "v1_normalize_batched_scales_shape"

# (kind, num_experts, dtype)
COMBOS = [
    ("none", 4, torch.float32),
    ("scalar", 1, torch.float32),
    ("scalar", 4, torch.float16),
    ("vector", 1, torch.float32),
    ("matrix", 2, torch.float32),
    ("matrix", 4, torch.float16),
    ("batched", 4, torch.float32),
]


def input_build(config):
    """scales: None, (), (8,), (8,4) or (4,2,4); num_experts as listed."""
    kind, experts, dtype = config
    shapes = {"scalar": (), "vector": (8,), "matrix": (8, 4), "batched": (4, 2, 4)}
    if kind == "none":
        return {"scales": None, "num_experts": experts}
    size = 1
    for dim in shapes[kind]:
        size *= dim
    return {"scales": torch.arange(size, dtype=dtype).reshape(shapes[kind]),
            "num_experts": experts}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_normalize_batched_scales_shape(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_normalize_batched_scales_shape(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    src_a, src_b = ref_inputs["scales"], res_inputs["scales"]
    if src_a is None:
        assert ref_out is None and res_out is None
        return
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == src_a.dtype
    assert res_out.stride() == ref_out.stride()
    assert torch.equal(res_out, ref_out)
    alias_a = ref_out.untyped_storage().data_ptr() == src_a.untyped_storage().data_ptr()
    alias_b = res_out.untyped_storage().data_ptr() == src_b.untyped_storage().data_ptr()
    assert alias_b == alias_a
    if src_a.ndim >= 3:
        assert ref_out is src_a, "upstream returns the 3-D input object unchanged"
        assert res_out is src_b
        assert alias_a
    elif src_a.numel() == 1:
        assert not torch._C._is_alias_of(ref_out, src_a)
        assert not torch._C._is_alias_of(res_out, src_b)
    else:
        assert alias_a and alias_b, "non-scalar reshape must be a view"


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_normalize_batched_scales_shape,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_normalize_batched_scales_shape,
                          inputs, kernelgenbench.device)


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_normalize_batched_scales_shape(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {k: v.clone() for k, v in ref_inputs.items() if isinstance(v, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    for name, value in before.items():
        assert torch.equal(value, ref_inputs[name]), f"reference modified {name}"

    # Selecting more experts than the element count allows fails in torch.view
    # upstream; both sides must raise.
    bad = torch.arange(8, dtype=torch.float32, device=kernelgenbench.device)
    with pytest.raises(RuntimeError):
        kernelgenbench.baseline.v1_normalize_batched_scales_shape(bad, 4)
    bad_b = torch.arange(8, dtype=torch.float32, device=kernelgenbench.device)
    with pytest.raises(RuntimeError):
        kernelgenbench.solution.v1_normalize_batched_scales_shape(bad_b, 4)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
