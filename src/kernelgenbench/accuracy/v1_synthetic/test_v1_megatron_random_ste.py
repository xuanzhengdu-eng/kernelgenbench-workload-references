"""Accuracy and performance for v1_megatron_random_ste.

Official entry: ``apply_random_logits`` / ``RandomSTE.forward`` (Megatron-LM
266f1c97dca477ca8b92c16087506da2000b0b84, moe_utils.py L1194-L1242).
The forward clones the input and fills the clone with N(0,1) draws from the
expert-parallel RNG tracker.  The test fixes the tracker seed, so the result is
reproducible; the input, its storage and its strides stay unchanged.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.moe_rows192_201_utils import (
    fresh,
    measure,
    timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    ((0, 8), torch.float32),
    ((4, 8), torch.float32),
    ((7, 5), torch.float16),
    ((2, 16), torch.bfloat16),
]


def input_build(config):
    shape, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(1931)
    logits = torch.randn(shape, dtype=dtype, generator=generator)
    return {"logits": logits, "expert_seed": 101}


def compute_ref(logits, expert_seed):
    return kernelgenbench.baseline.v1_megatron_random_ste(logits, expert_seed)


def compute_res(logits, expert_seed):
    return kernelgenbench.solution.v1_megatron_random_ste(logits, expert_seed)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape
    assert ref_out.dtype == res_out.dtype == dtype
    # Same seed through the tracker and through a fresh generator must match bitwise.
    assert torch.equal(ref_out, res_out)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_random_ste")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_random_ste(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_ref = ref_inputs["logits"].clone()
    before_res = res_inputs["logits"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The forward must not modify the logits or alias their storage.
    for side, before in ((ref_inputs, before_ref), (res_inputs, before_res)):
        assert torch.equal(side["logits"], before)
        assert not torch._C._is_alias_of(side["logits"], ref_out)
        assert not torch._C._is_alias_of(side["logits"], res_out)

    # Re-seeding the tracker reproduces the draw; a different seed changes it.
    repeat = compute_res(**fresh(inputs, kernelgenbench.device))
    assert torch.equal(repeat, res_out)
    changed = compute_res(logits=res_inputs["logits"], expert_seed=102)
    if res_out.numel():
        assert not torch.equal(changed, res_out)

    # RandomSTE.backward is the identity.
    for fn in (compute_ref, compute_res):
        grad_input = inputs["logits"].clone().to(kernelgenbench.device).requires_grad_(True)
        out = fn(logits=grad_input, expert_seed=101)
        out.sum().backward()
        assert torch.equal(grad_input.grad, torch.ones_like(grad_input))

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
