"""Accuracy and performance for v1_megatron_random_ste_shared.

Official entry: ``apply_biased_logits`` / ``RandomSTEShared.forward``
(Megatron-LM 266f1c97dca477ca8b92c16087506da2000b0b84, moe_utils.py
L1245-L1290).  Random logits come first; a bias drawn from the data-parallel
RNG is added.  A negative ``std`` caches that bias per layer and reuses it on
later calls; non-negative ``std`` never writes the cache.
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


# (shape, std, layer_number, has cached bias, dtype)
COMBOS = [
    ((4, 8), 0.2, 3, False, torch.float32),
    ((4, 8), -0.4, 3, False, torch.float32),
    ((4, 8), -0.4, 3, True, torch.float32),
    ((2, 16), -0.2, None, False, torch.float16),
    ((2, 16), 0.0, 3, False, torch.bfloat16),
    ((7, 5), 0.4, None, False, torch.bfloat16),
]


def input_build(config):
    shape, std, layer_number, cached, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(1932)
    logits = torch.randn(shape, dtype=dtype, generator=generator)
    cached_bias = None
    if cached:
        cached_bias = torch.randn(
            (shape[-1],), dtype=dtype, generator=generator
        )
    return {
        "logits": logits,
        "std": std,
        "layer_number": layer_number,
        "cached_bias": cached_bias,
        "expert_seed": 101,
        "shared_seed": 202,
    }


def compute_ref(logits, std, layer_number, cached_bias, expert_seed, shared_seed):
    return kernelgenbench.baseline.v1_megatron_random_ste_shared(
        logits, std, layer_number, cached_bias, expert_seed, shared_seed
    )


def compute_res(logits, std, layer_number, cached_bias, expert_seed, shared_seed):
    return kernelgenbench.solution.v1_megatron_random_ste_shared(
        logits, std, layer_number, cached_bias, expert_seed, shared_seed
    )


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    ref_value, ref_cache = ref_out
    res_value, res_cache = res_out
    assert isinstance(ref_value, torch.Tensor) and isinstance(res_value, torch.Tensor)
    assert ref_value.shape == res_value.shape
    assert ref_value.dtype == res_value.dtype == dtype
    assert torch.equal(ref_value, res_value)
    if ref_cache is None:
        assert res_cache is None
    else:
        assert isinstance(res_cache, torch.Tensor)
        assert ref_cache.shape == res_cache.shape
        assert ref_cache.dtype == res_cache.dtype
        assert torch.equal(ref_cache, res_cache)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_random_ste_shared")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_random_ste_shared(config):
    shape, std, layer_number, cached, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_logits = res_inputs["logits"].clone()
    before_cached = None if res_inputs["cached_bias"] is None else res_inputs["cached_bias"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Inputs are read-only; only the returned cache slot may be new.
    assert torch.equal(res_inputs["logits"], before_logits)
    if before_cached is not None:
        assert torch.equal(res_inputs["cached_bias"], before_cached)
    if std >= 0:
        assert ref_out[1] is None and res_out[1] is None
    elif layer_number is None:
        assert ref_out[1] is None and res_out[1] is None
    elif cached:
        assert ref_out[1] is ref_inputs["cached_bias"]
        assert res_out[1] is res_inputs["cached_bias"]
    else:
        assert ref_out[1] is not None and res_out[1] is not None

    # Both RandomSTE and RandomSTEShared backward are identities.
    for fn in (compute_ref, compute_res):
        grad_input = inputs["logits"].clone().to(kernelgenbench.device).requires_grad_(True)
        out, _ = fn(
            logits=grad_input,
            std=0.4,
            layer_number=None,
            cached_bias=None,
            expert_seed=101,
            shared_seed=202,
        )
        out.sum().backward()
        assert torch.equal(grad_input.grad, torch.ones_like(grad_input))

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
