"""Accuracy and performance for v1_megatron_dsa_indexer_backward (row 152).

Official entry: NVIDIA Megatron-LM ``bwd_fused_indexer_loss_naive`` at commit
266f1c97dca477ca8b92c16087506da2000b0b84, dsa.py#L346-L507.  The test checks
the three analytic gradients produced by the Solution against the reference,
including dtypes, shapes, finiteness, and that no input is modified.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows_152_161_utils import (
    FakePG,
    fresh,
    measure,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (seqlen_q, batch, heads, head_dim, seqlen_k, sparse_loss, dtype)
COMBOS = [
    (4, 2, 2, 4, 4, False, torch.float32),
    (4, 2, 2, 4, 4, True, torch.float32),
    (5, 1, 3, 8, 6, False, torch.bfloat16),
    (6, 2, 2, 8, 6, True, torch.bfloat16),
    (1, 1, 1, 4, 3, False, torch.float32),
    (7, 2, 4, 16, 7, True, torch.float32),
]

TOLERANCE = {
    torch.float32: (2e-5, 2e-6),
    torch.bfloat16: (3e-2, 2e-2),
}


def input_build(config):
    """Build indexer q/weights/k, attention query/key, topk and scalar grad_loss."""
    sq, b, heads, dim, sk, sparse, dtype = config
    rng = torch.Generator(device="cpu").manual_seed(15200 + sq * 13 + heads)
    q = torch.randn((sq, b, heads, dim), generator=rng, dtype=dtype)
    weights = torch.randn((sq, b, heads), generator=rng, dtype=dtype)
    k = torch.randn((sk, b, dim), generator=rng, dtype=dtype)
    query = torch.randn((sq, b, heads, dim), generator=rng, dtype=dtype)
    key = torch.randn((sk, b, heads, dim), generator=rng, dtype=dtype)
    K = min(2, sk)
    rows = torch.arange(sq).view(1, sq, 1)
    base = torch.arange(K).view(1, 1, K)
    topk_indices = torch.remainder(base, rows + 1).expand(b, sq, K).contiguous().long()
    return {
        "q": q,
        "weights": weights,
        "k": k,
        "query": query,
        "key": key,
        "topk_indices": topk_indices,
        "softmax_scale": 0.5,
        "loss_coeff": 0.3,
        "sparse_loss": sparse,
        "grad_loss": torch.tensor(0.7, dtype=torch.float32),
        "pg_collection": FakePG(),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_dsa_indexer_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_dsa_indexer_backward(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    rtol, atol = TOLERANCE[dtype]
    for name, expected, actual in zip(("dq", "dweights", "dk"), ref_out, res_out):
        assert expected.shape == actual.shape, name
        assert expected.dtype == actual.dtype == dtype, name
        assert torch.isfinite(expected).all(), f"{name} reference is not finite"
        assert torch.isfinite(actual).all(), f"{name} solution is not finite"
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_dsa_indexer_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_dsa_indexer_backward(config):
    sq, b, heads, dim, sk, sparse, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["q"].shape == (sq, b, heads, dim)
        assert side["k"].shape == (sk, b, dim)
        assert side["query"].shape == (sq, b, heads, dim)
        assert side["key"].shape == (sk, b, heads, dim)
        assert side["topk_indices"].shape == (b, sq, min(2, sk))
        assert side["grad_loss"].numel() == 1

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The backward reads all inputs and writes only its three outputs.
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            for side in (ref_inputs, res_inputs):
                assert torch.equal(side[name], value.to(kernelgenbench.device)), name
    for output in ref_out + res_out:
        for side in (ref_inputs, res_inputs):
            for value in side.values():
                if isinstance(value, torch.Tensor):
                    assert not torch._C._is_alias_of(output, value)

    ref_time = compute_ref_performance(ref_inputs)
    res_time = compute_res_performance(res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
