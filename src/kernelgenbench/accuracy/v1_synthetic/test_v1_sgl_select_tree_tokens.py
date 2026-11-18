"""Accuracy and performance for v1_sgl_select_tree_tokens (SGLang EAGLE).

The official entry is ``select_top_k_tokens`` in
``sglang.srt.speculative.spec_utils``.  Step 0 repeats each hidden row ``topk``
times and returns the flattened ids plus root tree metadata; later steps expand
parent scores, top-k select, gather ids/hidden rows and return updated metadata.
The test checks values, structure, in-place/aliasing relationships and the
empty-hidden early exit.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (step, batch, topk, hidden mode).  Step 0 input topk_p is [B,K];
# later steps use topk_p [B*K,K] and scores [B,K].  hidden is [B*K,7]
# for steps > 0, [B,7] for step 0, or None/empty to cover both exits.
COMBOS = [
    (0, 2, 3, "present"),
    (0, 1, 2, "none"),
    (1, 2, 3, "present"),
    (3, 2, 2, "empty"),
    (2, 3, 1, "none"),
]


def input_build(config):
    step, batch, topk, mode = config
    rng = torch.Generator(device="cpu").manual_seed(step * 100 + batch * 10 + topk)
    rows = batch if step == 0 else batch * topk
    topk_p = torch.rand((rows, topk), generator=rng) + 0.1
    topk_index = torch.arange(topk_p.numel(), dtype=torch.int64).reshape_as(topk_p) + 13
    if mode == "none":
        hidden = None
    elif mode == "empty":
        hidden = torch.empty((0, 7))
    else:
        hidden = torch.randn((rows, 7), generator=rng)
    scores = None if step == 0 else torch.rand((batch, topk), generator=rng) + 0.3
    return {
        "i": step,
        "topk_p": topk_p,
        "topk_index": topk_index,
        "hidden_states": hidden,
        "scores": scores,
        "topk": topk,
    }


def _to_device(inputs):
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(kernelgenbench.device) if isinstance(value, torch.Tensor) else value
    return out


def compute_ref(i, topk_p, topk_index, hidden_states, scores, topk):
    return kernelgenbench.baseline.v1_sgl_select_tree_tokens(
        i, topk_p, topk_index, hidden_states, scores, topk
    )


def compute_res(i, topk_p, topk_index, hidden_states, scores, topk):
    return kernelgenbench.solution.v1_sgl_select_tree_tokens(
        i, topk_p, topk_index, hidden_states, scores, topk
    )


def _compare_value(ref_value, res_value):
    if ref_value is None or res_value is None:
        assert ref_value is None and res_value is None
        return
    assert isinstance(res_value, torch.Tensor)
    assert res_value.shape == ref_value.shape
    assert res_value.dtype == ref_value.dtype
    assert res_value.device == ref_value.device
    torch.testing.assert_close(res_value, ref_value, rtol=0, atol=0)


def compare(ref_out, res_out):
    assert isinstance(ref_out, (tuple, list)) and isinstance(res_out, (tuple, list))
    assert len(ref_out) == len(res_out) == 4
    ids_ref, hidden_ref, probs_ref, tree_ref = ref_out
    ids_res, hidden_res, probs_res, tree_res = res_out
    _compare_value(ids_ref, ids_res)
    _compare_value(hidden_ref, hidden_res)
    _compare_value(probs_ref, probs_res)
    assert isinstance(tree_ref, tuple) and isinstance(tree_res, tuple)
    assert len(tree_ref) == len(tree_res) == 3
    for a, b in zip(tree_ref, tree_res):
        _compare_value(a, b)


def compute_ref_performance(i, topk_p, topk_index, hidden_states, scores, topk):
    return _bench(
        lambda: compute_ref(i, topk_p, topk_index, hidden_states, scores, topk)
    )


def compute_res_performance(i, topk_p, topk_index, hidden_states, scores, topk):
    return _bench(
        lambda: compute_res(i, topk_p, topk_index, hidden_states, scores, topk)
    )


def _bench(fn):
    import triton.testing

    return triton.testing.do_bench(fn, warmup=25, rep=100)


@label("v1_sgl_select_tree_tokens")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_select_tree_tokens(config):
    step, batch, topk, mode = config
    inputs = input_build(config)
    rows = batch if step == 0 else batch * topk
    assert inputs["topk_p"].shape == (rows, topk)
    assert inputs["topk_index"].shape == (rows, topk)
    assert inputs["topk_index"].dtype == torch.int64
    assert inputs["scores"] is None or inputs["scores"].shape == (batch, topk)
    if step == 0:
        assert inputs["scores"] is None
    before = {
        key: value.clone() for key, value in inputs.items() if isinstance(value, torch.Tensor)
    }

    ref_inputs = _to_device(inputs)
    res_inputs = _to_device(inputs)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    # Step 0 aliases the input routing tensors; later steps alias only the
    # flattened index buffer.  An empty hidden tensor is returned untouched.
    if step == 0:
        assert res_out[2] is res_inputs["topk_p"]
        assert torch._C._is_alias_of(res_out[3][0], res_inputs["topk_p"])
        assert res_out[3][1] is res_inputs["topk_index"]
        assert torch._C._is_alias_of(res_out[0], res_inputs["topk_index"])
    else:
        assert torch._C._is_alias_of(res_out[3][1], res_inputs["topk_index"])
    if mode == "empty":
        assert res_out[1] is res_inputs["hidden_states"]
    elif mode == "none":
        assert res_out[1] is None
    else:
        assert res_out[1] is not None

    for key, value in before.items():
        torch.testing.assert_close(res_inputs[key].cpu(), value, rtol=0, atol=0)
        torch.testing.assert_close(ref_inputs[key].cpu(), value, rtol=0, atol=0)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
