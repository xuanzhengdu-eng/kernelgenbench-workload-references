"""Accuracy and timing for v1_vllm_reorg_kvcache (vLLM MLA chunked context).

Official entry: vllm.model_executor.layers.attention.mla_attention.reorg_kvcache
Commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5, mla_attention.py#L2801-L2867.

Inputs are a rank-major all-gathered latent KV cache ``[ranks * toks, 5]`` and
RoPE cache ``[ranks * toks, 3]`` (float32 or bfloat16), the per-request padded
chunk caps, the per-rank context lengths, the per-request local start offsets,
and the two consistency scalars ``sum_seq_len``/``max_seq_len``.  The output is
request-major with ``sum_seq_len`` rows; the reference asserts both scalars.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows522_531_utils import fresh, measure, timings
from sandbox.verifier.test_parametrize import label, parametrize


# (chunk caps, per-rank contexts, local starts, dtype)
COMBOS = [
    ((4, 2), ((6, 2), (3, 1)), (0, 0), torch.float32),
    ((3, 2, 4), ((9, 5, 2), (0, 0, 3), (1, 2, 1)), (3, 2, 1), torch.bfloat16),
    ((2, 3), ((0, 4), (5, 1)), (1, 2), torch.float32),
    ((3,), ((4,),), (1,), torch.bfloat16),
    ((5, 5), ((5, 5), (5, 5)), (0, 0), torch.bfloat16),
    ((1, 1), ((2, 1), (1, 2)), (0, 1), torch.float32),
    ((6,), ((6, 6, 6),), (2,), torch.bfloat16),
]


def input_build(config):
    caps, contexts, starts, dtype = config
    ranks = len(contexts[0])
    toks = sum(caps)
    lengths = [
        sum(min(max(0, length - start), cap) for length in lens)
        for cap, lens, start in zip(caps, contexts, starts)
    ]
    kv = torch.arange(ranks * toks * 5, dtype=torch.float32).to(dtype).view(ranks * toks, 5)
    rope = (torch.arange(ranks * toks * 3, dtype=torch.float32) + 100).to(dtype).view(ranks * toks, 3)
    return {
        "allgatered_kv_c_normed": kv,
        "allgatered_k_pe": rope,
        "padded_local_chunk_seq_lens_lst": list(caps),
        "local_context_lens_allranks": [list(row) for row in contexts],
        "local_starts": list(starts),
        "sum_seq_len": sum(lengths),
        "max_seq_len": max(lengths),
        "toks": toks,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_reorg_kvcache(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_reorg_kvcache(**inputs)


def expected_rows(config):
    caps, contexts, starts, _ = config
    return sum(
        sum(min(max(0, length - start), cap) for length in lens)
        for cap, lens, start in zip(caps, contexts, starts)
    )


def compare(ref_out, res_out, config):
    dtype = config[3]
    rows = expected_rows(config)
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    for ref_side, res_side, width in ((ref_out[0], res_out[0], 5), (ref_out[1], res_out[1], 3)):
        assert ref_side.shape == res_side.shape == (rows, width)
        assert ref_side.dtype == res_side.dtype == dtype
        torch.testing.assert_close(res_side, ref_side, rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_vllm_reorg_kvcache")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_reorg_kvcache(config):
    caps, contexts, starts, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["allgatered_kv_c_normed"].shape == (len(contexts[0]) * sum(caps), 5)
        assert side["allgatered_kv_c_normed"].dtype == dtype
        assert side["local_starts"] == list(starts)

    before = {
        name: value.clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # Padded (non-copied) rows must be dropped and the surviving rows gathered
    # rank-major then request-major.  The index list is spelled out here so the
    # assertion does not depend on the reference's own gather.
    ranks = len(contexts[0])
    expected_index = []
    start = 0
    for cap, lens, local_start in zip(caps, contexts, starts):
        for rank, length in enumerate(lens):
            copied = min(max(0, length - local_start), cap)
            expected_index.extend(range(rank * sum(caps) + start, rank * sum(caps) + start + copied))
        start += cap
    index = torch.tensor(expected_index, device=kernelgenbench.device, dtype=torch.long)
    assert len(expected_index) == expected_rows(config)
    for out in (ref_out[0], res_out[0]):
        torch.testing.assert_close(out, res_inputs["allgatered_kv_c_normed"].index_select(0, index), rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
        assert not torch._C._is_alias_of(res_out[0], res_inputs[name])

    ref_ms = compute_ref_performance(**fresh(inputs, kernelgenbench.device))
    res_ms = compute_res_performance(**fresh(inputs, kernelgenbench.device))
    return timings(ref_ms, res_ms)
