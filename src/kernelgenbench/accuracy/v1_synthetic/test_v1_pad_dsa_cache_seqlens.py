"""SGLang DSA cache sequence-length metadata padding for DP padding modes.

Reference: v1_pad_dsa_cache_seqlens (no-op unless CP-interleave or DP token
counts are known; DP pads to max or to this rank's slot, then zero-fills).
Solution: the pinned ``pad_dsa_cache_seqlens`` driven through a real
``ForwardBatch``-shaped object and ``get_parallel().override`` for the DP width.
CP-v2 uses the current rank's physical row count, independently of the DP counts.
"""
import statistics

import kernelgenbench
import pytest
import torch

from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# input rows, global DP token counts or None, CP size/rank, CP interleave flag,
# CP-v2 active flag, per-rank physical rows, DP padding mode, DP slot
COMBOS = [
    (3, None, 1, 0, False, False, None, "slot", 0),
    (3, [2, 5], 1, 0, False, False, None, "max", 0),
    (4, [2, 5], 1, 1, False, False, None, "slot", 0),
    (1, [4, 7], 1, 1, False, False, None, "slot", 1),
    (0, [0, 0], 1, 0, False, False, None, "slot", 0),
    (5, [2, 5], 1, 0, False, False, None, "max", 0),
    (2, [3], 1, 0, False, False, None, "slot", 0),
    (2, None, 2, 0, True, True, [3, 5], "max", 0),
    (4, [20, 30], 2, 1, True, True, [3, 5], "max", 0),
    (6, [20, 30], 2, 1, True, True, [7, 5], "max", 0),
]


def input_build(config):
    rows, global_tokens, cp_size, cp_rank, cp_interleave, cp_active, physical, mode, slot = config
    values = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
    return {
        "dsa_cache_seqlens": values,
        "global_num_tokens": global_tokens,
        "attn_cp_size": cp_size,
        "attn_cp_rank": cp_rank,
        "cp_interleave": cp_interleave,
        "cp_active": cp_active,
        "per_rank_actual_token": physical,
        "dp_mode": mode,
        "dp_slot": slot,
    }


def _fresh(inputs):
    device = kernelgenbench.device
    return {name: (value.clone().to(device) if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_pad_dsa_cache_seqlens(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_pad_dsa_cache_seqlens(**inputs)


def compare(ref_out, res_out):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _bench(function, inputs, repeats=5):
    fresh = _fresh(inputs)
    function(**fresh)
    samples = []
    for _ in range(repeats):
        fresh = _fresh(inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function(**fresh)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)



def compute_ref_performance(**inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(**inputs):
    return _bench(compute_res, inputs)

@label("v1_pad_dsa_cache_seqlens")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_pad_dsa_cache_seqlens(config):
    rows, global_tokens, _, rank, _, active, physical, mode, slot = config
    if active:
        expected_rows = max(rows, physical[rank])
    elif global_tokens is None:
        expected_rows = rows
    elif mode == "max":
        expected_rows = max(rows, max(global_tokens))
    else:
        expected_rows = max(rows, global_tokens[slot])
    inputs = input_build(config)
    ref_inputs = _fresh(inputs)
    res_inputs = _fresh(inputs)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    assert res_out.shape == (expected_rows, 2)
    if expected_rows == rows:
        # No padding required: both sides return the caller's original object.
        assert res_out.data_ptr() == res_inputs["dsa_cache_seqlens"].data_ptr()
        assert ref_out.data_ptr() == ref_inputs["dsa_cache_seqlens"].data_ptr()
    else:
        assert res_out.data_ptr() != res_inputs["dsa_cache_seqlens"].data_ptr()
        assert torch.equal(res_out[rows:], torch.zeros_like(res_out[rows:]))
    torch.testing.assert_close(res_inputs["dsa_cache_seqlens"],
                               ref_inputs["dsa_cache_seqlens"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)


if __name__ == "__main__":
    for combo in COMBOS:
        test_accuracy_v1_pad_dsa_cache_seqlens(combo)
    print("PASS", len(COMBOS), "pad_dsa_cache_seqlens combos")
