"""SGLang speculative accept: move the accepted tree path to each window front.

Reference: v1_sgl_move_accept_kv_ (for each request copy
``num_correct_drafts+1`` accepted slots from the tree path positions to the
leading slots of that request's verify window). Solution: the pinned
``move_accept_tokens_to_target_kvcache`` pipeline, exercising its Triton
``fill_front_out_cache_loc`` / ``fill_accept_out_cache_loc`` helpers and the
real ``MHATokenToKVPool.move_kv_cache`` native copy path.
"""
import statistics

import kernelgenbench
import pytest
import torch

from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# tree paths per request (offsets inside that request's 6-slot verify window),
# layers, heads, head_dim, cache dtype
COMBOS = [
    (((0, 2, 4), (1, 3)), 2, 2, 5, torch.float32),
    (((3, 0, 2, 4), (2,)), 3, 1, 7, torch.bfloat16),
    (((0,), (1, 2, 3)), 1, 3, 4, torch.float16),
    (((4, 3, 2, 1, 0),), 2, 2, 2, torch.float32),
    (((1, 4, 3), (0, 2)), 1, 2, 3, torch.float16),
    (((0, 2, 4), (1, 3), (2, 4)), 3, 2, 2, torch.bfloat16),
]
WINDOW = 6


def input_build(config):
    paths, layers, heads, dim, dtype = config
    batch = len(paths)
    slots = batch * WINDOW + 5
    out_cache_loc = torch.arange(1, batch * WINDOW + 1, dtype=torch.int64)
    accept_index = torch.full((batch, 5), -1, dtype=torch.int64)
    num_correct_drafts = torch.empty(batch, dtype=torch.int32)
    for request, path in enumerate(paths):
        accept_index[request, :len(path)] = torch.tensor([request * WINDOW + offset for offset in path])
        num_correct_drafts[request] = len(path) - 1
    k_cache = torch.arange(layers * slots * heads * dim, dtype=torch.float32).reshape(
        layers, slots, heads, dim).to(dtype)
    v_cache = (k_cache.float() * -0.25 + 7).to(dtype)
    return {"k_cache": k_cache, "v_cache": v_cache, "out_cache_loc": out_cache_loc,
            "accept_index": accept_index, "num_correct_drafts": num_correct_drafts}


def _fresh(inputs):
    device = kernelgenbench.device
    return {name: (value.clone().to(device) if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_move_accept_kv_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_move_accept_kv_(**inputs)


def compare(ref_out, res_out):
    assert ref_out is None and res_out is None, "accept relocation mutates caches and returns None"


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

@label("v1_sgl_move_accept_kv_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_move_accept_kv_(config):
    paths = config[0]
    inputs = input_build(config)
    ref_inputs = _fresh(inputs)
    res_inputs = _fresh(inputs)
    original_k = res_inputs["k_cache"].clone()
    original_v = res_inputs["v_cache"].clone()
    ref_ptrs = (ref_inputs["k_cache"].data_ptr(), ref_inputs["v_cache"].data_ptr())
    res_ptrs = (res_inputs["k_cache"].data_ptr(), res_inputs["v_cache"].data_ptr())

    compare(compute_ref(**ref_inputs), compute_res(**res_inputs))

    assert (res_inputs["k_cache"].data_ptr(), res_inputs["v_cache"].data_ptr()) == res_ptrs
    torch.testing.assert_close(res_inputs["k_cache"], ref_inputs["k_cache"], rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["v_cache"], ref_inputs["v_cache"], rtol=0, atol=0)
    for name in ("out_cache_loc", "accept_index", "num_correct_drafts"):
        torch.testing.assert_close(res_inputs[name], ref_inputs[name], rtol=0, atol=0)

    # Slots that are not accepted destinations keep their pre-call payload.
    targets = []
    for request, path in enumerate(paths):
        for index in range(len(path)):
            targets.append(int(inputs["out_cache_loc"][request * WINDOW + index]))
    untouched = [slot for slot in range(original_k.shape[1]) if slot not in set(targets)]
    torch.testing.assert_close(res_inputs["k_cache"][:, untouched], original_k[:, untouched], rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["v_cache"][:, untouched], original_v[:, untouched], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)


if __name__ == "__main__":
    for combo in COMBOS:
        test_accuracy_v1_sgl_move_accept_kv_(combo)
    print("PASS", len(COMBOS), "move_accept combos")
