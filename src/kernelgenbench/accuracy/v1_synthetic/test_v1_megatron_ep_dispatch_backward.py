# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/fused_a2a.py#L131-L153
# https://github.com/deepseek-ai/DeepEP/blob/34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72/deep_ep/buffer.py
"""Exercise saved DeepEP intranode routing handles on two distinct H800s."""
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# Cross-destination top-k, asynchronous stream completion, per-rank uneven input.
COMBOS = [(1, False, False), (2, False, True), (2, True, False)]


def input_build(config):
    from megatron.core.transformer.moe.fused_a2a import FusedDispatch
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    topk, async_finish, uneven = config
    count = 4 + int(uneven and rank == 1)
    tokens = torch.arange(count, device="cuda")
    owner_values = (tokens + 10 * rank).to(torch.bfloat16)
    hidden = owner_values[:, None].expand(count, 128).contiguous()
    first = (tokens % 2).to(torch.int64)
    routes = first[:, None] if topk == 1 else torch.stack((first, 1 - first), dim=1)
    probs = torch.ones((count, topk), device="cuda")
    ctx = SimpleNamespace()
    received, _, _, _, _ = FusedDispatch.forward(ctx, hidden, routes, probs, 2, dist.group.WORLD,
                                                 async_finish, async_finish)
    # Each destination contributes a distinct gradient to the owner of each token.
    grad = received + rank + 1
    grad_probs = torch.full((len(received), topk), rank + 1, device="cuda", dtype=torch.bfloat16)
    if topk == 1:
        expected = hidden + first[:, None].to(hidden.dtype) + 1
        expected_probs = (first[:, None] + 1).float()
    else:
        expected = 2 * hidden + 3
        expected_probs = torch.full((count, topk), 3., device="cuda")
    return dict(ctx=ctx, grad_output=grad, grad_token_indices=None, grad_token_probs=grad_probs,
                grad_tokens_per_expert=None, grad_handle=None), (expected, expected_probs)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_ep_dispatch_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_ep_dispatch_backward(**inputs)


def compare(out, expected):
    assert isinstance(out, tuple) and len(out) == 7 and all(out[i] is None for i in (1, 3, 4, 5, 6))
    for actual, wanted in zip((out[0], out[2]), expected):
        assert actual.shape == wanted.shape and actual.dtype == wanted.dtype
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@label("v1_megatron_ep_dispatch_backward")
@parametrize("config", COMBOS)
def test_accuracy_v1_megatron_ep_dispatch_backward(config):
    inputs, expected = input_build(config)
    saved = inputs["grad_output"].clone()
    times = []
    for implementation in (compute_ref, compute_res):
        dist.barrier()
        start = time.perf_counter()
        out = implementation(**inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        compare(out, expected)
        torch.testing.assert_close(inputs["grad_output"], saved, rtol=0, atol=0)
    return CustomBenchmarkResult(ref_time=times[0], res_time=times[1], speedup=times[0] / times[1])
