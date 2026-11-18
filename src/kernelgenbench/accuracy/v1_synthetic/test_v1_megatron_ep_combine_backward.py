# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/fused_a2a.py#L192-L209
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
    return dict(ctx=ctx, grad_output=hidden), received


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_ep_combine_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_ep_combine_backward(**inputs)


def compare(out, expected):
    assert isinstance(out, tuple) and len(out) == 5 and all(x is None for x in out[1:])
    assert out[0].dtype == expected.dtype and out[0].shape == expected.shape
    torch.testing.assert_close(out[0], expected, rtol=0, atol=0)


@label("v1_megatron_ep_combine_backward")
@parametrize("config", COMBOS)
def test_accuracy_v1_megatron_ep_combine_backward(config):
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
