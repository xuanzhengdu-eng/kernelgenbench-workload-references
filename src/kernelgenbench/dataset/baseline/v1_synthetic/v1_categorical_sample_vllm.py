# Source scope: vLLM pinned random_sample with no per-request generators.
# Exponential noise is drawn in the probability dtype (in-place probs.div_) or
# in float64 (q.reciprocal_().mul_(probs)); the token is the argmax of the
# ratio.  The float32 path writes the scores back into probs.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L600-L606
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L609-L631
"""vLLM exponential-race categorical sampling."""
import torch


def v1_categorical_sample_vllm(probs, use_fp64_gumbel):
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    q = torch.empty(probs.shape, dtype=dtype, device=probs.device)
    q.exponential_(1.0)
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_().mul_(probs)
    return scores.argmax(dim=-1).view(-1)
