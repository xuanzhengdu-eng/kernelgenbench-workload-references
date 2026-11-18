# Source scope: vLLM pinned TopKTopPSampler.forward_native: apply_top_k_top_p
# masks logits in place, then random_sample draws with exponential noise and
# argmax, optionally in float64.  The per-request generator overwrite and the
# logprobs return-mode strings are part of this contract.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L295-L316
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L504-L524
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L527-L568
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_sampler.py#L609-L631
"""vLLM native top-k/top-p filtering plus exponential-race sampling."""
import torch


def v1_vllm_topk_topp_sampler(logits, top_ks, top_ps, generators,
                              logprobs_mode="raw", use_fp64_gumbel=False):
    if top_ks is not None or top_ps is not None:
        ordered, index = logits.sort(-1, descending=False)
        if top_ks is not None:
            threshold = ordered.gather(
                1, (ordered.shape[1] - top_ks.long())[:, None])
            ordered.masked_fill_(ordered < threshold, float("-inf"))
        if top_ps is not None:
            cumulative = ordered.softmax(-1).cumsum(-1)
            mask = cumulative <= 1 - top_ps[:, None]
            mask[:, -1] = False
            ordered.masked_fill_(mask, float("-inf"))
        logits.scatter_(1, index, ordered)
    if logprobs_mode == "processed_logits":
        extra = logits
    elif logprobs_mode == "processed_logprobs":
        extra = logits.log_softmax(-1, dtype=torch.float32)
    else:
        extra = None
    scores = logits.softmax(-1, dtype=torch.float32)
    generators = generators or {}
    q = torch.empty(scores.shape, device=scores.device,
                    dtype=torch.float64 if use_fp64_gumbel else scores.dtype)
    if len(generators) != scores.shape[0]:
        q.exponential_()
    for row, generator in generators.items():
        q[row].exponential_(generator=generator)
    return (scores / q).argmax(-1).view(-1), extra
