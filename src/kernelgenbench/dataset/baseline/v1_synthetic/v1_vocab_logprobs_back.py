"""DeepSpeed selected log-probability backward, single-rank source behavior.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L231-L257

This mirrors ``_VocabParallelLogProbs.backward`` for one vocabulary shard: the
saved global max and shifted log-sum-exp reconstruct ``softmax``, the upstream
gradient is summed per row, a row whose indices are all ignored (or whose max is
-inf) contributes no softmax term, and the one-hot terms are scatter-added.  The
result has the logits' shape and dtype.
"""
import torch

def v1_vocab_logprobs_back(logits, global_max, log_sum_exp, index, ignore_index, grad_output):
    x = logits.reshape(-1, logits.shape[-1]).float()
    idx = index.reshape(x.shape[0], index.shape[-1])
    valid = idx != ignore_index
    safe = idx.clamp(0, x.shape[-1] - 1)
    upstream = grad_output.reshape_as(idx).float() * valid
    p = torch.exp(x - global_max[:, None] - log_sum_exp[:, None])
    out = -p * upstream.sum(dim=-1, keepdim=True)
    zero = (~valid.any(dim=-1)) | (global_max == float("-inf"))
    out = out.masked_fill(zero[:, None], 0.0)
    out.scatter_add_(1, safe, upstream)
    return out.reshape_as(logits).to(logits.dtype)
