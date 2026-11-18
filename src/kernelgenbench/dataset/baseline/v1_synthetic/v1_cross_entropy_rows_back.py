"""DeepSpeed per-row cross entropy backward from the saved log-sum-exp.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L181-L195

Contract:
  * ``logits`` is ``[rows, vocab]``; ``log_sum_exp`` is the fp32
    ``logsumexp(logits.float(), dim=-1)`` saved by the forward pass.
  * ``grad_output`` is ``[rows]`` (one incoming gradient per row) and is cast to
    fp32; row i is scaled by ``grad_output[i]``.
  * ``prob = exp(logits.float() - log_sum_exp[:, None])``; one is subtracted from
    the target class per row, ignored rows contribute zero gradient, and the
    result is cast back to the logits dtype. No label smoothing term exists
    upstream.
  * Returns a fresh tensor of the logits shape; inputs are not modified.
"""
import torch


def v1_cross_entropy_rows_back(logits, target, ignore_index, log_sum_exp, grad_output):
    valid = target != ignore_index
    safe = target.masked_fill(~valid, 0)
    prob = torch.exp(logits.float() - log_sum_exp[:, None])
    prob.scatter_add_(1, safe[:, None], -torch.ones_like(prob[:, :1]))
    return (prob * torch.where(valid, grad_output.float(), 0.0)[:, None]).to(logits.dtype)
