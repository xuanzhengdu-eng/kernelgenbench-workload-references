"""SGLang row logsumexp with stable top-k, finite-input branch.

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/logsumexp.py#L247-L298
Kernel with the same (max, log_sum, top_vals, top_idx) contract:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/logsumexp.py#L112-L247

Contract (finite inputs):
  * x is 2-D with dtype fp16/bf16/fp32 and 1 <= k <= min(8, columns).
  * Returns (maximum, log_sum, values, indices):
      maximum  float32 [rows]   = row max of x
      log_sum  float32 [rows]   = log(sum(exp(x - maximum))), i.e. the
                                  normaliser minus the max (NOT the absolute
                                  logsumexp)
      values   float32 [rows,k] = raw top-k values sorted descending
      indices  int64   [rows,k] = top-k column indices; value ties prefer the
                                  lower index (stable descending sort)
  * The input is not modified; an empty batch returns empty tensors.

Upstream also guards +/-inf rows inside ``_accumulate_block_lse``. The legacy
reference only implemented the finite branch and is kept scoped to finite
inputs here; the non-finite guards are source-reviewed but not extracted.
"""
import torch


def v1_row_logsumexp_topk(x, k):
    assert x.ndim == 2 and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert 1 <= k <= min(8, x.shape[1])
    xf = x.float()
    maximum = xf.amax(dim=1)
    log_sum = torch.log(torch.exp(xf - maximum[:, None]).sum(dim=1))
    indices = torch.argsort(xf, dim=1, descending=True, stable=True)[:, :k]
    values = torch.gather(xf, 1, indices)
    return maximum, log_sum, values, indices
