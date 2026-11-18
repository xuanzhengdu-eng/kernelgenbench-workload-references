"""SGLang sqrt-softplus top-k, native torch path.

Official source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/topk.py#L1157-L1205

Scoring is ``sqrt(softplus(gating_output))``.  Without a correction bias the
input is upcast to FP32, top-k is sorted and ids are int64.  With a bias the
gating dtype is preserved for the scores, ids come from the unsorted
bias-adjusted top-k and weights are gathered from the unbiased scores.
``renormalize`` divides by the FP32 sum plus ``1e-20``.  Inputs are not modified.
"""
import torch
import torch.nn.functional as F


def v1_sqrtsoftplus_expert_topk(hidden_states, gating_output, topk, renormalize, correction_bias):
    if correction_bias is not None:
        scores = F.softplus(gating_output).sqrt()
        scores_for_choice = scores.view(-1, gating_output.shape[-1]) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert hidden_states.shape[0] == gating_output.shape[0], (
            f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
        )
        topk_weights = F.softplus(gating_output.float()).sqrt()
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True, dtype=torch.float32) + 1e-20
        )
    return topk_weights, topk_ids
