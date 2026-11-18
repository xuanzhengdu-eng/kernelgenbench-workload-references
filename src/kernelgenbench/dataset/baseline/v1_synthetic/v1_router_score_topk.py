# Source-checked reference for KernelGenBench row 70 (SGLang branch).
"""SGLang torch-native router top-k with softmax/sigmoid/sqrtsoftplus scoring.

Without correction bias the scores are computed in fp32 and the selection is
sorted (torch.topk default); with bias the score dtype follows the input and the
ordering is unspecified, while renormalising divides by an fp32 sum and promotes
the returned weights to fp32.  ids are int64.  Ties are not covered.

Official source (SGLang fixed commit 62384de1):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/topk.py#L1157-L1204
"""
import torch
import torch.nn.functional as F

_RENORMALIZE_SUM_EPSILON = 1e-20


def v1_router_score_topk(hidden_states, gating_output, topk, renormalize,
                         correction_bias=None, scoring_func="softmax"):
    def scoring(gating):
        if scoring_func == "softmax":
            return gating.softmax(dim=-1)
        if scoring_func == "sigmoid":
            return gating.sigmoid()
        if scoring_func == "sqrtsoftplus":
            return F.softplus(gating).sqrt()
        raise ValueError(f"Invalid scoring function: {scoring_func}")

    if correction_bias is not None:
        n_routed_experts = gating_output.shape[-1]
        scores = scoring(gating_output)
        scores_for_choice = scores.view(-1, n_routed_experts) + correction_bias.unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
    else:
        assert hidden_states.shape[0] == gating_output.shape[0], (
            "Number of tokens mismatch between hidden_states and gating_output"
        )
        topk_weights = scoring(gating_output.float())
        topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)

    if renormalize:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True, dtype=torch.float32)
            + _RENORMALIZE_SUM_EPSILON
        )
    return topk_weights, topk_ids
