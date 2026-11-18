# Source scope: SGLang pinned Leviathan draft proposal.  sample_draft_proposal
# returns q = softmax(logits / T), q(X) and X, replacing X by the row argmax for
# greedy rows (top_k <= 1).  The unseeded draw uses fast_sample.
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L214-L264
"""SGLang draft proposal: q, q(X), X for one speculative draft step."""
import torch


def v1_sgl_topk_topp_minp_sample_draft(next_token_logits, temperatures,
                                       top_ks=None):
    probs = torch.softmax(next_token_logits / temperatures, dim=-1)
    index = torch.multinomial(probs, 1)
    if top_ks is not None:
        greedy = (top_ks <= 1).view(-1, 1)
        index = torch.where(greedy, probs.argmax(dim=-1, keepdim=True), index)
    return probs, probs.gather(1, index), index
