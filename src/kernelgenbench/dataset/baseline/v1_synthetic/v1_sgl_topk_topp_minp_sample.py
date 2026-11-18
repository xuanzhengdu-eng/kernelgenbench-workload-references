# Source scope: SGLang pinned top-k/top-p/min-p sampling from probabilities,
# unseeded torch.multinomial path.  The official function also implements the
# seeded multinomial variant, but that contract is carried by the separate
# v1_sgl_topk_topp_minp_sample_seeded_draft reference.
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/sampler.py#L786-L859
"""SGLang top-k/top-p/min-p filtered sampling; returns IDs, filtered probs,
token permutation and the selected weight."""
import torch


def v1_sgl_topk_topp_minp_sample(probs, top_ks, top_ps, min_ps,
                                 need_min_p_sampling):
    sorted_prob, ids = torch.sort(probs, dim=-1, descending=True)
    cumsum = sorted_prob.cumsum(-1)
    rank = torch.arange(probs.shape[-1], device=probs.device)
    sorted_prob = torch.where(rank[None, :] < top_ks[:, None], sorted_prob, 0.0)
    sorted_prob = torch.where(cumsum - sorted_prob <= top_ps[:, None],
                              sorted_prob, 0.0)
    if need_min_p_sampling:
        sorted_prob = torch.where(
            sorted_prob >= sorted_prob[:, :1] * min_ps[:, None],
            sorted_prob, 0.0)
    selected = torch.multinomial(sorted_prob, 1)
    weights = sorted_prob.gather(1, selected).view(-1)
    ids = ids.to(torch.int32)
    tokens = ids.gather(1, selected).view(-1)
    return tokens, sorted_prob, ids, weights
