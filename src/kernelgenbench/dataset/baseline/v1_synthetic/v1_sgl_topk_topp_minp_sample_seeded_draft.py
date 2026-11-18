# Source scope: SGLang pinned deterministic draft sampling.  fast_sample derives
# one exponential variate per (request, position, draft step, vocabulary entry)
# with murmur_hash32 and samples by the exponential-race form of Gumbel-max.
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L114-L195
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/sampling/murmur_hash.py#L1-L128
"""Seeded SGLang draft sampling: q, q(X) and the drafted token index."""
import torch

_DRAFT_MASK = 0xffffffff


def _draft_rot(x, n):
    return ((x << n) | (x >> (32 - n))) & _DRAFT_MASK


def _draft_mix(h, x):
    x = (x * 0xcc9e2d51) & _DRAFT_MASK
    x = _draft_rot(x, 15)
    x = (x * 0x1b873593) & _DRAFT_MASK
    return (_draft_rot(h ^ x, 13) * 5 + 0xe6546b64) & _DRAFT_MASK


def _draft_hash(seed, position, columns):
    state = torch.zeros((seed.numel(), columns.numel()), dtype=torch.int64,
                        device=seed.device)
    key = (seed + 0x44524146545f0000).view(-1, 1)
    for x in (key & _DRAFT_MASK, (key >> 32) & _DRAFT_MASK,
              position.view(-1, 1) & _DRAFT_MASK,
              columns.view(1, -1) & _DRAFT_MASK):
        state = _draft_mix(state, x)
    state = state ^ 16
    state = state ^ (state >> 16)
    state = (state * 0x85ebca6b) & _DRAFT_MASK
    state = state ^ (state >> 13)
    state = (state * 0xc2b2ae35) & _DRAFT_MASK
    return state ^ (state >> 16)


def v1_sgl_topk_topp_minp_sample_seeded_draft(
        probs, temperatures, top_ks, sampling_seed, positions, draft_step=0):
    n = probs.shape[-1]
    q = torch.softmax(probs / temperatures, dim=-1)
    columns = torch.arange(draft_step * n, (draft_step + 1) * n,
                           dtype=torch.int64, device=probs.device)
    hashed = _draft_hash(sampling_seed, positions, columns)
    scores = (hashed.double() + 0.5).div(2.0 ** 32).log().neg().reciprocal() * q
    index = scores.argmax(-1, keepdim=True)
    if top_ks is not None:
        index = torch.where((top_ks <= 1).view(-1, 1),
                            q.argmax(-1, keepdim=True), index)
    return q, q.gather(1, index), index
