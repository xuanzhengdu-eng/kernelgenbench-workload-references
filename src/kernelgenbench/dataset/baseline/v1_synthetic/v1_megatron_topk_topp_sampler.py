# Source scope: Megatron-LM pinned TextGenerationController._torch_sampling_func.
# Greedy rows (top_k == 1) return argmax; otherwise temperature scaling, top-k
# or top-p masking, softmax and torch.multinomial with the controller RNG, then
# optional vocabulary clamping.  Top-k and top-p cannot both be active.
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/text_generation_controllers/text_generation_controller.py#L272-L359
"""Megatron-LM torch sampling of last-token logits."""
import torch


def v1_megatron_topk_topp_sampler(probs, temperature, top_k, top_p,
                                  generator, vocab_size=None):
    assert isinstance(top_p, float) and isinstance(top_k, int)
    assert not (top_k > 0 and top_p > 0.0)
    assert top_p <= 1.0
    if top_k == 1:
        return torch.argmax(probs, dim=-1)
    logits = probs.clone()
    if temperature != 1.0:
        logits.div_(temperature)
    if top_k > 1:
        assert top_k <= logits.shape[-1] and (not vocab_size or top_k < vocab_size)
        logits.masked_fill_(
            logits < logits.topk(top_k, dim=-1).values[:, -1, None],
            float("-inf"))
    elif top_p > 0.0:
        ordered, index = logits.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(-1).cumsum(-1)
        mask = cumulative > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        mask = mask.scatter(1, index, mask)
        logits.masked_fill_(mask, float("-inf"))
    sample = torch.multinomial(logits.softmax(-1), 1,
                               generator=generator).view(-1)
    return sample.clamp(0, vocab_size - 1) if vocab_size else sample
