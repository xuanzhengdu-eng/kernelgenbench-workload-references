# Source scope: SGLang pinned sampling_from_probs_torch, unseeded branch.  With
# no sampling seed it either uses the gumbel-max exponential race (when
# SGLANG_OPT_USE_GUMBEL_SAMPLE is enabled) or torch.multinomial, then returns
# int32 token IDs.  Probabilities are not modified.
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/sampler.py#L979-L1005
"""SGLang unseeded categorical sampling from a float32 probability matrix."""
import torch


def v1_categorical_sample_sglang(probs):
    return torch.multinomial(probs, 1).view(-1).to(torch.int32)
