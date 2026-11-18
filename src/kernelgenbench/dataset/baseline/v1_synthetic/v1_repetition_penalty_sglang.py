"""SGLang repetition penalty with the cumulated per-(request, token) scale.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/sampling/penaltylib/repetition_penalty.py#L10-L16

``apply_scaling_penalties`` writes ``logits[:] = torch.where(logits < 0,
logits * scaling_penalties, logits / scaling_penalties)`` and keeps the tensor
identity. ``scaling_penalties`` is a dense ``[num_seqs, vocab]`` float32 tensor;
entries that never appeared in the output are 1.0. Mutates ``logits`` in place
and returns ``None``.
"""
import torch


def v1_repetition_penalty_sglang(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )
    return None
