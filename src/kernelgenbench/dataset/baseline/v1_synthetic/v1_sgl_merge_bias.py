"""SGLang sampling-batch bias merge (``merge_bias_tensor``).

Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/sampling/sampling_batch_info.py#L719-L755

Reached when two ``SamplingBatchInfo`` batches merge: a side without a bias
tensor is materialised as ``bs`` rows of ``default`` with the other side's
trailing shape and dtype, then both halves are concatenated on the batch axis.
Both sides missing returns ``None``; both present concatenates them unchanged.
The result is always a fresh tensor.
"""
import torch


def v1_sgl_merge_bias(lhs, rhs, bs1, bs2, device, default):
    if lhs is None and rhs is None:
        return None
    if lhs is not None and rhs is not None:
        return torch.cat([lhs, rhs], dim=0)
    template = lhs if lhs is not None else rhs
    shape, dtype = template.shape[1:], template.dtype
    if lhs is None:
        lhs = torch.empty((bs1, *shape), device=device, dtype=dtype).fill_(default)
    if rhs is None:
        rhs = torch.empty((bs2, *shape), device=device, dtype=dtype).fill_(default)
    return torch.cat([lhs, rhs], dim=0)
