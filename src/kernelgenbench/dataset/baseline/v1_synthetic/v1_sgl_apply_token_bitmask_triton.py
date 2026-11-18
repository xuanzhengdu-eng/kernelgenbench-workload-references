"""SGLang Triton fallback for applying packed int32 grammar bitmasks.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/grammar/bitmask_ops.py#L84-L141

Bit ``b`` of ``bitmask[row, b // 32]`` is 1 for an allowed token and 0 for a
masked token.  Only the first ``min(vocab, words * 32)`` columns are touched;
the rest of a wider logits row is left alone.  ``indices`` selects rows of the
bitmask/logits pair; ``None`` applies row ``i`` to row ``i``.  The operation is
in-place and returns ``None``.  This is a transcription of the Triton body, not
a call into it; the test binds the pinned Triton function as the Solution.
"""
import torch


def v1_sgl_apply_token_bitmask_triton(logits, bitmask, indices=None):
    assert bitmask.dtype == torch.int32
    x = logits.view(1, -1) if logits.ndim == 1 else logits
    m = bitmask.view(1, -1) if bitmask.ndim == 1 else bitmask
    width = min(x.shape[1], m.shape[1] * 32)
    cols = torch.arange(width, device=x.device, dtype=torch.int64)
    words = torch.div(cols, 32, rounding_mode="floor")
    bits = cols % 32
    rows = range(x.shape[0]) if indices is None else (int(i) for i in indices)
    for row in rows:
        allowed = ((m[row, words].to(torch.int64) >> bits) & 1).bool()
        x[row, :width].masked_fill_(~allowed, float("-inf"))
    return None
