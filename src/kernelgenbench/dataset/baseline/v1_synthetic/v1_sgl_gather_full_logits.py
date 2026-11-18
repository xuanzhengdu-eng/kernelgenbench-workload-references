"""SGLang sharded-verify vocabulary all-gather, crop and FP32 cast.

Source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/sharded_verify.py#L693-L706

Upstream signature is ``gather_full_logits(logits, shard, tp_group)`` with
2-D ``logits``.  The reference takes the per-rank local tensors explicitly
and reproduces the same gather order (rank 0 first) plus the unconditional
FP32 result.  The separate DSpark helper ``gather_and_crop_vocab`` preserves
the input dtype and is a different operator item.
"""
import torch


def v1_sgl_gather_full_logits(logits, vocab_size, rank_logits):
    assert logits.ndim == 2
    assert len(rank_logits) >= 1
    assert all(t.shape == logits.shape for t in rank_logits)

    # all_gather_into_tensor writes rank order into out[rank]; movedim(0, 1)
    # then reshapes to [rows, world * local_vocab].
    full = torch.cat([t for t in rank_logits], dim=-1)
    return full[:, :vocab_size].float()
