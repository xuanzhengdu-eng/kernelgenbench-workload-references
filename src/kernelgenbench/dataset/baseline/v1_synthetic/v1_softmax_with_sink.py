# Fixed official source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L154-L176
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# License: tests/megatron_official/upstream/LICENSE
# Derived from NVIDIA/Megatron-LM; see docs/megatron-official-softmax-one.md.
"""Per-head sink-logit softmax, existing OP-c4e1011d8375."""
import torch


def softmax_one(x, sink):
    """Softmax over [x, sink] per head, then remove the sink output.

    x: [B,H,Q,K]; sink: Tensor with H elements on x.device. Output uses torch.cat
    promotion rules and is not normalized to sum one over the returned K entries.
    """
    offset = sink.reshape(1, -1, 1, 1).expand(x.size(0), -1, x.size(2), -1)
    logits = torch.cat((x, offset), dim=-1)
    return torch.softmax(logits, dim=-1)[..., :-1]


def v1_softmax_with_sink(x, sink):
    return softmax_one(x, sink)
