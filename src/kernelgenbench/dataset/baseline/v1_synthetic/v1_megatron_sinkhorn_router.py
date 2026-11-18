# Source-checked reference for KernelGenBench row 71.
"""Megatron TopKRouter.sinkhorn_load_balancing (training and eval).

Training routes on the sinkhorn-scaled cost in fp32 no_grad; eval routes on the
activation scores directly.  topk=1 uses sigmoid, topk>1 uses fp32 softmax cast
back to the logits dtype.  moe_aux_loss_coeff must be 0.

Official sources (NVIDIA fixed commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/router.py#L230-L262
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L181-L203
"""
import torch


def v1_megatron_sinkhorn_router(logits, topk, training):
    if topk == 1:
        activation = torch.sigmoid(logits)
    else:
        activation = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
    if training:
        with torch.no_grad():
            cost = logits.float().exp()
            d0 = torch.ones(cost.shape[0], device=cost.device)
            d1 = torch.ones(cost.shape[1], device=cost.device)
            error = 1e9
            d1_old = d1
            while error > 1e-4:
                d0 = (1 / d0.numel()) * 1 / ((d1 * cost).sum(1) + 1e-8)
                new = (1 / d1.numel()) * 1 / ((d0.unsqueeze(1) * cost).sum(0) + 1e-8)
                error = (d1_old - new).abs().mean()
                d1_old = new
                d1 = new
            chosen = (d1 * cost * d0.unsqueeze(1)).topk(topk, dim=1).indices
    else:
        chosen = activation.topk(topk, dim=1).indices
    routing = torch.zeros_like(logits, dtype=torch.int32).scatter_(1, chosen, 1).bool()
    return activation * routing, routing
