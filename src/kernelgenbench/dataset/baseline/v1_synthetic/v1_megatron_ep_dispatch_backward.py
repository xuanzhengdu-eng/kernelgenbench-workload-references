# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/fused_a2a.py#L131-L153
# https://github.com/deepseek-ai/DeepEP/blob/34152ae28f80bcc3ee38d7a12cb2ad87cfd4ea72/deep_ep/buffer.py
import torch
import torch.distributed as dist
def source_token_map(ctx):
    rank = dist.get_rank(ctx.group)
    prefixes, _, _, tokens, _, _ = ctx.handle
    ends = prefixes[:, rank].cpu().tolist()
    starts = [0] + ends[:-1]
    return [(owner, token) for owner, lo, hi in zip(range(len(ends)), starts, ends)
            for token in tokens[lo:hi].cpu().tolist()]

@torch.no_grad()
def v1_megatron_ep_dispatch_backward(ctx, grad_output, grad_token_indices, grad_token_probs,
                                      grad_tokens_per_expert, grad_handle):
    """Sum the two gradients received for each source token in the saved route."""
    local = (source_token_map(ctx), grad_output.detach().cpu(), grad_token_probs.detach().float().cpu())
    peers = [None] * dist.get_world_size(ctx.group)
    dist.all_gather_object(peers, local, group=ctx.group)
    count = ctx.handle[4].shape[0]
    gradient = grad_output.new_zeros((count, grad_output.shape[1]))
    weights = grad_token_probs.new_zeros((count, grad_token_probs.shape[1]),dtype=torch.float32)
    for route, values, probs in peers:
        for i, (owner, token) in enumerate(route):
            if owner == dist.get_rank(ctx.group):
                gradient[token].add_(values[i].to(gradient.device))
                weights[token].add_(probs[i].to(weights.device))
    return gradient, None, weights, None, None, None, None
