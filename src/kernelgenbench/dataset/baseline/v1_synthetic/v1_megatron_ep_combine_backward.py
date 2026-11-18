# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/fused_a2a.py#L192-L209
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
def v1_megatron_ep_combine_backward(ctx, grad_output, previous_event=None):
    """Replay the saved intranode routes and return the five backward slots."""
    peers = [None] * dist.get_world_size(ctx.group)
    dist.all_gather_object(peers, grad_output.detach().cpu(), group=ctx.group)
    route = source_token_map(ctx)
    if route:
        gradient = torch.stack([peers[owner][token] for owner, token in route]).to(grad_output.device)
    else:
        gradient = grad_output.new_empty((0, grad_output.shape[1]))
    return gradient, None, None, None, None
