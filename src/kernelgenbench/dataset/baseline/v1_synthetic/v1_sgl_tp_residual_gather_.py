# Source (limited scope): https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layer_boundary/residual/add_norm.py#L272-L286
# Scope: PlainAdd.update modifies hidden_states in place.
# Source (limited scope): https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layer_boundary/ops.py#L369-L388
# Scope: update_attn_tp_gather_output source; reference fixes PlainAdd and equal-size explicit Gloo-like gather.
"""In-place local residual add followed by equal-size attention TP gather."""
import torch,torch.distributed as dist

def v1_sgl_tp_residual_gather_(hidden_states,residual,group=None):
    hidden_states.add_(residual)
    world=dist.get_world_size(group);buf=[torch.empty_like(hidden_states) for _ in range(world)]
    dist.all_gather(buf,hidden_states.contiguous(),group=group)
    return torch.cat(buf,dim=0),None
