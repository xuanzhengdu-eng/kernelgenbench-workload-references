"""Sum rank contributions and scatter variable rows into caller output.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/device_communicators/all2all.py#L183-L198
"""
import torch.distributed as dist
def v1_vllm_combine_into_output_(hidden_states,output,sizes,group=None):
    reduced=hidden_states.clone();dist.all_reduce(reduced,group=group)
    rank=dist.get_rank(group);offset=sum(sizes[:rank]);output.copy_(reduced[offset:offset+sizes[rank]])
    # Upstream returns output.movedim(0, dim).contiguous(); for dim=0 this is a
    # view sharing the caller buffer, not the same Python object.
    return output.movedim(0,0).contiguous()
