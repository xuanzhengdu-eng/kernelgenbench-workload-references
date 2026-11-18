"""vLLM device-communicator ``all_gather``: move the gather axis to dim 0.

vLLM gathers with ``all_gather_into_tensor`` into a dim-0-major buffer, then
reshapes to ``(world, *input)``, moves dim 0 to the requested axis and flattens the
two adjacent axes.  A negative ``dim`` is normalized first, so the returned
layout can be non-contiguous (the reshape view is returned as-is).
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/device_communicators/base_device_communicator.py#L269-L292
"""
import torch
import torch.distributed as dist

def v1_vllm_all_gather(input_,group,dim=-1):
    world=group.size()
    if dim<0:dim+=input_.dim()
    size=input_.size()
    output=torch.empty((size[0]*world,)+size[1:],dtype=input_.dtype,device=input_.device)
    dist.all_gather_into_tensor(output,input_,group=group)
    output=output.reshape((world,)+size).movedim(0,dim)
    return output.reshape(size[:dim]+(world*size[dim],)+size[dim+1:])
