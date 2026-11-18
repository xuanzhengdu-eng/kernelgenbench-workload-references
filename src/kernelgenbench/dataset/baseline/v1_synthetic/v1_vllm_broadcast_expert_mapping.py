"""vLLM elastic EP expert mapping: broadcast 2D shape then values.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/elastic_ep/elastic_execute.py#L125-L151
Gloo adapter uses one process group for metadata and data; source uses TCPStore
for host shape and device communicator or TCPStore for data.
"""
import torch
import torch.distributed as dist

def v1_vllm_broadcast_expert_mapping(physical_to_logical,group,device,src_rank=0):
    rank=dist.get_rank(group);src=dist.get_global_rank(group,src_rank)
    if rank==src_rank:
        assert physical_to_logical is not None and physical_to_logical.dtype==torch.int64 and physical_to_logical.ndim==2
        shape=torch.tensor(physical_to_logical.shape,dtype=torch.int64,device='cpu')
    else:shape=torch.empty(2,dtype=torch.int64,device='cpu')
    dist.broadcast(shape,src,group=group)
    if rank!=src_rank:physical_to_logical=torch.empty(tuple(shape.tolist()),dtype=torch.int64,device=device)
    dist.broadcast(physical_to_logical,src,group=group)
    return physical_to_logical
