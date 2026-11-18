"""DeepSpeed ``_AllGatherVariableRows.forward``: padded gather of unequal rows.

Each rank pads its rows to ``max_rows`` with zeros, all-gathers the equal-shaped
padded tensors, then trims every peer back to its own ``counts`` entry and
concatenates in rank order.  ``max_rows == 0`` returns a new empty tensor of the
same trailing shape without communicating.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_tp_dispatch.py#L283-L340
"""
import torch
import torch.distributed as dist

def v1_ds_gather_variable_rows(input_,group,counts,max_rows):
    if max_rows==0:return input_.new_empty((0,*input_.shape[1:]))
    padded=input_.new_zeros((max_rows,*input_.shape[1:]))
    padded[:input_.shape[0]].copy_(input_)
    gathered=[torch.zeros_like(padded) for _ in counts]
    dist.all_gather(gathered,padded,group=group)
    return torch.cat([chunk[:count] for chunk,count in zip(gathered,counts)],dim=0)
