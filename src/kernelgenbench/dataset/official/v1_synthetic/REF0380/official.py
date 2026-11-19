# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_tp_dispatch.py#L283-L340
from deepspeed.moe.ep_tp_dispatch import _AllGatherVariableRows


def official_gather_variable_rows(tensor, group, counts, max_rows):

    return _AllGatherVariableRows.apply(tensor, group, tuple(counts), int(max_rows))
