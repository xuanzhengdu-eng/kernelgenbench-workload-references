# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L542-L552
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L631-L637
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L119-L164
import torch


def run(shard, graph_id, ds_id, dtype=None):
    out = torch.ops.dc.allgather_param.default(shard, graph_id, ds_id, dtype)
    return torch.ops.dc.wait_allgather.default(out, graph_id, ds_id)
