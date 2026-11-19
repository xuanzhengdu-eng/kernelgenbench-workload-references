# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/deepcompile.cpp#L105-L135
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z2.cpp#L33-L86
import torch

reduce_grad = torch.ops.dc.reduce_grad
end_backward = torch.ops.dc.end_backward


def run(grad, graph_id, ds_id, deps=None, release_reduce_buckets=True):
    out = reduce_grad(grad, graph_id, ds_id)
    end_backward(deps, graph_id, release_reduce_buckets)
    return out
