# Official entry: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z1.cpp#L33-L87
# Op schema: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/init.cpp#L19
# Dispatch: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/deepcompile.cpp#L105-L118
# Bindings: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/init.cpp#L98-L101
# Registration used by the compiler: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/compile/passes/zero_1_and_2_compile.py#L24-L59
# Handle loader: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/compile/util.py#L46-L53
import torch

from deepspeed.compile.util import get_deepcompile_handle

_dc = get_deepcompile_handle()


def reduce_grad(grad, graph_id, ds_id):
    return torch.ops.dc.reduce_grad(grad, graph_id, ds_id)


def end_backward(deps, graph_id, release_reduce_buckets=True):
    return torch.ops.dc.end_backward(deps, graph_id, release_reduce_buckets)


def register_graph_z1(graph_id, ds_ids):
    return _dc.register_graph_z1(graph_id, ds_ids)


def register_param(ds_id, ds_shape, ds_tensor, grad_buffer, offset):
    return _dc.register_param(ds_id, ds_shape, ds_tensor, grad_buffer, offset)


def update_param_grad_buffer(ds_id, grad_buffer, offset):
    return _dc.update_param_grad_buffer(ds_id, grad_buffer, offset)


def start_backward(update):
    return _dc.start_backward(update)


def init(process_group, compile_config, initial_reduce_bucket_size):
    return _dc.init(process_group, compile_config, initial_reduce_bucket_size)


def reset():
    return _dc.reset()


def cleanup():
    return _dc.cleanup()
