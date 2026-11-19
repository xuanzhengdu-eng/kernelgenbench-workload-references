# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/init.cpp#L95-L97
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z1.cpp#L130-L134
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/deepcompile.h#L413-L417
from deepspeed.compile.util import get_deepcompile_handle


def register_param(ds_id, ds_shape, ds_tensor, grad_buffer, offset):
    return get_deepcompile_handle().register_param(
        ds_id, ds_shape, ds_tensor, grad_buffer, offset)


def run(ds_id, grad_buffer, offset):
    return get_deepcompile_handle().update_param_grad_buffer(ds_id, grad_buffer, offset)
