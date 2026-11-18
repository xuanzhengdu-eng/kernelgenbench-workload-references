"""DeepSpeed DeepCompile registered-parameter gradient buffer replacement.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z1.cpp#L130-L134
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/deepcompile.h#L413-L417
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/init.cpp#L95-L97

Contract:
  * ``update_param_grad_buffer`` is a native binding over
    ``DSParamRegistry::updateGradBuffer``; it returns ``None``.
  * A non-empty replacement buffer is zeroed in place; an empty buffer
    (``numel() == 0``) is left untouched.
  * The registry entry keyed by ``ds_id`` is rebound to the new buffer and to
    ``offset``. The registry is internal to the extension; the rebinding is only
    observable through later calls on the same id and through the error raised for
    an unknown id.
  * ``ds_id`` must already be registered; the native op raises for an unknown id.
"""
def v1_ds_update_grad_buffer_(grad_buffer, offset, ds_id):
    if grad_buffer.numel() > 0:
        grad_buffer.zero_()
    return None
