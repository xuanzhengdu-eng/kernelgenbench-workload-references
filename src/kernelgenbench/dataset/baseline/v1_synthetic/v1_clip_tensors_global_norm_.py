"""DeepSpeed in-place clipping using a supplied global norm.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/utils.py#L1054-L1071

``clip_tensors_by_global_norm`` returns the supplied norm and scales each tensor
in place only when ``max_norm / (global_norm + eps)`` is below one.  This item
keeps the original precomputed-norm contract (the ``global_norm=None`` branch of
the upstream signature computes the norm itself and is out of scope).  CSV row
172's ``source='deepspeed'`` branch is this same entry and is deduplicated here.
"""
def v1_clip_tensors_global_norm_(input_tensors, max_norm, global_norm, eps):
    clip_coef = max_norm / (global_norm + eps)
    if clip_coef < 1:
        for tensor in input_tensors:
            tensor.detach().mul_(clip_coef)
    return global_norm
