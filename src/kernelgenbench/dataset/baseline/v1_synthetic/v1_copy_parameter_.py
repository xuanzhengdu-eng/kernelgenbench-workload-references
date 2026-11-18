"""vLLM parameter copy: in-place copy that returns the destination tensor.

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/utils.py#L79-L103

vLLM's weight loaders call this helper when copying a checkpoint tensor into a
parameter.  It is a plain ``dst.copy_(src)``: cast follows torch's copy rules
(including downcasts), the destination is modified in place and the very same
tensor object is returned.  It performs no shape assertion of its own; a shape
mismatch surfaces from ``copy_``.  This is a different contract from SGLang's
``copy_with_check`` (returns None, rejects downcasts), so the two are separate
items.
"""
import torch


def v1_copy_parameter_(target, loaded_weight):
    return target.copy_(loaded_weight)
