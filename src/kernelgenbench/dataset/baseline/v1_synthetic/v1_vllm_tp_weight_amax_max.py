"""vLLM tensor-parallel weight ``amax`` reduction (MAX).

``amax_for_tp_weight_quant`` reduces a sharded weight's absolute maximum over the
TP group with ``ReduceOp.MAX`` and returns the same tensor object.  The reduction
is unconditional (there is no enabled flag) and requires a dense tensor.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/quant_utils.py#L48-L59
"""
import torch
import torch.distributed as dist

def v1_vllm_tp_weight_amax_max(amax,group):
    assert amax.is_contiguous()
    dist.all_reduce(amax,op=dist.ReduceOp.MAX,group=group)
    return amax
