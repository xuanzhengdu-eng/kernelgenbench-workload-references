"""vLLM tensor-parallel weight ``amax`` reduction (``ReduceOp.MAX``).

Fixed vLLM source, entry ``amax_for_tp_weight_quant``:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/quant_utils.py#L48-L59
"""
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    amax_for_tp_weight_quant as run,
)

__all__ = ["run"]
