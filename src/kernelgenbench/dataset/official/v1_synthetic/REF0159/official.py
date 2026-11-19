# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/input_hadamard.py#L24-L42
from vllm.model_executor.layers.quantization.input_hadamard import (
    block_hadamard_rotate,
    hadamard_matrix,
)

run = block_hadamard_rotate

__all__ = ["block_hadamard_rotate", "hadamard_matrix", "run"]
