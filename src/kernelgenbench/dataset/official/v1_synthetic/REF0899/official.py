# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L217-L245
from megatron.core.tensor_parallel.mappings import _ReduceFromModelParallelRegion as _entry
run = _entry.backward
