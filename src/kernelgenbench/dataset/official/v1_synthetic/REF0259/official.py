# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_cross_entropy.py#L124-L134
from megatron.core.fusions.fused_cross_entropy import _VocabParallelCrossEntropy

backward = _VocabParallelCrossEntropy.backward
