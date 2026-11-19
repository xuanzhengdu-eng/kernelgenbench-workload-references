# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/cross_entropy.py#L85-L119
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/cross_entropy.py#L192-L217
from megatron.core.tensor_parallel.cross_entropy import _VocabParallelCrossEntropy

run = _VocabParallelCrossEntropy.backward

v1_megatron_vocab_parallel_ce_backward_ = _VocabParallelCrossEntropy.backward
