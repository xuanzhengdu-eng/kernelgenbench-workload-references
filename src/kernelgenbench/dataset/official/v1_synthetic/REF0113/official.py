# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L911-L935
from megatron.core.transformer.custom_layers.batch_invariant_kernels import BatchInvariantRMSNormFn

run = BatchInvariantRMSNormFn.backward
