# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/optimizer.py#L497-L522
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer

run = MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan
