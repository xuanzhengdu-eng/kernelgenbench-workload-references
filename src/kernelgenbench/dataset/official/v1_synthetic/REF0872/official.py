# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/random_ltd/dropping_utils.py#L119-L132
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/random_ltd/pt_binding.cpp#L26-L81
from deepspeed.ops.random_ltd.dropping_utils import ScatterTokens

run = ScatterTokens.backward
