# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_experts.py#L269-L290
from deepspeed.moe.ep_experts import _ZeroEmptyExpertGrad

run = _ZeroEmptyExpertGrad.backward
