"""DeepSpeed gradient-norm nonfinite replacement; source ignores val.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/utils.py#L963-L968
"""
import torch
def v1_replace_nonfinite_inplace(input,device=None,val=-1.0):
    input.masked_fill_(~torch.isfinite(input),-1.0)
    return None
