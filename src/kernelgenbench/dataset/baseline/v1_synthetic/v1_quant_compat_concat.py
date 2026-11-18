"""DeepSpeed quantized-weight compatibility packaging with inverse views.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L191-L221
"""
import torch
from torch import nn
def v1_quant_compat_concat(quantized_weight,quant_scale,quant_min,return_param):
    shapes=[t.shape for t in (quantized_weight,quant_scale,quant_min)]
    counts=[t.numel() for t in (quantized_weight,quant_scale,quant_min)]
    result=torch.cat([quantized_weight.reshape(-1),quant_scale.reshape(-1),quant_min.reshape(-1)])
    if return_param:result=nn.Parameter(result,requires_grad=False)
    def deconcat(tensor):
        pos=0;parts=[]
        for shape,n in zip(shapes,counts):
            parts.append(tensor.narrow(0,pos,n).view(shape));pos+=n
        return tuple(parts)
    result.deconcat=deconcat
    return result
