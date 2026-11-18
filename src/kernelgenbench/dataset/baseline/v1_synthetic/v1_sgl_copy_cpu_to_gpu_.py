"""SGLang short int32 CPU-to-GPU in-place copy.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/elementwise/copy.cu#L21-L57
"""
import torch
def v1_sgl_copy_cpu_to_gpu_(input,output):
    assert input.device.type=='cpu' and output.device.type=='cuda'
    assert input.dtype==output.dtype==torch.int32 and input.ndim==output.ndim==1
    assert input.is_contiguous() and output.is_contiguous() and input.numel()==output.numel() and input.numel() in (32,64,72)
    output.copy_(input);return None
