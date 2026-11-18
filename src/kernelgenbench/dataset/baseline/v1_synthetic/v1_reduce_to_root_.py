"""SUM reduce to root into a supplied output buffer; non-root output unspecified.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/device_communicators/pynccl.py#L429-L455
"""
import torch.distributed as dist
def v1_reduce_to_root_(output_tensor,input_tensor,root,group,disabled=False):
    if disabled:return
    buf=input_tensor.clone();dist.reduce(buf,dst=root,op=dist.ReduceOp.SUM,group=group)
    if dist.get_rank(group)==root:output_tensor.copy_(buf)
