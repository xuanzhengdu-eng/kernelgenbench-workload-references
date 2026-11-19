# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L252-L339
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L51-L66
import torch
from deepspeed.compile.util import get_deepcompile_handle

dc = get_deepcompile_handle()

reduce_grad = torch.ops.dc.reduce_grad.default
end_backward = torch.ops.dc.end_backward.default
register_graph_z3 = dc.register_graph_z3
register_z3_param = dc.register_z3_param
start_backward = dc.start_backward
init = dc.init
cleanup = dc.cleanup
