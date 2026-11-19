# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/elementwise/copy.cu#L21-L57
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/python/sgl_kernel/elementwise.py#L366-L368
from sgl_kernel.elementwise import copy_to_gpu_no_ce as run
