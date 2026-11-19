# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/python/sgl_kernel/kvcacheio.py#L218-L228
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L822-L971
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/common_extension.cc#L269-L272
from sgl_kernel.kvcacheio import transfer_embedding_ranges_direct as run
