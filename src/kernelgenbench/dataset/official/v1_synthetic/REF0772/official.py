# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/spec.cpp#L689-L789
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/python/sgl_kernel/speculative.py#L92-L123
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/speculative/reconstruct_tree.py#L163-L193
from sglang.kernels.ops.speculative.reconstruct_tree import (
    reconstruct_indices_from_tree_mask_func as run,
)
from sgl_kernel.speculative import (
    reconstruct_indices_from_tree_mask as run_cpu_dispatch,
)
