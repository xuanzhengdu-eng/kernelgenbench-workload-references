"""marlin_moe_permute_scales from the pinned vllm Marlin layout helpers."""
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L535-L547
import torch
from kernelgenbench.dataset.baseline.v1_synthetic.v1_marlin_permute_scales import v1_marlin_permute_scales as marlin_permute_scales


def v1_marlin_moe_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
):
    num_experts = s.shape[0]
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output
