"""vLLM masked activation amax -> per-tensor quantization scale.

Source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L458-L503
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L506-L593

The official implementation is a Triton kernel that atomically reduces a
blocked amax into a one-element FP32 tensor.  This reference reproduces the
same valid-row selection, scale_max and eps clamp with torch reductions.
Exactly one of row_valid / num_valid / expert_num_tokens selects valid rows,
and rows_per_expert is valid only together with expert_num_tokens.
"""
import torch


def v1_masked_amax_scale(
    x,
    quant_dtype,
    *,
    row_valid=None,
    num_valid=None,
    expert_num_tokens=None,
    rows_per_expert=None,
    eps=1e-10,
):
    assert x.dim() == 2, f"expected 2D activations, got {x.shape}"
    assert (
        sum(
            m is not None
            for m in (row_valid, num_valid, expert_num_tokens)
        )
        == 1
    ), "pass exactly one of row_valid / num_valid / expert_num_tokens"
    assert (expert_num_tokens is None) == (rows_per_expert is None), (
        "rows_per_expert is only valid together with expert_num_tokens"
    )

    rows = torch.arange(x.size(0), device=x.device)
    if row_valid is not None:
        assert row_valid.shape == (x.size(0),), (
            f"expected {x.size(0)} row flags, got {tuple(row_valid.shape)}"
        )
        valid = row_valid.bool()
    elif num_valid is not None:
        assert num_valid.numel() == 1, f"expected 1 row count, got {num_valid}"
        valid = rows < num_valid
    else:
        assert rows_per_expert is not None and rows_per_expert > 0
        assert x.size(0) % rows_per_expert == 0, (
            f"{x.size(0)} rows is not a multiple of {rows_per_expert}"
        )
        assert expert_num_tokens.numel() == x.size(0) // rows_per_expert, (
            f"expected {x.size(0) // rows_per_expert} expert counts, "
            f"got {expert_num_tokens.numel()}"
        )
        expert = rows // rows_per_expert
        valid = (rows % rows_per_expert) < expert_num_tokens[expert]

    scale_max = (
        127.0
        if quant_dtype == torch.int8
        else float(torch.finfo(quant_dtype).max)
    )
    values = torch.where(
        valid[:, None], x.float().abs(), torch.zeros((), device=x.device)
    )
    # The Triton output tensor is always one-element float32.
    return (values.amax() / scale_max).clamp_min(eps).reshape(1).float()
