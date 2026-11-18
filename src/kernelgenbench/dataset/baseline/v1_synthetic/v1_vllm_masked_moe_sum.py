"""vLLM unweighted masked top-k expert sum (in place).

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py#L1005-L1018

Contract (``masked_moe_sum`` Triton kernel):
- ``intermediate`` is ``[M, topk, H]`` contiguous, ``topk_ids`` is
  ``[M, topk]`` int with negative entries marking invalid/non-local slots, and
  ``output`` is a pre-allocated contiguous ``[M, H]`` tensor of the same dtype.
- Valid slots are added in fp32 and rounded once to ``output.dtype``; invalid
  slots are skipped entirely, so NaN payloads there are not read.
- Every output row is written, including all-invalid rows, which become exactly
  zero (contrast with the weighted ``moe_fused_mul_sum`` preserve behaviour).
- Returns ``None`` and mutates ``output`` only; ``intermediate`` and
  ``topk_ids`` are untouched.
"""
import torch


def v1_vllm_masked_moe_sum(intermediate, topk_ids, output):
    assert intermediate.ndim == 3 and intermediate.is_contiguous()
    rows, top_k, hidden = intermediate.shape
    assert topk_ids.shape == (rows, top_k)
    assert output.shape == (rows, hidden) and output.is_contiguous()
    assert output.dtype == intermediate.dtype
    for row in range(rows):
        total = torch.zeros(hidden, dtype=torch.float32, device=intermediate.device)
        for slot in range(top_k):
            if int(topk_ids[row, slot]) >= 0:
                total = total + intermediate[row, slot].float()
        output[row].copy_(total.to(output.dtype))
    return None
