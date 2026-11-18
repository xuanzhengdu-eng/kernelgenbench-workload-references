"""vLLM fused weighted expert-output reduction with optional route masking.

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py#L95-L177

Contract (``moe_fused_mul_sum`` Triton kernel):
- ``inputs`` is ``[M, K, H]`` contiguous, ``topk_weights`` is ``[M, K]``
  contiguous; both support fp32/fp16/bf16. The accumulation is fp32 and the
  result is cast once to ``inputs.dtype``.
- ``outputs`` may be pre-allocated ``[M, H]`` (returned as the same object) or
  omitted, in which case a fresh tensor is allocated and returned.
- ``topk_ids`` (int32/int64, ``[M, K]``) marks slots with id ``< 0`` as
  invalid; ``expert_map`` (only with ``topk_ids``) additionally drops slots
  whose mapped local id is ``< 0``. Invalid slots are never loaded, so NaN
  payloads there do not propagate.
- A row whose ids are all invalid is left untouched (``preserve``): the kernel
  early-returns before writing it. ``num_valid_tokens`` is a device scalar; rows
  at or past it are also left untouched.
- Inputs are not modified.
- This is the weighted vLLM branch split out of the former unified reference;
  it is distinct from ``masked_moe_sum`` (unweighted, zeroes all-invalid rows,
  in-place) and from SGLang ``moe_sum`` (unweighted, input-dtype accumulation).
"""
import torch


def v1_vllm_moe_fused_mul_sum(
    inputs, topk_weights, outputs=None, topk_ids=None, expert_map=None, num_valid_tokens=None
):
    assert inputs.ndim == 3 and inputs.is_contiguous()
    assert topk_weights.ndim == 2 and topk_weights.is_contiguous()
    assert inputs.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype in (torch.float32, torch.float16, torch.bfloat16)
    num_tokens, top_k, hidden = inputs.shape
    shape = (num_tokens, hidden)
    if outputs is None:
        outputs = torch.empty(shape, dtype=inputs.dtype, device=inputs.device)
    assert outputs.shape == shape
    assert topk_weights.shape == (num_tokens, top_k)
    assert expert_map is None or topk_ids is not None
    if topk_ids is not None:
        assert topk_ids.shape == (num_tokens, top_k) and topk_ids.is_contiguous()
        assert topk_ids.dtype in (torch.int32, torch.int64)
    valid_rows = num_tokens if num_valid_tokens is None else int(num_valid_tokens.item())

    for row in range(min(valid_rows, num_tokens)):
        takes = []
        if topk_ids is not None:
            any_valid = False
            for slot in range(top_k):
                expert = int(topk_ids[row, slot])
                take = expert >= 0
                any_valid = any_valid or take
                if take and expert_map is not None:
                    take = int(expert_map[expert]) >= 0
                takes.append(take)
            if not any_valid:
                # All-padding row: the kernel returns before writing the output.
                continue
        else:
            takes = [True] * top_k
        total = torch.zeros(hidden, dtype=torch.float32, device=inputs.device)
        for slot, take in enumerate(takes):
            if take:
                total = total + inputs[row, slot].float() * topk_weights[row, slot].float()
        outputs[row].copy_(total.to(outputs.dtype))
    return outputs
