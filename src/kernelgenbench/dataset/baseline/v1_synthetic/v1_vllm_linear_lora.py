"""vLLM base linear plus per-token LoRA, CPU/PyTorch punica backend (723-row 101).

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/layers/base_linear.py#L208-L242
Shrink/expand ops:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/ops/torch_ops/lora_ops.py#L60-L95
Wrapper:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/punica_wrapper/punica_cpu.py#L264-L311

Reachable branch: ``_apply_sync`` computes the quant method's base linear and
then ``_apply_lora_to_output`` adds the adapter for every token whose mapping is
not ``-1``.  ``adapter_ids == -1`` leaves that row at the base output; a valid id
adds ``b @ (a @ x) * scale`` (scale 1.0 here).  The shrink/expand pair runs in
fp32 for the rank reduction (``add_shrink`` allocates an fp32 buffer) and the
result is added in place into the output tensor, which is the object returned.
Leading dimensions are flattened to ``[tokens, K]`` before the adapter math and
the result is reshaped back.
"""
import torch
import torch.nn.functional as F


def v1_vllm_linear_lora(x, base_weight, bias, lora_a, lora_b, adapter_ids,
                       prefill=False):
    y = F.linear(x, base_weight, bias)
    shape = y.shape
    flat_x = x.reshape(-1, x.shape[-1])
    out = y.reshape(-1, y.shape[-1])
    ids = adapter_ids.reshape(-1)
    safe = ids.clamp_min(0)
    a = lora_a[safe, 0].float()
    b = lora_b[safe, 0].to(y.dtype)
    low = torch.bmm(a, flat_x.float().unsqueeze(-1)).squeeze(-1)
    low = low.masked_fill((ids == -1)[:, None], 0)
    delta = torch.bmm(b, low.to(y.dtype).unsqueeze(-1)).squeeze(-1)
    delta = delta.masked_fill((ids == -1)[:, None], 0)
    out.add_(delta)
    # ``prefill`` picks the sgmv_* pair instead of bgmv_*; both expand the same
    # per-token mapping, so the arithmetic below is identical.
    _ = prefill
    return out.reshape(shape)
