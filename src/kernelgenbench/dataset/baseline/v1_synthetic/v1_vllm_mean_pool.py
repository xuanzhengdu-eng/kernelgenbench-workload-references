"""vLLM sequence-wise MEAN pooling over packed prompt hidden states.

Official source (vLLM main, commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/pooler/seqwise/methods.py#L60-L107

The official MeanPool.forward reads a PoolingCursor built from prompt lengths:
  * empty batch -> new_empty((0, hidden), float32);
  * otherwise segment ids are repeated by prompt length and FP32 index_add
    accumulates each packed sequence, followed by division by prompt length;
  * partial prefill is rejected with RuntimeError.
A zero-length sequence therefore divides its zero sum by zero and yields NaN.
Inputs and pooling metadata are not modified.
"""
import torch

_MEAN_POOL_ACCUMULATION_CHUNK_BYTES = 16 * 1024 * 1024


def v1_vllm_mean_pool(hidden_states, prompt_lens_cpu, partial_prefill=False):
    if partial_prefill:
        raise RuntimeError("partial prefill is not supported with MEAN pooling")

    num_seqs = int(prompt_lens_cpu.numel())
    hidden_size = int(hidden_states.shape[-1])
    if num_seqs == 0:
        # Mirrors the upstream early return: no data, FP32 dtype and (0, H) shape.
        return hidden_states.new_empty((0, hidden_size), dtype=torch.float32)

    prompt_lens = prompt_lens_cpu.to(device=hidden_states.device, dtype=torch.int64)
    segment_ids = torch.repeat_interleave(
        torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
        prompt_lens,
        output_size=int(prompt_lens_cpu.sum()),
    )
    segment_sums = torch.zeros(
        (num_seqs, hidden_size), dtype=torch.float32, device=hidden_states.device
    )

    bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
    chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        segment_sums.index_add_(
            0,
            segment_ids[start:end],
            hidden_states[start:end].to(dtype=torch.float32),
        )

    return segment_sums / prompt_lens.unsqueeze(1)
