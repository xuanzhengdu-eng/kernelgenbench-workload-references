"""vLLM LAST-token sequence pooling (LastPool.forward).

Official source (vLLM main, commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/pooler/seqwise/methods.py#L50-L57
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/pool/metadata.py#L120-L138

``LastPool.forward`` reads ``pooling_cursor.last_token_indices_gpu`` and returns
``hidden_states[indices]``. The cursor's last indices are the inclusive prefix
sums of the prompt lengths minus one, i.e. ``cumsum(lengths)[1:] - 1``.

Unlike ``CLSPool`` this entry performs **no** partial-prefill check: a cursor
flagged ``partial_prefill`` is still pooled. That missing RuntimeError is the
observable difference between the two poolers, so this reference accepts the
flag and ignores it on purpose.
"""
import torch


def v1_vllm_last_pool(hidden_states, prompt_lens_cpu, partial_prefill=False):
    del partial_prefill  # LastPool has no partial-prefill guard upstream.
    assert prompt_lens_cpu.ndim == 1 and prompt_lens_cpu.numel() > 0
    prompt_lens = prompt_lens_cpu.to(device=hidden_states.device, dtype=torch.int64)
    cumsum = torch.zeros(
        prompt_lens.numel() + 1, dtype=torch.int64, device=hidden_states.device
    )
    torch.cumsum(prompt_lens, dim=0, out=cumsum[1:])
    return hidden_states[cumsum[1:] - 1]
