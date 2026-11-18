"""vLLM CLS-token sequence pooling (CLSPool.forward).

Official source (vLLM main, commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/pooler/seqwise/methods.py#L37-L48
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/pool/metadata.py#L120-L138

``CLSPool.forward`` first rejects a partial-prefill cursor::

    if pooling_cursor.is_partial_prefill():
        raise RuntimeError("partial prefill is not supported with CLS pooling")

and then returns ``hidden_states[pooling_cursor.first_token_indices_gpu]``, whose
indices are the exclusive prefix sums of the prompt lengths. Both the guard and
the different index set separate this entry from ``LastPool``.
"""
import torch


def v1_vllm_cls_pool(hidden_states, prompt_lens_cpu, partial_prefill=False):
    if partial_prefill:
        raise RuntimeError("partial prefill is not supported with CLS pooling")
    assert prompt_lens_cpu.ndim == 1 and prompt_lens_cpu.numel() > 0
    prompt_lens = prompt_lens_cpu.to(device=hidden_states.device, dtype=torch.int64)
    cumsum = torch.zeros(
        prompt_lens.numel() + 1, dtype=torch.int64, device=hidden_states.device
    )
    torch.cumsum(prompt_lens, dim=0, out=cumsum[1:])
    return hidden_states[cumsum[:-1]]
