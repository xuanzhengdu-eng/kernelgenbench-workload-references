"""SGLang packed-sequence pooling driven by the real PoolingType enum.

Official source (sgl-project/sglang, commit
62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/pooler.py#L48-L76

``pool_hidden_states(pooling_type, hidden_states, forward_batch)`` has one
logical entry: the ``PoolingType`` IntEnum (LAST=0, CLS=1, MEAN=2) selects the
calculation, and every branch returns a single ``(batch_size, hidden_size)``
tensor of the hidden-state dtype on the hidden-state device.

  * LAST  -> rows ``cumsum(extend_seq_lens) - 1``
  * CLS   -> rows from the exclusive prefix sum
  * MEAN  -> ``(cumsum[end] - cumsum[end-1]) / length`` computed in the input
             dtype (no FP32 upcast; unlike the vLLM mean pooler)

No branch writes its inputs, and none has a partial-prefill guard (that exists
in the vLLM pooler, not here). An unknown pooling type raises ValueError.
"""
import torch


def v1_sequence_first_last_pool(hidden_states, extend_seq_lens, pooling_type):
    # The real argument is a sglang PoolingType member; dispatch on its name so
    # this reference stays loadable without importing sglang.
    mode = getattr(pooling_type, "name", pooling_type)
    prompt_lens = extend_seq_lens
    if mode == "LAST":
        last_token_indices = torch.cumsum(prompt_lens, dim=0) - 1
        return hidden_states[last_token_indices]
    if mode == "CLS":
        first_token_flat_indices = torch.zeros_like(prompt_lens)
        first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
        return hidden_states[first_token_flat_indices]
    if mode == "MEAN":
        end_indices = torch.cumsum(prompt_lens, dim=0) - 1
        cumulative_hidden_states = torch.cumsum(hidden_states, dim=0)
        sums = cumulative_hidden_states[end_indices]
        preceding_sums = torch.zeros_like(sums)
        preceding_sums[1:] = cumulative_hidden_states[end_indices[:-1]]
        return (sums - preceding_sums) / prompt_lens.unsqueeze(-1)
    raise ValueError(f"Unsupported pooling type: {pooling_type}")
