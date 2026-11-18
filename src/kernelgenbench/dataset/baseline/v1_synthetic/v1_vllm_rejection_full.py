"""vLLM speculative rejection: greedy, random and mixed batches.

Official sources (fixed commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
- ``rejection_sample`` dispatch and greedy/random/mixed composition:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L432-L586
- ``generate_uniform_probs`` (float64 per-request uniforms):
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L689-L742
- ``sample_recovered_tokens`` (per-request exponential draw ``q``):
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L745-L788
- ``rejection_greedy_sample_kernel``:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L792-L847
- ``rejection_random_sample_kernel``:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L895-L967
- ``sample_recovered_tokens_kernel``:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/rejection_sampler.py#L1070-L1151

Contract (``rejection_sample`` with a ``SpecDecodeMetadata`` and a
``SamplingMetadata``):

- ``draft_token_ids`` is a flat contiguous int32 tensor; ``num_draft_tokens[i]``
  gives request ``i``'s draft count and must agree with ``cu_num_draft_tokens``.
  ``bonus_token_ids`` is ``[batch, 1]`` and is appended when every draft token is
  accepted.  The output is int32 ``[batch, max_spec_len + 1]`` padded with ``-1``.
- Greedy requests compare the draft tokens with the target argmax and stop at the
  first mismatch, keeping the target token at the rejected position.
- Random requests accept while ``target_prob / draft_prob >= uniform`` (with
  ``draft_prob = 1`` when no draft probabilities are given); otherwise the
  recovered token is ``argmax(relu(target - draft) / q)`` where ``q`` is an
  exponential draw.  Uniforms are float64 and drawn per request through the
  supplied generators; ``q`` is drawn per batch row with the same generators.
- ``mode='mixed'`` marks requests greedy through ``is_greedy``; greedy rows never
  consume the random path's output.

This file holds the vLLM entry only.  The SGLang DFlash, tree target-only and UNO
sparse components that the historical file bundled are separate official entries
with different contracts, and live in their own reference files.
"""
import torch

def _uniform_probs(num_tokens, num_draft_tokens, generators, device):
    uniform = torch.rand((num_tokens,), dtype=torch.float64, device=device)
    start = 0
    for request, count in enumerate(num_draft_tokens):
        if count == 0:
            continue
        end = start + count
        generator = generators.get(request)
        if generator is not None:
            uniform[start:end].uniform_(generator=generator)
        start = end
    return uniform


def v1_vllm_rejection_full(draft_token_ids, num_draft_tokens, max_spec_len,
                           cu_num_draft_tokens, target_logits, bonus_token_ids, mode,
                           draft_probs=None, is_greedy=None, generators=None,
                           use_fp64_gumbel=False):
    batch = len(num_draft_tokens)
    assert batch > 0 and max_spec_len >= 0 and mode in ("greedy", "random", "mixed")
    assert draft_token_ids.ndim == 1 and draft_token_ids.is_contiguous()
    assert draft_token_ids.dtype == torch.int32
    assert cu_num_draft_tokens.shape == (batch,) and cu_num_draft_tokens.dtype == torch.int32
    assert bonus_token_ids.shape == (batch, 1) and bonus_token_ids.is_contiguous()
    assert target_logits.ndim == 2 and target_logits.shape[0] == draft_token_ids.numel()
    assert sum(num_draft_tokens) == draft_token_ids.numel()
    running = 0
    prefix = []
    for count in num_draft_tokens:
        running += count
        prefix.append(running)
    assert cu_num_draft_tokens.cpu().tolist() == prefix
    if mode == "mixed":
        assert is_greedy is not None and is_greedy.shape == (batch,)
        assert is_greedy.dtype == torch.bool and is_greedy.device == target_logits.device
    if generators is None:
        generators = {}
    assert all(isinstance(key, int) and 0 <= key < batch
               and isinstance(value, torch.Generator) for key, value in generators.items())

    out = torch.full((batch, max_spec_len + 1), -1, device=target_logits.device,
                     dtype=torch.int32)
    start = 0
    if mode != "random":
        target_argmax = target_logits.argmax(-1)
        for request, count in enumerate(num_draft_tokens):
            greedy = mode == "greedy" or bool(is_greedy[request])
            if not greedy:
                start += count
                continue
            if count:
                tokens = draft_token_ids[start:start + count]
                argmax = target_argmax[start:start + count]
                mismatch = torch.nonzero(tokens != argmax).reshape(-1)
                if mismatch.numel():
                    end = int(mismatch[0]) + 1
                    out[request, :end] = argmax[:end].to(torch.int32)
                else:
                    out[request, :count] = argmax.to(torch.int32)
                    out[request, count] = bonus_token_ids[request, 0].to(torch.int32)
            else:
                out[request, 0] = bonus_token_ids[request, 0].to(torch.int32)
            start += count
        if mode == "greedy":
            return out

    assert max_spec_len > 0
    uniform = _uniform_probs(draft_token_ids.numel(), num_draft_tokens, generators,
                             target_logits.device)
    target = target_logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty((batch, target.shape[1]),
                    dtype=torch.float64 if use_fp64_gumbel else torch.float32,
                    device=target_logits.device).exponential_()
    for request, generator in generators.items():
        if num_draft_tokens[request] > 0:
            q[request].exponential_(generator=generator)
    inverse_q = q.reciprocal()

    start = 0
    for request, count in enumerate(num_draft_tokens):
        if mode == "mixed" and bool(is_greedy[request]):
            start += count
            continue
        rejected = False
        for position in range(count):
            index = start + position
            token = int(draft_token_ids[index])
            valid = token >= 0
            draft_prob = (1.0 if draft_probs is None
                          else float(draft_probs[index, token])) if valid else 0.0
            if valid and draft_prob > 0:
                ratio = (target[index, token] if draft_probs is None
                         else target[index, token] / draft_probs[index, token])
                accepted = float(ratio) >= float(uniform[index])
            else:
                accepted = False
            if accepted:
                out[request, position] = token
                continue
            if draft_probs is None:
                residual = target[index].clone()
                if valid:
                    residual[token] = 0
            else:
                residual = (target[index] - draft_probs[index]).clamp_min(0.0)
            out[request, position] = (residual * inverse_q[request]).argmax().to(torch.int32)
            rejected = True
            break
        if not rejected:
            out[request, count] = bonus_token_ids[request, 0].to(torch.int32)
        start += count
    return out
