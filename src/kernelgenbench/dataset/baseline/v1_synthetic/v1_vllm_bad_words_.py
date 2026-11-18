"""vLLM bad-word masking with speculative input tokens, in place.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/bad_words.py#L79-L180

Contract, as reached through ``BadWordsState.apply`` and the Triton
``_bad_words_kernel``:

* ``logits`` is ``[num_tokens, vocab]``; rows may be strided. Vocabulary
  positions are indexed by the last bad-word token.
* For row ``r``: ``req = expanded_idx_mapping[r]`` and
  ``pos = expanded_local_pos[r]`` with ``begin = r - pos``. The history is the
  committed tokens after the prompt, ``all_token_ids[req, prompt_len:total_len]``
  followed by the speculative ``input_ids[begin + 1 : begin + pos + 1]``.
* A bad word masks its final token with ``-inf`` iff its prefix fits in the
  history and equals the history suffix. No other entry is written.
* Mutates ``logits`` in place and returns the same tensor. The int32 metadata
  tensors are read-only.
"""


def v1_vllm_bad_words_(
    logits,
    expanded_idx_mapping,
    bad_word_token_ids,
    bad_word_offsets,
    num_bad_words,
    all_token_ids,
    prompt_len,
    total_len,
    input_ids,
    expanded_local_pos,
):
    for row in range(logits.shape[0]):
        req = int(expanded_idx_mapping[row])
        pos = int(expanded_local_pos[row])
        begin = row - pos
        prompt = int(prompt_len[req])
        total = int(total_len[req])
        history = (
            all_token_ids[req, prompt:total].tolist()
            + input_ids[begin + 1 : begin + pos + 1].tolist()
        )
        for j in range(int(num_bad_words[req])):
            left = int(bad_word_offsets[req, j])
            right = int(bad_word_offsets[req, j + 1])
            word = bad_word_token_ids[req, left:right].tolist()
            prefix = word[:-1]
            if len(prefix) <= len(history) and (
                not prefix or history[-len(prefix) :] == prefix
            ):
                logits[row, word[-1]] = float("-inf")
    return logits
