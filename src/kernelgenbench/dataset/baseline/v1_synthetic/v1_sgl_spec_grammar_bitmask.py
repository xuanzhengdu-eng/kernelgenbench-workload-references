"""SGLang speculative-tree grammar vocabulary bitmask generation (CPU).

Source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L689-L754
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L757-L804

The public entry is ``generate_token_bitmask``; ``traverse_tree`` is its
internal recursive DFS and is inlined here so the reference is a single
standalone function.  Grammar state is mutated (accept/rollback) but every
node is rolled back before returning.
"""
import torch


def v1_sgl_spec_grammar_bitmask(
    reqs,
    retrieve_next_token_cpu,
    retrieve_next_sibling_cpu,
    draft_tokens_cpu,
    vocab_size,
):
    rows, width = draft_tokens_cpu.shape
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    assert retrieve_next_token_cpu.shape == retrieve_next_sibling_cpu.shape
    assert retrieve_next_sibling_cpu.shape == draft_tokens_cpu.shape

    allocate_token_bitmask = None
    last_grammar = None

    for batch, req in enumerate(reqs):
        grammar = req.grammar
        if grammar is None:
            continue

        if allocate_token_bitmask is None:
            allocate_token_bitmask = grammar.allocate_vocab_mask(
                vocab_size=vocab_size,
                batch_size=draft_tokens_cpu.numel(),
                device="cpu",
            )
        last_grammar = grammar
        mask = allocate_token_bitmask[
            batch * width : (batch + 1) * width
        ]

        def dfs(node, parent_pos):
            if node == 0:
                # The root token is always accepted from the previous step.
                is_accepted = True
            else:
                parent_bits = mask[parent_pos]
                token = int(draft_tokens_cpu[batch, node])
                if vocab_size and token >= vocab_size:
                    is_accepted = False
                else:
                    # 32 mask bits are packed into one int32 word.
                    is_accepted = (
                        int(parent_bits[token // 32])
                        & (1 << (token % 32))
                    ) != 0

            if is_accepted:
                if node != 0:
                    grammar.accept_token(int(draft_tokens_cpu[batch, node]))
                if not grammar.is_terminated():
                    grammar.fill_vocab_mask(mask, node)
                    child = int(retrieve_next_token_cpu[batch, node])
                    if child != -1:
                        dfs(child, node)
                if node != 0:
                    grammar.rollback(1)

            sibling = int(retrieve_next_sibling_cpu[batch, node])
            if sibling != -1:
                dfs(sibling, parent_pos)

        dfs(0, -1)

    return allocate_token_bitmask, last_grammar
