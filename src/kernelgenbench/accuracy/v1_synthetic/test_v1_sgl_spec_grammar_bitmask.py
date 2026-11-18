"""Accuracy and timing for v1_sgl_spec_grammar_bitmask.

The test drives a tiny DFA grammar over a speculative draft tree.  It covers
branching (a sibling subtree), a single chain, a request without grammar
(returns (None, None)) and a terminal root (no mask bits are produced).
After traversal the grammar stack and accept/rollback counts must be back to
their initial values.  The oracle walks the same table independently.
"""
import time

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows642_651_test_utils import (
    clone_tree,
    measure_ms,
    move_tree,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


CASES = ["branching", "single", "no_grammar", "terminal_root"]
# The verifier register reads COMBOS; each entry selects one tree/grammar case.
COMBOS = CASES


class DFA:
    def __init__(self, table, terminal=False):
        self.table = table
        self.stack = [0]
        self.terminal_root = terminal
        self.allocations = 0
        self.rollbacks = 0
        self.accepts = 0

    def allocate_vocab_mask(self, vocab_size, batch_size, device):
        self.allocations += 1
        return torch.zeros(
            (batch_size, (vocab_size + 31) // 32),
            dtype=torch.int32,
            device=device,
        )

    def accept_token(self, token):
        self.stack.append(self.table[self.stack[-1]][token])
        self.accepts += 1

    def rollback(self, count):
        del self.stack[-count:]
        self.rollbacks += count

    def is_terminated(self):
        return self.stack[-1] == 3 or (
            self.terminal_root and len(self.stack) == 1
        )

    def fill_vocab_mask(self, out, row):
        for token, next_state in enumerate(self.table[self.stack[-1]]):
            if next_state >= 0:
                word = token // 32
                value = int(out[row, word]) | (1 << (token % 32))
                out[row, word] = (
                    value - (1 << 32) if value >= (1 << 31) else value
                )


class Req:
    def __init__(self, grammar):
        self.grammar = grammar


def input_build(case):
    vocab = 64
    table = [[-1] * vocab for _ in range(4)]
    for token, state in [(1, 1), (2, 2), (31, 3), (33, 1)]:
        table[0][token] = state
    for token, state in [(31, 3), (33, 2), (2, 1)]:
        table[1][token] = state
    for token, state in [(3, 3), (31, 1)]:
        table[2][token] = state

    if case == "no_grammar":
        reqs = [Req(None), Req(None)]
    elif case == "single":
        reqs = [Req(DFA(table))]
    elif case == "terminal_root":
        reqs = [Req(DFA(table, True))]
    else:
        reqs = [Req(DFA(table)), Req(None), Req(DFA(table))]

    batch = len(reqs)
    retrieve_next_token = torch.tensor(
        [[1, 3, 5, -1, -1, -1]] * batch, dtype=torch.int64
    )
    retrieve_next_sibling = torch.tensor(
        [[-1, 2, -1, 4, -1, -1]] * batch, dtype=torch.int64
    )
    draft_tokens = torch.tensor(
        [[7, 1, 2, 31, 33, 70]] * batch, dtype=torch.int64
    )
    return {
        "reqs": reqs,
        "retrieve_next_token_cpu": retrieve_next_token,
        "retrieve_next_sibling_cpu": retrieve_next_sibling,
        "draft_tokens_cpu": draft_tokens,
        "vocab_size": vocab,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_spec_grammar_bitmask(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_spec_grammar_bitmask(**inputs)


def _oracle(inputs):
    reqs = inputs["reqs"]
    batch, width = inputs["draft_tokens_cpu"].shape
    vocab = inputs["vocab_size"]
    if all(req.grammar is None for req in reqs):
        return None, None

    out = torch.zeros(
        (batch * width, (vocab + 31) // 32), dtype=torch.int32
    )
    last = None
    for index, req in enumerate(reqs):
        grammar = req.grammar
        if grammar is None:
            continue
        last = index
        # Immutable integer DFA state instead of the mutable grammar object.
        todo = [(0, 0, True)]
        while todo:
            node, state, valid = todo.pop()
            sibling = int(inputs["retrieve_next_sibling_cpu"][index, node])
            if sibling != -1:
                todo.append((sibling, state, True))
            token = int(inputs["draft_tokens_cpu"][index, node])
            next_state = (
                state
                if node == 0
                else (
                    grammar.table[state][token]
                    if 0 <= token < vocab
                    else -1
                )
            )
            if (
                next_state < 0
                or next_state == 3
                or (grammar.terminal_root and node == 0)
            ):
                continue
            for candidate, target in enumerate(grammar.table[next_state]):
                if target >= 0:
                    position = index * width + node
                    word = candidate // 32
                    bits = int(out[position, word]) | (
                        1 << (candidate % 32)
                    )
                    out[position, word] = (
                        bits - (1 << 32) if bits >= (1 << 31) else bits
                    )
            child = int(inputs["retrieve_next_token_cpu"][index, node])
            if child != -1:
                todo.append((child, next_state, True))
    return out, last


def compare(ref_out, res_out, expected, last, ref_inputs, res_inputs):
    ref_mask, ref_grammar = ref_out
    res_mask, res_grammar = res_out
    if expected is None:
        assert ref_mask is None and res_mask is None
        assert ref_grammar is None and res_grammar is None
        return
    assert isinstance(res_mask, torch.Tensor)
    assert res_mask.dtype == ref_mask.dtype == torch.int32
    assert res_mask.shape == ref_mask.shape == expected.shape
    torch.testing.assert_close(res_mask, expected, rtol=0, atol=0)
    torch.testing.assert_close(ref_mask, expected, rtol=0, atol=0)
    assert ref_grammar is ref_inputs["reqs"][last].grammar
    assert res_grammar is res_inputs["reqs"][last].grammar
    for side in (ref_inputs, res_inputs):
        for req in side["reqs"]:
            if req.grammar is not None:
                assert req.grammar.stack == [0]
                assert req.grammar.accepts == req.grammar.rollbacks


def compute_ref_performance(**inputs):
    return measure_ms(compute_ref, inputs, torch.device("cpu"), repetitions=30)


def compute_res_performance(**inputs):
    return measure_ms(compute_res, inputs, torch.device("cpu"), repetitions=30)


@label("v1_sgl_spec_grammar_bitmask")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_sgl_spec_grammar_bitmask(case):
    inputs = input_build(case)
    expected, last = _oracle(inputs)
    ref_inputs = move_tree(clone_tree(inputs), torch.device("cpu"))
    res_inputs = move_tree(clone_tree(inputs), torch.device("cpu"))

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(
        ref_out, res_out, expected, last, ref_inputs, res_inputs
    )

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )


def _real_xgrammar(vocab):
    """Build SGLang's real xgrammar-backed grammar object over a tiny vocab."""
    import xgrammar as xg
    from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar

    info = xg.TokenizerInfo(list(vocab), xg.VocabType.RAW)
    compiler = xg.GrammarCompiler(info)
    compiled = compiler.compile_grammar(
        xg.Grammar.from_ebnf('root ::= "a" "b" | "c"')
    )
    matcher = xg.GrammarMatcher(compiled, max_rollback_tokens=16)
    return XGrammarGrammar(
        matcher, info.vocab_size, compiled, None
    ), info.vocab_size


def test_v1_sgl_spec_grammar_bitmask_real_xgrammar():
    """Exercise the real generate_token_bitmask DFS with an xgrammar matcher.

    The tiny RAW vocabulary makes the token strings explicit: id 1 = "a",
    id 2 = "b", id 3 = "c".  The grammar accepts "ab" or "c", so the root row
    must allow both "a" and "c" and the row after accepting "a" must allow
    only "b".  Terminal rows carry no mask bits and every accept is rolled
    back before returning.
    """
    pytest.importorskip("xgrammar")
    pytest.importorskip("sglang.srt.constrained.xgrammar_backend")

    vocab = ["<pad>"] + [chr(ord("a") + i) for i in range(26)] + [
        f"t{i}" for i in range(5, 64)
    ]
    a_id, b_id, c_id = 1, 2, 3

    def build_inputs():
        grammar, vocab_size = _real_xgrammar(vocab)
        return {
            "reqs": [Req(grammar)],
            "retrieve_next_token_cpu": torch.tensor(
                [[1, 2, -1, -1]], dtype=torch.int64
            ),
            "retrieve_next_sibling_cpu": torch.tensor(
                [[-1, 3, -1, -1]], dtype=torch.int64
            ),
            "draft_tokens_cpu": torch.tensor(
                [[0, a_id, b_id, c_id]], dtype=torch.int64
            ),
            "vocab_size": vocab_size,
        }

    ref_inputs = build_inputs()
    res_inputs = build_inputs()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    ref_mask, ref_grammar = ref_out
    res_mask, res_grammar = res_out
    assert isinstance(ref_mask, torch.Tensor)
    assert isinstance(res_mask, torch.Tensor)
    assert ref_mask.dtype == res_mask.dtype == torch.int32
    assert ref_mask.shape == res_mask.shape
    torch.testing.assert_close(ref_mask, res_mask, rtol=0, atol=0)
    assert ref_grammar is ref_inputs["reqs"][0].grammar
    assert res_grammar is res_inputs["reqs"][0].grammar

    def bits(mask, row, token):
        return bool(int(mask[row, token // 32]) & (1 << (token % 32)))

    for mask in (ref_mask, res_mask):
        # Root row: both "a" and "c" start a valid sentence.
        assert bits(mask, 0, a_id)
        assert bits(mask, 0, c_id)
        assert not bits(mask, 0, b_id)
        # After accepting "a" only "b" may follow.
        assert bits(mask, 1, b_id)
        assert not bits(mask, 1, a_id)
        assert not bits(mask, 1, c_id)
        # Rows 2 and 3 are terminal; traverse_tree writes no mask bits.
        assert int(mask[2].abs().sum()) == 0
        assert int(mask[3].abs().sum()) == 0

    for grammar in (ref_grammar, res_grammar):
        assert grammar.accepted_tokens == []
        assert grammar.is_terminated() is False
