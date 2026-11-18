"""Accuracy and performance for v1_vllm_rejection_full (vLLM classic rejection).

Each case packs several requests into one flat draft-token tensor, with request
``i`` contributing ``counts[i]`` draft tokens.  Target logits are ``[tokens, vocab]``
and the bonus token per request is appended when the whole draft is accepted.

Key assertions: int32 output of shape ``[batch, max_spec_len+1]`` with ``-1``
padding; exact token equality with the fixed-commit kernels (both sides receive
identical generators and seed, so the uniform and exponential draws line up);
greedy rows stop at the first mismatch and keep the target argmax; a request with
no draft tokens returns only its bonus token; and every input tensor and
generator state is left untouched.

Reference: Torch implementation of the greedy/random/mixed kernels.
Solution: vLLM's ``rejection_sample`` driven by ``SpecDecodeMetadata`` and
``SamplingMetadata``.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.row_pair_utils import fresh, measure_cuda, to_device
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


CASES = [
    # (mode, counts, vocab, target dtype, use draft probs, greedy mask,
    #  generator request ids, fp64 gumbel, seed)
    ("greedy", (0, 3, 2), 11, torch.float32, None, None, (), False, 0),
    ("greedy", (4, 2, 0, 3), 17, torch.float16, None, None, (), False, 0),
    ("greedy", (1, 1, 1, 1), 5, torch.bfloat16, None, None, (), False, 0),
    ("greedy", (0, 0, 0), 9, torch.float32, None, None, (), False, 0),
    ("greedy", (5, 3), 1031, torch.float32, None, None, (), False, 0),
    ("random", (3, 2, 1), 13, torch.float32, False, None, (1,), False, 17),
    ("random", (2, 0, 3), 17, torch.float32, True, None, (0, 2), True, 931),
    ("random", (1, 3, 2, 0), 5, torch.float32, True, None, (0, 1, 2, 3), False, 123),
    ("random", (4, 1, 2), 1031, torch.float32, True, None, (0, 1, 2), True, 817),
    ("mixed", (0, 2, 3, 1), 13, torch.float32, True, (True, False, True, False),
     (0, 1, 2), False, 301),
    ("mixed", (3, 1, 2), 17, torch.float32, False, (False, True, False), (0, 1),
     True, 412),
    ("mixed", (2, 0, 4), 7, torch.float32, True, (True, False, False), (1, 2),
     True, 512),
    ("mixed", (4, 1, 3, 2), 1031, torch.float32, True, (False, True, False, True),
     (0, 2, 3), False, 612),
]
COMBOS = CASES


def input_build(config):
    mode, counts, vocab, dtype, use_draft_probs, greedy_mask, generator_ids, fp64, seed = config
    total = sum(counts)
    generator = torch.Generator(device="cpu").manual_seed(700 + seed)
    draft, target = [], []
    for request, count in enumerate(counts):
        for position in range(count):
            token = (request * 7 + position * 3 + 1) % vocab
            target.append(token)
            # Sparse mismatches: every third request differs at its first draft
            # token, so greedy rows must truncate there.
            mismatch = request % 3 == 1 and position == 0
            draft.append((token + 1) % vocab if mismatch else token)
    logits = torch.full((total, vocab), -3.0, dtype=dtype)
    if total:
        logits[torch.arange(total), torch.tensor(target)] = 4.0 if mode == "greedy" else 0.3
    cumulative = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)
    probs = None
    if use_draft_probs:
        raw = torch.arange(total * vocab, dtype=torch.float32).reshape(total, vocab)
        probs = ((raw * 7) % 19 + 1)
        probs = probs / probs.sum(-1, keepdim=True)
        if total > 1 and draft and draft[0] >= 0:
            probs[0, draft[0]] = 0.0
            probs[0] /= probs[0].sum()
    inputs = {
        "draft_token_ids": torch.tensor(draft, dtype=torch.int32),
        "num_draft_tokens": list(counts),
        "max_spec_len": max(counts),
        "cu_num_draft_tokens": torch.tensor(cumulative, dtype=torch.int32),
        "target_logits": logits,
        "bonus_token_ids": torch.tensor([[(i * 5 + 2) % vocab] for i in range(len(counts))],
                                        dtype=torch.int64),
        "mode": mode,
        "draft_probs": probs,
        "is_greedy": torch.tensor(greedy_mask, dtype=torch.bool) if greedy_mask else None,
        "generator_seeds": {request: 10000 + request * 97 for request in generator_ids},
        "use_fp64_gumbel": fp64,
        "seed": seed,
    }
    return inputs


def _generators(seeds, device):
    return {request: torch.Generator(device=device).manual_seed(seed)
            for request, seed in seeds.items()}


def compute_ref(seed, generator_seeds, **inputs):
    torch.manual_seed(seed)
    generators = _generators(generator_seeds, inputs["target_logits"].device)
    result = kernelgenbench.baseline.v1_vllm_rejection_full(generators=generators, **inputs)
    return result, {key: gen.get_state() for key, gen in generators.items()}


def compute_res(seed, generator_seeds, **inputs):
    torch.manual_seed(seed)
    generators = _generators(generator_seeds, inputs["target_logits"].device)
    result = kernelgenbench.solution.v1_vllm_rejection_full(generators=generators, **inputs)
    return result, {key: gen.get_state() for key, gen in generators.items()}


def compare(ref_out, res_out, inputs):
    ref_tokens, ref_states = ref_out
    res_tokens, res_states = res_out
    assert isinstance(res_tokens, torch.Tensor)
    assert res_tokens.dtype == ref_tokens.dtype == torch.int32
    assert res_tokens.shape == ref_tokens.shape == (
        len(inputs["num_draft_tokens"]), inputs["max_spec_len"] + 1)
    assert torch.equal(res_tokens, ref_tokens), "accepted token rows differ"
    # Padding beyond a request's emitted tokens stays -1.
    for request, count in enumerate(inputs["num_draft_tokens"]):
        row = res_tokens[request]
        if count == 0:
            assert int(row[0]) >= 0 and torch.equal(row[1:], torch.full_like(row[1:], -1))
        else:
            assert torch.equal(row[count + 1:], torch.full_like(row[count + 1:], -1))
    # The reference consumes exactly the same per-request generator draws.
    for key in ref_states:
        assert torch.equal(ref_states[key], res_states[key]), f"generator {key} state"


def compute_ref_performance(inputs):
    return measure_cuda(lambda **kw: compute_ref(**kw)[0], inputs)


def compute_res_performance(inputs):
    return measure_cuda(lambda **kw: compute_res(**kw)[0], inputs)


@label("v1_vllm_rejection_full")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_rejection_full(config):
    mode, counts, vocab, dtype, use_draft_probs, greedy_mask, generator_ids, fp64, seed = config
    inputs = to_device(input_build(config))
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    assert ref_inputs["target_logits"].shape == (sum(counts), vocab)
    assert ref_inputs["cu_num_draft_tokens"].tolist()[-1:] == [sum(counts)]
    assert ref_inputs["draft_token_ids"].dtype == torch.int32

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, inputs)

    if mode in ("greedy", "mixed"):
        # Greedy rows truncate at the first mismatch: at most one target token
        # beyond the accepted prefix comes from the argmax comparison.
        for request, count in enumerate(counts):
            is_greedy = mode == "greedy" or bool(greedy_mask[request])
            if not is_greedy or count == 0:
                continue
            if request % 3 == 1:
                assert torch.equal(
                    res_out[0][request, 1:],
                    torch.full_like(res_out[0][request, 1:], -1))
    for name in ("draft_token_ids", "target_logits", "bonus_token_ids",
                 "cu_num_draft_tokens"):
        torch.testing.assert_close(res_inputs[name], inputs[name], rtol=0, atol=0)
    if use_draft_probs:
        torch.testing.assert_close(res_inputs["draft_probs"], inputs["draft_probs"],
                                   rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
