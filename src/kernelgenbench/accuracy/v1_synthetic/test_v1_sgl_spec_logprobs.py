"""Accuracy and performance for v1_sgl_spec_logprobs.

SGLang ``compute_spec_logprobs`` writes the accepted-token log-probabilities of a
speculative step onto ``logits_output`` and returns None.

Inputs: ``batch`` carries ``seq_lens`` and ``sampling_info`` (``is_all_greedy``
and the per-request ``temperatures``), ``logits_output.next_token_logits`` is
``(rows, vocab)`` in fp32/fp16/bf16, ``predict`` holds one token id per selected
row, and exactly one of ``accept_index`` (shape ``(batch, width)``, may contain
negative indices) or ``chain_stride`` selects the rows.

Key assertions: the written ``next_token_logprobs`` is ``(batch, width)`` and
equals an independent ``log_softmax(logits / temperature)`` gather (plain
``log_softmax`` when every request is greedy or ``original_logprob`` is set),
clamped to the dtype minimum, with exactly the same top-k and token-id fields;
``next_token_logits`` is never modified.
"""
import types

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import measure_cuda
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (mode, is_all_greedy, original_logprob, extra_fields, dtype)
COMBOS = [
    (mode, greedy, original, extra, dtype)
    for mode in ("accept", "chain")
    for greedy, original in ((True, False), (False, False), (False, True))
    for extra in (True, False)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]

FIELDS = ["next_token_logprobs", "next_token_top_logprobs_val", "next_token_top_logprobs_idx",
          "next_token_token_ids_logprobs_val", "next_token_token_ids_logprobs_idx"]


def input_build(config):
    mode, greedy, original, extra, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    # 6 rows = 2 requests x width 3; row 0's last logit is -inf on purpose.
    logits = torch.randn((6, 8), dtype=dtype, generator=generator)
    logits[0, 7] = float("-inf")
    accept = None
    chain = None
    if mode == "accept":
        # -1 repeats the previous row: the official entry keeps negative indexing.
        accept = torch.tensor([[0, 2, -1], [3, 4, 5]], dtype=torch.int64)
    else:
        chain = 3
    return {
        "batch": types.SimpleNamespace(
            seq_lens=[4, 7],
            sampling_info=types.SimpleNamespace(
                is_all_greedy=greedy,
                # (requests, 1) is broadcast over the vocabulary by the official entry.
                temperatures=torch.tensor([[1.5], [0.8]], dtype=torch.float32),
            ),
            top_logprobs_nums=[2, 0] if extra else [],
            token_ids_logprobs=[None, [1, 5]] if extra else [],
        ),
        "logits": logits,
        "predict": torch.tensor([2, 3, 1, 7, 0, 4], dtype=torch.int64),
        "accept_index": accept,
        "chain_stride": chain,
        "original_logprob": original,
    }


def compute_ref(batch, logits_output, predict, accept_index, chain_stride, original_logprob):
    return kernelgenbench.baseline.v1_sgl_spec_logprobs(
        batch, logits_output, predict, accept_index, chain_stride, original_logprob)


def compute_res(batch, logits_output, predict, accept_index, chain_stride, original_logprob):
    return kernelgenbench.solution.v1_sgl_spec_logprobs(
        batch, logits_output, predict, accept_index, chain_stride, original_logprob)


def _selected_rows(cpu, width):
    """Row ids and target tokens of each selected speculative position."""
    if cpu["accept_index"] is not None:
        rows = cpu["accept_index"].reshape(-1)
    else:
        rows = torch.arange(cpu["logits"].shape[0], dtype=torch.int64)
    if cpu["accept_index"] is not None:
        tokens = cpu["predict"][rows]
    else:
        tokens = cpu["predict"]
    return rows, tokens


def _expected_matrix(cpu, width):
    """Independent float64 log-probability matrix, one row per selected position."""
    logits = cpu["logits"].double()
    rows, tokens = _selected_rows(cpu, width)
    selected = logits[rows]
    if cpu["batch"].sampling_info.is_all_greedy or cpu["original_logprob"]:
        scores = selected
    else:
        temperatures = torch.repeat_interleave(
            cpu["batch"].sampling_info.temperatures.double(), width, dim=0)
        scores = selected / temperatures
    matrix = torch.log_softmax(scores, dim=-1)
    return matrix, tokens


def compare(ref_out, res_out):
    """Both sides return None; the payload is compared field by field."""
    assert ref_out is None and res_out is None


def _check_field(name, a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{name}: solution field is not a tensor"
        assert a.shape == b.shape and a.dtype == b.dtype, f"{name}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5, equal_nan=True, msg=name)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{name}: list length"
        for x, y in zip(a, b):
            _check_field(name, x, y)
    else:
        assert a == b, f"{name}: {a!r} vs {b!r}"


def _cuda_side(cpu, width):
    batch = types.SimpleNamespace(
        seq_lens=cpu["batch"].seq_lens,
        sampling_info=types.SimpleNamespace(
            is_all_greedy=cpu["batch"].sampling_info.is_all_greedy,
            temperatures=cpu["batch"].sampling_info.temperatures.cuda(),
        ),
        top_logprobs_nums=cpu["batch"].top_logprobs_nums,
        token_ids_logprobs=cpu["batch"].token_ids_logprobs,
    )
    output = types.SimpleNamespace(next_token_logits=cpu["logits"].cuda(),
                                   **{key: None for key in FIELDS})
    accept = None if cpu["accept_index"] is None else cpu["accept_index"].cuda()
    return dict(batch=batch, logits_output=output, predict=cpu["predict"].cuda(),
                accept_index=accept, chain_stride=cpu["chain_stride"],
                original_logprob=cpu["original_logprob"])


def compute_ref_performance(inputs):
    """Time the reference entry on the prepared inputs."""
    return measure_cuda(lambda: compute_ref(**inputs))


def compute_res_performance(inputs):
    """Time the solution entry on the same prepared inputs."""
    return measure_cuda(lambda: compute_res(**inputs))


@label("v1_sgl_spec_logprobs")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_spec_logprobs(config):
    mode, greedy, original, extra, dtype = config
    width = 3
    batch_rows = len(input_build(config)["batch"].seq_lens)
    cpu = input_build(config)

    ref_inputs = _cuda_side(cpu, width)
    res_inputs = _cuda_side(cpu, width)
    for side in (ref_inputs, res_inputs):
        assert side["logits_output"].next_token_logits.shape == (batch_rows * width, 8)
        assert side["logits_output"].next_token_logits.dtype == dtype
        assert (side["accept_index"] is None) != (side["chain_stride"] is None)
    ref_logits = ref_inputs["logits_output"].next_token_logits.clone()
    res_logits = res_inputs["logits_output"].next_token_logits.clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    tolerance = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 3e-2}[dtype]
    matrix, tokens = _expected_matrix(cpu, width)
    expected = matrix.gather(1, tokens.reshape(-1, 1)).reshape(batch_rows, width).cuda()
    for side, inputs in (("reference", ref_inputs), ("solution", res_inputs)):
        written = inputs["logits_output"].next_token_logprobs
        assert written.shape == (batch_rows, width), f"{side} shape {tuple(written.shape)}"
        # Plain log_softmax keeps the logits' dtype; dividing by the fp32
        # per-request temperatures promotes fp16/bf16 logits to fp32.
        expected_dtype = dtype if (greedy or original) else torch.float32
        assert written.dtype == expected_dtype, f"{side} dtype {written.dtype}"
        torch.testing.assert_close(written.double(), expected.double(), rtol=tolerance,
                                   atol=tolerance, msg=f"{side} next_token_logprobs")
        # The logits are read, never rewritten.
        torch.testing.assert_close(inputs["logits_output"].next_token_logits.cpu(),
                                   ref_logits.cpu(), rtol=0, atol=0,
                                   msg=f"{side} modified next_token_logits")
        if extra:
            values = inputs["logits_output"].next_token_top_logprobs_val
            indices = inputs["logits_output"].next_token_top_logprobs_idx
            assert values is not None and indices is not None
            assert len(values) == batch_rows * width
            # Request 1 asks for top-2 and request 2 for none (expanded per position).
            rows_cpu, _ = _selected_rows(cpu, width)
            rows_cuda = rows_cpu.cuda()
            selected32 = inputs["logits_output"].next_token_logits[rows_cuda].float()
            if not (cpu["batch"].sampling_info.is_all_greedy or cpu["original_logprob"]):
                temperatures = torch.repeat_interleave(
                    inputs["batch"].sampling_info.temperatures, width, dim=0)
                selected32 = selected32 / temperatures
            fp32_matrix = torch.log_softmax(selected32, dim=-1)
            for row in range(width):
                assert values[row].shape[0] == 2
                # The reported ids must select the reported values in this row's
                # fp32 log-probabilities, and the values must be the row's top two.
                ids = indices[row].long()
                got = torch.as_tensor(values[row], dtype=torch.float64, device="cuda")
                torch.testing.assert_close(fp32_matrix[row][ids].double(), got, rtol=tolerance,
                                           atol=tolerance,
                                           msg=f"{side} top-k ids row {row} do not select the values")
                expected_top = fp32_matrix[row].topk(2).values.double()
                torch.testing.assert_close(got, expected_top, rtol=tolerance,
                                           atol=tolerance, msg=f"{side} top-k values row {row}")
            for row in range(width, batch_rows * width):
                assert values[row].numel() == 0 and indices[row].numel() == 0
            token_vals = inputs["logits_output"].next_token_token_ids_logprobs_val
            token_ids = inputs["logits_output"].next_token_token_ids_logprobs_idx
            assert token_vals is not None and token_ids is not None
            assert len(token_vals) == batch_rows * width
            # Request 1 has no requested ids; request 2 asks for [1, 5].
            for row in range(width):
                assert token_vals[row].numel() == 0 and len(token_ids[row]) == 0
            for row in range(width, batch_rows * width):
                requested = torch.tensor([1, 5], dtype=torch.int64)
                torch.testing.assert_close(token_vals[row].double(),
                                           matrix[row][requested].cuda().double(),
                                           rtol=tolerance, atol=tolerance,
                                           msg=f"{side} requested-id logprobs row {row}")
                assert list(token_ids[row]) == [1, 5]
        else:
            assert inputs["logits_output"].next_token_top_logprobs_val is None
            assert inputs["logits_output"].next_token_token_ids_logprobs_val is None

    for name in FIELDS:
        _check_field(name, getattr(ref_inputs["logits_output"], name),
                     getattr(res_inputs["logits_output"], name))
    # Same underlying rows and tokens on both sides.
    assert torch.equal(ref_inputs["logits_output"].next_token_logits.cpu(),
                       res_inputs["logits_output"].next_token_logits.cpu())

    ref_time = compute_ref_performance(ref_inputs)
    res_time = compute_res_performance(res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
