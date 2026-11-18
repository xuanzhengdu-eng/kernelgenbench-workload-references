# Synthetic workload; no production capture or production frequency claim.
"""OP-c4159a95fc8b: Megatron inference MoE unpermute_tokens.

Inputs: expert_output is [output_size, hidden], permuted_probs is fp32
[output_size] and permutation_map is int32 [output_size] with -1 padding rows.
Several rows may map to one token, so the kernel accumulates them. The output is
a fresh fp32 [num_tokens, hidden] tensor. A scalar Python loop rebuilds the
atomic accumulation, and untouched tokens must stay exactly zero.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# num_tokens, output_size, hidden, dtype, padding rows
COMBOS = [
    (6, 12, 16, torch.bfloat16, True),
    (8, 8, 32, torch.float16, False),
    (5, 10, 8, torch.float32, True),
    (1, 1, 4, torch.bfloat16, False),
]


def input_build(config):
    num_tokens, output_size, hidden, dtype, padding = config
    generator = torch.Generator(device="cpu").manual_seed(871)
    expert_output = torch.randn((output_size, hidden), dtype=dtype, generator=generator)
    permuted_probs = torch.rand((output_size,), generator=generator)
    rows = torch.arange(output_size, dtype=torch.int64)
    mapping = ((rows * 7 + 3) % num_tokens).to(torch.int32)
    if padding:
        mapping[::5] = -1
    return {"expert_output": expert_output, "permuted_probs": permuted_probs,
            "permutation_map": mapping, "num_tokens": num_tokens}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_inference_moe_unpermute(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_inference_moe_unpermute(**inputs)


def _tol(dtype):
    return {torch.float32: (2e-6, 2e-6), torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (8e-3, 8e-3)}[dtype]


def _oracle(expert_output, probs, mapping, num_tokens, hidden):
    expected = torch.zeros((num_tokens, hidden), dtype=torch.float64)
    for row in range(mapping.numel()):
        token = int(mapping[row])
        if token >= 0:
            expected[token] += expert_output[row].double() * float(probs[row])
    return expected


def compare(ref_out, res_out, dtype, expert_output, probs, mapping, num_tokens, hidden):
    assert torch.is_tensor(res_out)
    assert res_out.shape == ref_out.shape == (num_tokens, hidden)
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.device == ref_out.device
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
    expected = _oracle(expert_output.cpu(), probs.cpu(), mapping.cpu(), num_tokens, hidden)
    torch.testing.assert_close(res_out.cpu(), expected.float(), rtol=rtol, atol=atol)
    targeted = set(int(x) for x in mapping.cpu() if int(x) >= 0)
    untouched = [t for t in range(num_tokens) if t not in targeted]
    if untouched:
        assert torch.count_nonzero(res_out[untouched]) == 0, "untargeted tokens must stay zero"


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_megatron_inference_moe_unpermute")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_inference_moe_unpermute(config):
    num_tokens, hidden, dtype = config[0], config[2], config[3]
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = {name: value.clone() for name, value in res_inputs.items()
              if torch.is_tensor(value)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, res_inputs["expert_output"],
            res_inputs["permuted_probs"], res_inputs["permutation_map"],
            num_tokens, hidden)
    for name, value in before.items():
        torch.testing.assert_close(res_inputs[name], value, rtol=0, atol=0)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
