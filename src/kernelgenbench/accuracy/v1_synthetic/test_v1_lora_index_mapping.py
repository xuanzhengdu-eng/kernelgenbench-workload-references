"""Accuracy and performance for v1_lora_index_mapping.

Operator: vLLM ``convert_mapping`` (LoRA id -> slot and sampler/embedding index
metadata).
Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/punica_wrapper/utils.py#L52-L148

Contract: ``mapping`` holds host tuples ``index_mapping`` (per token) and
``prompt_mapping`` (per request); ``lora_index_to_id`` maps slots to LoRA ids
with ``None`` holes.  Returns
``(base_indices [B], sampler_indices [R], sampler_indices_padded [R],
embeddings_indices [B], indices_len [4])``.  ``-1``/``0`` request ids mean "no
LoRA": base/sampler use -1, the embedding index uses 0, and the padded sampler
index uses ``max_loras - 1`` before adding ``row * R``.  Embedding indices are
multiplied by ``vocab_size``.  Every returned tensor is int64 on ``device``.
Host inputs are not modified.

Item: v1_lora_index_mapping (723-row 635).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import (
    fresh, measure_host, timings)

# index_mapping (tokens), prompt_mapping (requests), lora_index_to_id,
# max_loras, vocab_size.
CASES = [
    ((101, 0, 202, -1, 101), (202, 0, 101), (101, None, 202), 4, 32000),
    ((7, 7, 0), (7, 7), (7, None, 7), 3, 10),
    ((), (), (None, None), 2, 16),
]
COMBOS = [(case, device) for case in CASES for device in ("cpu", "cuda")]


def input_build(config):
    """Token ids B and prompt ids R are host lists; slots may contain None."""
    case, device = config
    ids, prompt, slots, limit, vocab = case
    return {
        "index_mapping": list(ids),
        "prompt_mapping": list(prompt),
        "lora_index_to_id": list(slots),
        "max_loras": limit,
        "vocab_size": vocab,
        "device": torch.device(device),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_lora_index_mapping(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_lora_index_mapping(**inputs)


def compare(ref_out, res_out, device, max_loras):
    assert isinstance(ref_out, tuple) and len(ref_out) == 5
    assert isinstance(res_out, tuple) and len(res_out) == 5
    assert res_out[4] == ref_out[4], "indices_len mismatch"
    for index, (left, right) in enumerate(zip(ref_out[:4], res_out[:4])):
        assert isinstance(right, torch.Tensor), f"output {index} is not a tensor"
        assert right.dtype == torch.int64 == left.dtype, f"output {index}: dtype"
        assert right.shape == left.shape, f"output {index}: shape"
        assert right.device.type == torch.device(device).type, f"output {index}: device"
        assert torch.equal(right.cpu(), left.cpu()), f"output {index}: values"
    assert res_out[0].storage_offset() == ref_out[0].storage_offset()
    # Padded sampler layout: row + slot * num_requests, with -1 -> max_loras - 1.
    sampler = res_out[1]
    expected = torch.arange(len(sampler), dtype=torch.long, device=sampler.device) + torch.where(
        sampler == -1, max_loras - 1, sampler) * len(sampler)
    assert torch.equal(res_out[2], expected), "padded sampler index mismatch"


def compute_ref_performance(**inputs):
    return measure_host(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure_host(compute_res, inputs, kernelgenbench.device)


@label("v1_lora_index_mapping")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_lora_index_mapping(config):
    device = config[1]
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    compare(compute_ref(**ref_inputs), compute_res(**res_inputs), device,
            inputs["max_loras"])
    for name in ("index_mapping", "prompt_mapping", "lora_index_to_id"):
        assert ref_inputs[name] == inputs[name], f"reference modified {name}"
        assert res_inputs[name] == inputs[name], f"candidate modified {name}"
    return timings(compute_ref_performance(**inputs),
                   compute_res_performance(**inputs))
