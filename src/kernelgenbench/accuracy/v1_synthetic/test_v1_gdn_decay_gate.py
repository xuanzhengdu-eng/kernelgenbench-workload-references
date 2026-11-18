# Synthetic workload; no production capture or production frequency claim.
"""OP-27091145bf94, source sub-contract: Megatron GDN decay and update gates; A_log and dt_bias are float32 channel vectors, g is fp32 and beta sigmoid keeps beta dtype."""
import pytest
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from kernelgenbench.dataset.baseline.v1_synthetic.v1_gdn_decay_gate import v1_gdn_decay_gate
from kernelgenbench.accuracy.v1_synthetic.synthetic_inputs import build_case, check_output

# Synthetic shape parameter (B,T,H) is paired with each listed dtype.
# Actual tensor shapes: alpha=beta=(B,T,H), A_log_local_cp=dt_bias_local_cp=(H,).
COMBOS = [(shape, dtype) for shape in [(2, 16, 128), (2, 128, 1024)]
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]

def get_input(shape, dtype, device='cuda'):
    return build_case('gdn_decay', shape, dtype, device)

def input_build(shape, dtype):
    return get_input(shape, dtype)

def compute_ref(**inputs):
    return v1_gdn_decay_gate(**inputs)

def compute_res(**inputs):
    import kernelgenbench
    return kernelgenbench.triton.v1_gdn_decay_gate(**inputs)

COMBOS = [(shape, dtype) for shape in [(2, 16, 128), (2, 128, 1024)]
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


@label('v1_gdn_decay_gate')
@parametrize('config', COMBOS)
@pytest.mark.parametrize('config', COMBOS)
def test_accuracy_v1_gdn_decay_gate(config):
    shape, dtype = config
    inputs = get_input(shape, dtype)
    expected = compute_ref(**{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
    candidate_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    actual = compute_res(**candidate_inputs)
    check_output(expected, actual, dtype)
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and not torch.equal(inputs[key], candidate_inputs[key]):
            raise AssertionError(f'candidate modified input: {key}')
    import triton.testing
    ref_time = triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)
    res_time = triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float('inf'))
