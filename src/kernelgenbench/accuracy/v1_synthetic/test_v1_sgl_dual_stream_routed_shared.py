# Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_v2.py#L1173-L1346
import time
import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_dual_stream_routed_shared(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_dual_stream_routed_shared(**inputs)


def _time(fn, args):
    if any(isinstance(v, torch.Tensor) and v.is_cuda for v in args.values()):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        fn(**args)
    if any(isinstance(v, torch.Tensor) and v.is_cuda for v in args.values()):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 3


def _timing(args, actual):
    a, b = _time(compute_ref, args), _time(compute_res, actual)
    return CustomBenchmarkResult(ref_time=a, res_time=b, speedup=a / b if b else float("inf"))


COMBOS = [(1, 1, False), (3, 2, True), (7, 2, False), (4, 3, True)]


def input_build(config, model_config):
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    num_tokens, _, _ = config
    with torch.device("cuda"):
        layer = DeepseekV2MoE(model_config, 0, alt_stream=torch.cuda.Stream(), reduce_results=False)
    # The shared expert reads the same hidden-state buffer on another CUDA stream.
    layer.experts.moe_runner_config.inplace = False
    generator = torch.Generator(device="cuda").manual_seed(702)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(0.0, 0.07, generator=generator)
    x = torch.randn(num_tokens, 64, device="cuda", generator=generator)
    return dict(layer=layer, hidden_states=x)


@label("v1_sgl_dual_stream_routed_shared")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dual_stream_routed_shared(config):
    import tempfile
    from transformers import DeepseekV2Config
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.runtime_context import create_context, use_context, SpawnRanks
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment, initialize_model_parallel,
        destroy_model_parallel, destroy_distributed_environment,
    )
    token_count, k, renormalize = config
    model_config = DeepseekV2Config(hidden_size=64, moe_intermediate_size=64, n_routed_experts=4,
                                    n_shared_experts=1, num_experts_per_tok=k, n_group=1, topk_group=1,
                                    norm_topk_prob=renormalize, routed_scaling_factor=1., hidden_act="silu",
                                    scoring_func="softmax", topk_method="greedy")
    model_config.architectures = ["DeepseekV2ForCausalLM"]
    with tempfile.TemporaryDirectory() as directory:
        model_config.save_pretrained(directory)
        server_args = ServerArgs(model_path=directory, device="cuda", dtype="float32",
                                 skip_tokenizer_init=True, disable_cuda_graph=True)
        context = create_context(server_args, role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        with use_context(context), torch.no_grad():
            own_group = not torch.distributed.is_initialized()
            if own_group:
                init_distributed_environment(1, 0, "file://" + directory + "/nccl", 0, backend="nccl")
                initialize_model_parallel()
            try:
                args = input_build(config, model_config)
                x = args["hidden_states"]
                before = x.clone()
                saved = [p.clone() for p in args["layer"].parameters()]
                reference = compute_ref(**args)
                out = compute_res(**args)
                assert isinstance(out, torch.Tensor) and out.shape == (token_count, 64)
                assert out.dtype == torch.float32 and not torch._C._is_alias_of(out, x)
                torch.testing.assert_close(out, reference, atol=4e-4, rtol=1e-3)
                assert torch.equal(x, before)
                assert all(torch.equal(v, p) for v, p in zip(saved, args["layer"].parameters()))
                return _timing(args, args)
            finally:
                if own_group:
                    destroy_model_parallel()
                    destroy_distributed_environment()
