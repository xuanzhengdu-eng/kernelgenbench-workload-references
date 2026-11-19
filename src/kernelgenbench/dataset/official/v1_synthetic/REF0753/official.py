# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/watermark.py#L407-L441
from vllm.v1.worker.gpu.sample.watermark import draft_watermarking_mask as v1_vllm_draft_watermark_mask_triton

__all__ = ["v1_vllm_draft_watermark_mask_triton"]
