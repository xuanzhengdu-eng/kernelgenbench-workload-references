# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/text_generation_controllers/text_generation_controller.py#L272-L359
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController as _TextGenerationController,
)

run = _TextGenerationController._torch_sampling_func
