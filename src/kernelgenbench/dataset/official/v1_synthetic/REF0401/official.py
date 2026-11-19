# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/muon.py#L137-L188
# https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L30-L64
# https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon_utils.py#L114-L207
# https://github.com/NVIDIA-NeMo/Emerging-Optimizers/blob/1effa026ff096b7fa1063ca2fba19d98be6e6cdf/emerging_optimizers/orthogonalized_optimizers/muon.py#L136-L162
from megatron.core.optimizer.muon import TensorParallelMuon

orthogonalize = TensorParallelMuon.orthogonalize
