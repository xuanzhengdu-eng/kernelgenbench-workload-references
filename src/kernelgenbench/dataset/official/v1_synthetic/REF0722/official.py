# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/distributed/parallel_state.py#L1254-L1311
from sglang.srt.distributed.parallel_state import GroupCoordinator

run = GroupCoordinator.reduce_scatterv
