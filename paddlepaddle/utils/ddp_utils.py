#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#


import paddle
import paddle.distributed as dist
from utils import logger


def is_master(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    return node_rank == 0


def dist_barrier():
    dist.barrier()


def is_start_rank_node(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    def_rank = getattr(opts, "ddp.start_rank", 0)
    return node_rank == def_rank


