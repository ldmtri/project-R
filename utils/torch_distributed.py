from torch.distributed.optim import ZeroRedundancyOptimizer
import os
from torch import distributed as dist
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from contextlib import contextmanager
from pydantic import BaseModel, ConfigDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

"""
torch_distributed.py

This module provides simple wrapper for torch distributed application.
It is in the early stage, with only wrapper tested for a single node,
multiple devices.

For more detail about usage, checkout _test function in this module file.

To run distributed script, run with `torchrun`. Some arguments:
    –standalone: used when run on a single node.
    –nnodes: number of nodes used.
    –nproc-per-node: is the number of gpu per node, `gpu` for using all GPUs.
    –redirects 3 redirects the stdout & stderr into files
    –log-dir ../logs: configure the log directory.

use '@record' to save errors of workers into file.
"""


class _DistInfo(BaseModel):
    # Enable arbitrary types for this specific model, fix torch.device
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rank: int
    local_rank: int
    world_size: int
    device: torch.device


dist_info: _DistInfo | None = None


def get_info():
    global dist_info
    if dist_info is None:
        has_cuda = torch.cuda.is_available()
        rank = int(os.getenv("RANK", "0"))
        n_local_devices = max(torch.cuda.device_count(), 1)
        local_rank = rank % n_local_devices
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        device = torch.device(f"cuda:{local_rank}") if has_cuda else torch.device("cpu")
        dist_info = _DistInfo(
            rank=rank, local_rank=local_rank, world_size=world_size, device=device
        )
    return dist_info

@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()

@contextmanager
def distributed():
    dist_info = _init_distribution()
    try:
        yield dist_info
    finally:
        _destroy_dist()


def preapre_model(model: torch.nn.Module, local_rank: int):
    if dist.is_initialized():
        return DistributedDataParallel(model, device_ids=[local_rank])
    return model

def broadcast_object(object, src):
    di = get_info()
    l = [None]
    if di.rank == src:
        l = [object]

    if di.world_size > 1:
        dist.broadcast_object_list(l, src=src)

    assert l[0] is not None
    return l[0]


# Replicate DataLoader common args so LSP can suggest it.
def prepare_dataloader(
    *,
    dataset,
    batch_size: int | None = 1,
    collate_fn=None,
    shuffle: bool | None = None,
    drop_last: bool = False,
    **kwargs,
):
    sampler = None
    if dist.is_initialized():
        assert shuffle is not None
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # fallback to its default value of DataLoader,
        # so the effect is similar as not defining.
        shuffle = None
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs,
    )


def zero_optimizer(
    model: torch.nn.Module, optim_class: type[torch.optim.Optimizer], **kwargs
):
    """
    https://arxiv.org/abs/1910.02054
    This wrapper is on the early stage, as I haven't had any expericencing using it.
    Q&A:
        - Where to define other optimizer's arguments ?
            --> in kwargs

    """
    from torch.distributed.optim import ZeroRedundancyOptimizer

    if dist.is_initialized():
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=optim_class, **kwargs
        )
    else:
        return optim_class(model.parameters(), **kwargs)

    return optimizer


def _init_distribution():
    has_cuda = torch.cuda.is_available()
    di = get_info()
    if has_cuda:
        torch.cuda.set_device(di.device)
        dist.init_process_group(
            rank=di.rank, world_size=di.world_size, device_id=di.device
        )
        logging.info(
            f"Initialize distributed successfully with WORLD_SIZE={di.world_size}."
        )
    else:
        logging.info(
            f"Attempt to initialize distributed: no GPU found, fallback to cpu."
        )
    return di


def _destroy_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


# TEST:
@record
def _test():
    from torch.utils.data import TensorDataset
    from torch.optim import AdamW

    data = TensorDataset(torch.rand(100, 250, 250))  # n_samples, ...

    try:
        from logger import init_logging  # type: ignore

        init_logging()
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format=f"[rank={get_info().rank}] [%(asctime)s] %(levelname)s:%(message)s",
            handlers=[logging.StreamHandler()],
        )

    with distributed() as di:
        logging.info(str(di))
        model = torch.nn.Linear(3, 3).to(di.device)
        model = preapre_model(model, di.local_rank)
        print(model)
        print(type(model))
        loader = prepare_dataloader(
            dataset=data,
            batch_size=5,
            shuffle=True,
            drop_last=False,
            num_workers=2,
            pin_memory_device=di.device,
        )
        print(vars(loader))
        for i in loader:
            print(i[0][0, 0])
            break

        zero_opt = zero_optimizer(model, AdamW, lr=5e-3, betas=(0.9, 0.98), eps=1e-7)
        if dist.is_initialized():
            print(vars(zero_opt.optim))  # type: ignore
        else:
            print(vars(zero_opt))




if __name__ == "__main__":
    _test()
