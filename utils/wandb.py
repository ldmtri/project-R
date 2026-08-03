import wandb
import omegaconf
from contextlib import contextmanager

from . import torch_distributed as u_dist

def init(cfg):
    run = wandb.init(
        name=cfg.run_name,
        id=cfg.run_name,
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config = omegaconf.OmegaConf.to_container( # type: ignore
            cfg, resolve=True, throw_on_missing=True
        )
    )
    return run

@contextmanager
def run_wandb(cfg):
    run = None
    try:
        if cfg.run_wandb and u_dist.info.rank == 0:
            run = init(cfg)
        yield run
    finally:
        if run is not None:
            run.finish()
