import sys
import hydra
from omegaconf import OmegaConf
from . import torch_distributed as u_dist


def set_no_output():
    di = u_dist.get_info()
    no_output = [
        "hydra/job_logging=none",
        "hydra/hydra_logging=none",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ]
    sys.argv.extend(no_output)

def resolve_cfg(cfg):
    OmegaConf.resolve(cfg)
    return cfg

def to_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)

def to_yaml(cfg):
    return OmegaConf.to_yaml(cfg)
