import hydra
import logging
import torch

from utils import (
    u_dist,
    u_hydra,
    save_to_file
)


@hydra.main(config_path="config", config_name="train", version_base=None)
def launch(cfg):
    with u_dist.distributed() as di:
        cfg = u_hydra.resolve_cfg(cfg)
        cfg = u_dist.broadcast_object(cfg, src=0)
        logging.info("=" * 10 + "\n" + u_hydra.to_yaml(cfg))
        logging.info("=" * 10)
        torch.random.manual_seed(cfg.seed + di.rank)



if __name__ == "__main__":
    if u_dist.get_info().rank != 0:
        u_hydra.set_no_output()
    launch()
