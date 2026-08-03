import hydra
import logging
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW

from utils import u_dist, u_hydra, u_wandb, u_torch
from build_sudoku_dataset import load_data


class InfiniteDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.epoch = 0

        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.epoch)

        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1

            if hasattr(self.loader.sampler, "set_epoch"):
                self.loader.sampler.set_epoch(self.epoch)

            self.iterator = iter(self.loader)
            return next(self.iterator)

    def __iter__(self):
        return self


def prepare_loaders(cfg, features):
    train_loader = u_dist.prepare_dataloader(
        dataset=features["train"],
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = u_dist.prepare_dataloader(
        dataset=features["test"],
        batch_size=cfg.tester.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader


def prepare_opt_sched(model, cfg):
    opt_kwargs = cfg.optimizer_cfg
    opt = AdamW(model.parameters(), lr=cfg.lr, **opt_kwargs)
    num_warmups = (
        cfg.warmup_steps
        if cfg.warmup_steps > 1
        else cfg.total_steps * float(cfg.warmup_steps)
    )
    logging.info(f"warmup steps/total steps: {num_warmups}/{cfg.total_steps} .")
    sched = u_torch.get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=num_warmups, num_training_steps=cfg.total_steps
    )
    return opt, sched


@hydra.main(config_path=".", config_name="config", version_base=None)
def launch(cfg):
    cfg = u_hydra.resolve_cfg(cfg)
    if u_dist.info.rank == 0 and cfg.use_wandb:
        u_wandb.init(cfg)
    cfg = u_dist.broadcast_object(cfg, src=0)
    logging.info("=" * 10 + "\n" + u_hydra.to_yaml(cfg))
    logging.info("=" * 10 + "\n")
    torch.random.manual_seed(cfg.seed + u_dist.info.rank)
    features = load_data(cfg.dataset)

    train_loader, test_loader = prepare_loaders(cfg, features)
    train_loader = InfiniteDataLoader(train_loader)
    model = nn.Linear(3, 3).to(u_dist.info.device)
    model = u_dist.prepare_model(model)
    otp, sched = prepare_opt_sched(model, cfg.trainer)

    cur_step = 0
    while cur_step < cfg.trainer.total_steps:
        batch_data = next(train_loader)
        print(batch_data["input"])
        break

    # clean
    if u_dist.info.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    if u_dist.info.rank != 0:
        u_hydra.set_no_output()
    with u_dist.distributed() as di:
        launch()
