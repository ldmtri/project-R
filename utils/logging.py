import logging
from . import torch_distributed as u_dist


def init(file_path: str | None = None, force=False):
    """
    This function should be called before any other modules in this package.
    Q&A:
        - do I need to call this function when using 'hydra' with logging enabled ?
            --> No, hydra will handle it.
    """
    handlers = [logging.StreamHandler()]
    if file_path is not None:
        handlers.append(
            logging.FileHandler(file_path)  # type: ignore
        )
    di = u_dist.get_info()
    if di.world_size == 1:
        log_format = f"[%(asctime)s] %(levelname)s:%(message)s"
    else:
        log_format = f"[rank={di.rank}] [%(asctime)s] %(levelname)s:%(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_format, handlers=handlers, force=force
    )
