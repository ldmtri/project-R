import logging


def init_logging(file_path=None):
    handlers = [logging.StreamHandler()]
    if file_path is not None:
        handlers.append(
            logging.FileHandler(file_path)  # type: ignore
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", handlers=handlers
    )
