import json
import numpy as np
import os


def save_to_file(file_path, content, mode):
    file_dir = os.path.dirname(file_path)
    os.makedirs(file_dir, exist_ok=True)
    match mode:
        case "json":
            with open(file_path, "w") as f:
                json.dump(content, f, indent=4)
        case "numpy":
            np.save(file_path, content)
        case _:
            raise NotImplementedError()


def load_from_file(file_path, mode):
    assert os.path.exists(file_path)
    match mode:
        case "json":
            with open(file_path, "r") as f:
                return json.load(file_path)
        case "numpy":
            return np.load(file_path)
        case _:
            raise NotImplementedError()
