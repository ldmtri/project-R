import sys
import os
import csv
import numpy as np
import hydra
import json
from hashlib import sha1
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import logging

sys.path.append(".")

from utils import save_to_file, init_logging  # type: ignore


class DataProcessConfig(BaseModel):
    source_repo: str
    output_dir: str
    final_output_dir: str | None = None

    subsample_size: int | None
    min_difficulty: int | None
    num_aug: int

    def model_post_init(self, __context):
        # Use hash of config file as final output directory.
        if self.final_output_dir is None:
            self.final_output_dir = os.path.join(self.output_dir, self.dataset_id())
        os.makedirs(self.final_output_dir, exist_ok=True)

    def dataset_id(self) -> str:
        payload = json.dumps(
            self.model_dump(exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha1(payload.encode()).hexdigest()[:8]


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged.
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))

    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []

    with open(
        hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"),
        newline="",
    ) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (
                int(rating) >= config.min_difficulty
            ):
                assert len(q) == 81 and len(a) == 81

                zero = ord("0")
                question = (
                    np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(
                        9, 9
                    )
                    - zero
                )
                answer = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - zero
                inputs.append(question)
                labels.append(answer)
    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(
                total_samples, size=config.subsample_size, replace=False
            )
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {
        k: list()
        for k in [
            "inputs",
            "labels",
            "puzzle_indices",
        ]
    }
    example_id = 0

    results["puzzle_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            results["inputs"].append(inp)  # type: ignore
            results["labels"].append(out)  # type: ignore
            example_id += 1

            results["puzzle_indices"].append(example_id)

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)

        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1  # TODO: ? why +1, remove it if not necessary.

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
    }

    assert config.final_output_dir is not None
    save_dir = os.path.join(config.final_output_dir, set_name)

    # Save data
    logging.info(f"Saving processed data to: {save_dir}")
    for k, v in results.items():
        save_to_file(os.path.join(save_dir, f"all__{k}.npy"), v, mode="numpy")
    logging.info("Done")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="build-data",
)
def preprocess_data(cfg):
    cfg = DataProcessConfig(**cfg.dataset)
    init_logging(os.path.join(cfg.final_output_dir, "build_data.log"))  # type: ignore
    convert_subset("train", cfg)
    convert_subset("test", cfg)

    # Save metadata
    metadata = cfg.model_dump(mode="json")
    save_to_file(
        os.path.join(cfg.final_output_dir, "metadata.json"),  # type: ignore
        metadata,
        mode="json",
    )


if __name__ == "__main__":
    preprocess_data()
