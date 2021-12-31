import ast
import logging
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


is_kaggle_env = os.path.exists("/kaggle")


def log_and_maybe_print(s):
    logging.info(s)
    if is_kaggle_env:
        print(s)


class PCLTransformersDataset(Dataset):
    def __init__(self, **kwargs):
        self.valid_attrs = []
        self.num_examples = 0
        for attr, values in kwargs.items():
            self.valid_attrs.append(attr)
            setattr(self, attr, values)
            self.num_examples = len(values)

    def __getitem__(self, item):
        return {k: getattr(self, k)[item] for k in self.valid_attrs}

    def __len__(self):
        return self.num_examples


def load_binary_dataset(path):
    _df = pd.read_csv(path, sep="\t")
    if "proba_binary_label" in _df.columns:
        _df["proba_binary_label"] = _df["proba_binary_label"].apply(ast.literal_eval)

    return _df


def train_dev_test_split(data):
    num_examples = data.shape[0]
    indices = np.random.permutation(num_examples)

    # 80/10/10 split
    train_indices = indices[:int(0.8 * num_examples)]
    dev_indices = indices[int(0.8 * num_examples): int(0.9 * num_examples)]
    test_indices = indices[int(0.9 * num_examples):]

    return {
        "train": data.iloc[train_indices].reset_index(drop=True),
        "dev": data.iloc[dev_indices].reset_index(drop=True),
        "test": data.iloc[test_indices].reset_index(drop=True)
    }


if __name__ == "__main__":
    df = load_binary_dataset("/home/matej/Documents/multiview-pcl-detection/data/interim/binary_pcl.tsv")
    split = train_dev_test_split(df)
