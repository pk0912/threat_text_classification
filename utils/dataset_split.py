"""
Script to divide data into training, validation and testing datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from settings import LUCKY_SEED
from .helpers import logger


def read_csv_data(path, header="infer"):
    return pd.read_csv(path, header=header, encoding="utf-8")


def stratified_split(data, split_col, n_splits=1, split_ratio=0.2):
    split = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=split_ratio, random_state=LUCKY_SEED
    )
    train_set = None
    test_set = None
    try:
        for train_index, test_index in split.split(data, data[split_col]):
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]
    except Exception as e:
        logger.error("Exception in Stratified split : {}".format(str(e)))
    return train_set, test_set


def save_to_multiple_csv_files(data, save_path, name_prefix, header, n_parts=10):
    path_format = os.path.join(save_path, "{}_{:02d}.csv")
    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths
