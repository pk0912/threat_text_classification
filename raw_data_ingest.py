"""
Python script to retrieve required raw data for classification task.
"""

import os
import pandas as pd
from utils.dataset_split import read_csv_data, stratified_split
from utils.helpers import logger, save_csv_data

from settings import RAW_DATA_DIR, KAGGLE_RAW_DATA_DIR, LUCKY_SEED


def get_data():
    try:
        raw_data = read_csv_data(os.path.join(KAGGLE_RAW_DATA_DIR, "train.csv"))
        raw_data = raw_data[["comment_text", "threat"]]
        threat_instances_count = raw_data["threat"].value_counts()[1] - 1
        data_threat = raw_data.loc[raw_data["threat"] == 1, :]
        data_non_threat = raw_data.loc[raw_data["threat"] == 0, :]
        data_non_threat = data_non_threat.sample(
            frac=1, random_state=LUCKY_SEED
        ).reset_index(drop=True)
        data_non_threat = data_non_threat.loc[:threat_instances_count, :]
        final_data = (
            pd.concat([data_threat, data_non_threat], axis=0)
            .sample(frac=1, random_state=LUCKY_SEED)
            .reset_index(drop=True)
        )
        train_data, test_data = stratified_split(
            final_data, split_col="threat", split_ratio=0.163
        )
        save_csv_data(train_data, os.path.join(RAW_DATA_DIR, "train_data.csv"))
        save_csv_data(test_data, os.path.join(RAW_DATA_DIR, "test_data.csv"))
        return True
    except Exception as e:
        logger.error("Exception in getting data : {}".format(str(e)))
        return False
