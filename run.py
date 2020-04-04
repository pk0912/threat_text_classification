"""
Entry point for training and predicting
"""

import os
import numpy as np
import pandas as pd
from joblib import dump, load
from tensorflow.keras.optimizers import Nadam

from utils.helpers import logger, save_csv_data
from utils.dataset_split import stratified_split
from raw_data_ingest import get_data
from preprocessing import preprocess
from vectorizer import vectorize
from classification import classifier
from config import (
    RAW_DATA_INGEST,
    SIMPLE_PROCESSING_TYPE,
    COMPLEX_PROCESSING_TYPE,
    VECTORIZE_DATA_SIMPLE,
    VECTORIZE_DATA_COMPLEX,
    SIMPLE_DATA_CLASSIFICATION,
    COMPLEX_DATA_CLASSIFICATION,
    GLOVE_PATH,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    EMBEDDING_DIM,
    CLASSES,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    LOSS,
    METRICS,
    MODEL_NAME,
    MODEL_SAVE_FORMAT,
)
from settings import (
    RAW_DATA_DIR,
    SIMPLE_PROCESSED_DATA_DIR,
    COMPLEX_PROCESSED_DATA_DIR,
    OBJECTS_DIR,
    TRAIN_DATA_DIR_WI,
    VAL_DATA_DIR_WI,
    OUTPUTS_DIR,
    VERSION,
)


def get_config():
    config_dict = dict()
    config_dict["glove_path"] = GLOVE_PATH
    config_dict["vocab_size"] = MAX_VOCAB_SIZE
    config_dict["seq_len"] = MAX_SEQUENCE_LENGTH
    config_dict["embed_dim"] = EMBEDDING_DIM
    config_dict["classes"] = CLASSES
    config_dict["outputs_path"] = OUTPUTS_DIR
    config_dict["objects_path"] = OBJECTS_DIR
    config_dict["model_name"] = MODEL_NAME + str(VERSION) + MODEL_SAVE_FORMAT
    config_dict["loss"] = LOSS
    config_dict["optimizer"] = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
    config_dict["metrics"] = [METRICS]
    config_dict["batch_size"] = BATCH_SIZE
    config_dict["epochs"] = EPOCHS
    config_dict["early_stop_pat"] = EARLY_STOP_PATIENCE
    return config_dict


def vectorize_data(data_path, processing_type):
    data = pd.read_csv(data_path, encoding="utf-8")
    tokenizer, vectors = vectorize(
        MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH, data["text"].values
    )
    if tokenizer is not None and vectors is not None:
        dump(tokenizer, os.path.join(OBJECTS_DIR, "tokenizer.joblib"))
        vec_df = pd.DataFrame(vectors)
        vec_df = pd.concat([vec_df, data.drop(columns=["text"])], axis=1)
        train_vec_data, val_vec_data = stratified_split(vec_df, split_col="threat")
        save_csv_data(
            train_vec_data,
            os.path.join(
                TRAIN_DATA_DIR_WI, "train_vectors_{}.csv".format(processing_type)
            ),
        )
        save_csv_data(
            val_vec_data,
            os.path.join(VAL_DATA_DIR_WI, "val_vectors_{}.csv".format(processing_type)),
        )
    else:
        logger.error("Error in vectorizing data!!!")


def perform_classification(data_type="complex"):
    train_data = pd.read_csv(
        os.path.join(TRAIN_DATA_DIR_WI, "train_vectors_{}.csv".format(data_type))
    )
    val_data = pd.read_csv(
        os.path.join(VAL_DATA_DIR_WI, "val_vectors_{}.csv".format(data_type))
    )
    configs = get_config()
    X = train_data[[str(i) for i in np.arange(0, MAX_SEQUENCE_LENGTH, 1)]].values
    y = train_data[CLASSES].values
    val_X = val_data[[str(i) for i in np.arange(0, MAX_SEQUENCE_LENGTH, 1)]].values
    val_y = val_data[CLASSES].values
    tokenizer = load(os.path.join(OBJECTS_DIR, "tokenizer.joblib"))
    clf_status = classifier(X, y, (val_X, val_y), tokenizer, configs)
    if not clf_status:
        logger.error("Error in performing classification!!!")


def main():
    logger.info("Execution Started!!!")
    if RAW_DATA_INGEST:
        logger.info("Getting data.")
        ingest_status = get_data()
        if not ingest_status:
            logger.error("Execution abruptly stopped while creating raw dataset!!!")
            return
    try:
        train_data = pd.read_csv(
            os.path.join(RAW_DATA_DIR, "train_data.csv"), encoding="utf-8"
        )
        if SIMPLE_PROCESSING_TYPE:
            logger.info("Performing simple text processing.")
            train_data_simple = preprocess(train_data)
            if type(train_data_simple) == pd.core.frame.DataFrame:
                train_data_simple.to_csv(
                    os.path.join(SIMPLE_PROCESSED_DATA_DIR, "train_data_simple.csv"),
                    index=False,
                    encoding="utf-8",
                )
            else:
                logger.error("Unable to write simple processed data!!!")
        if COMPLEX_PROCESSING_TYPE:
            logger.info("Performing complex text processing.")
            train_data_complex = preprocess(train_data, preprocess_type="complex")
            if type(train_data_complex) == pd.core.frame.DataFrame:
                train_data_complex.to_csv(
                    os.path.join(COMPLEX_PROCESSED_DATA_DIR, "train_data_complex.csv"),
                    index=False,
                    encoding="utf-8",
                )
            else:
                logger.error("Unable to write complex processed data!!!")
        if VECTORIZE_DATA_SIMPLE:
            vectorize_data(
                os.path.join(SIMPLE_PROCESSED_DATA_DIR, "train_data_simple.csv"),
                "simple",
            )
        if VECTORIZE_DATA_COMPLEX:
            vectorize_data(
                os.path.join(COMPLEX_PROCESSED_DATA_DIR, "train_data_complex.csv"),
                "complex",
            )
        if SIMPLE_DATA_CLASSIFICATION:
            perform_classification("simple")
        if COMPLEX_DATA_CLASSIFICATION:
            perform_classification("complex")
    except Exception as e:
        logger.error("Exception in main method : {}".format(str(e)))
        return
    logger.info("Execution successfully completed.")


if __name__ == "__main__":
    main()
