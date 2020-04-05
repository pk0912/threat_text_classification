"""
Python file to perform prediction using trained model.
"""

import os
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

import preprocessing as pp
from utils.text_processing import get_text_sequences
from utils.helpers import logger, Singleton
from config import (
    PREDICTION_DATA_TYPE,
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
    MODEL_SAVE_FORMAT,
    THRESHOLD,
)
from settings import OBJECTS_DIR, VERSION


class LoadObjects(metaclass=Singleton):
    def __init__(self):
        self.model = load_model(
            os.path.join(OBJECTS_DIR, MODEL_NAME + str(VERSION) + MODEL_SAVE_FORMAT)
        )
        self.tokenizer = load(os.path.join(OBJECTS_DIR, "tokenizer.joblib"))

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer


def get_results(data, tokenizer, model):
    vectored_data = get_text_sequences(tokenizer, data, MAX_SEQUENCE_LENGTH)
    probabilities = model.predict(vectored_data)
    predictions = np.where(probabilities > THRESHOLD, 1, 0)
    return probabilities, predictions


def predict(data):
    lo = LoadObjects()
    model = lo.get_model()
    tokenizer = lo.get_tokenizer()
    assert type(data) == list
    if len(data) == 0:
        logger.warning("List with 0 size has been passed!!!")
        return [], []
    try:
        data = pd.DataFrame(data, columns=["text"])
        if PREDICTION_DATA_TYPE == "simple":
            data["text"] = data["text"].map(pp.simple_processing)
        else:
            data["text"] = data["text"].map(pp.complex_processing)
        data = data["text"].values
        probabilities, predictions = get_results(data, tokenizer, model)
        return probabilities, predictions
    except Exception as e:
        logger.error("Exception while predicting : {}".format(str(e)))
        return [], []
