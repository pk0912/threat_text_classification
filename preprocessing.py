"""
Python script for data pre-processing
"""

import re

import utils.text_processing as tp
from utils.helpers import logger


def remove_referenced_name(text):
    pattern = r"@[^\s\.\?,;:!]*"
    return re.sub(pattern, " ", text).strip()


def simple_processing(text):
    text = tp.unicode_normalize(text)
    text = tp.general_regex(text)
    text = remove_referenced_name(text)
    return text


def complex_processing(text):
    text = tp.unicode_normalize(text)
    text = tp.lowercasing(text)
    text = tp.general_regex(text)
    text = remove_referenced_name(text)
    text = tp.get_decontracted_form(text)
    text = tp.keep_alpha_space(text)
    text = tp.remove_repeating_chars(text)
    text = tp.remove_stopwords(text)
    if text != "":
        text = tp.perform_lemmatization(text)
    return text


def preprocess(data, preprocess_type="simple"):
    try:
        if preprocess_type == "simple":
            data["text"] = data["comment_text"].map(simple_processing)
        else:
            data["text"] = data["comment_text"].map(complex_processing)
        data = data.drop(columns=["id", "comment_text"])
        data = data.dropna()
        data = data.drop(data.loc[data["text"] == ""].index)
        return data.reset_index(drop=True)
    except Exception as e:
        logger.error("Exception in pre-processing : {}".format(str(e)))
    return None
