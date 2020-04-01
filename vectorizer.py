"""
Python file containing scripts to convert processed text into vectors
"""

from utils import text_processing as tp
from utils.helpers import logger


def vectorize(vocab_size, seq_len, data):
    try:
        tokenizer = tp.get_tokenizer_object(data, num_words=vocab_size)
        vectors = tp.get_text_sequences(tokenizer, data, seq_len=seq_len)
        return tokenizer, vectors
    except Exception as e:
        logger.error("Exception in vectorize function : {}".format(str(e)))
        return None, None
