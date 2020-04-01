"""
Python script containing methods for building machine learning model
"""

import utils.text_processing as tp


def classifier(X, y, tokenizer, config):
    word_ind_dict = tokenizer.word_index
    glove_path = config.get("glove_path")
    vocab_size = config.get("vocab_size")
    seq_len = config.get("seq_len")
    embed_dim = config.get("embed_dim")
    num_words = min(vocab_size, len(word_ind_dict) + 1)
    embed_matrix = tp.get_embedding_matrix(
        glove_path, word_ind_dict, num_words, embed_dim, vocab_size
    )
    embed_layer = tp.get_embedding_layer(num_words, embed_dim, embed_matrix, seq_len)
