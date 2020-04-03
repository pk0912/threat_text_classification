"""
Python script containing methods for building machine learning model
"""

import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GlobalMaxPool1D, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

import utils.text_processing as tp
from utils.helpers import logger, save_model_summary
from evaluation import evaluate


def build_model(seq_len, embed_layer, classes):
    input_ = Input(shape=(seq_len,))
    x = embed_layer(input_)
    x = Bidirectional(
        GRU(9, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)
    )(x)
    x = GlobalMaxPool1D()(x)
    output = Dense(len(classes), activation="sigmoid")(x)
    model = Model(input_, output)
    return model


def train_model(
    model,
    X,
    y,
    val_data,
    loss,
    optimizer,
    metrics,
    batch_size=32,
    epochs=10,
    early_stop_pat=10,
):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[EarlyStopping(patience=early_stop_pat)],
    )
    return history, model


def classifier(X, y, val_data, tokenizer, config):
    try:
        word_ind_dict = tokenizer.word_index
        glove_path = config.get("glove_path")
        vocab_size = config.get("vocab_size")
        seq_len = config.get("seq_len")
        embed_dim = config.get("embed_dim")
        classes = config.get("classes")
        outputs_path = config.get("outputs_path")
        objects_path = config.get("objects_path")
        model_name = config.get("model_name")
        loss = config.get("loss")
        optimizer = config.get("optimizer")
        metrics = config.get("metrics")
        batch_size = config.get("batch_size")
        epochs = config.get("epochs")
        early_stop_pat = config.get("early_stop_pat")
        num_words = min(vocab_size, len(word_ind_dict) + 1)
        embed_matrix = tp.get_embedding_matrix(
            glove_path, word_ind_dict, num_words, embed_dim, vocab_size
        )
        embed_layer = tp.get_embedding_layer(
            num_words, embed_dim, embed_matrix, seq_len
        )
        model = build_model(seq_len, embed_layer, classes)
        save_model_summary(model, os.path.join(outputs_path, "model_summary.txt"))
        history, model = train_model(
            model,
            X,
            y,
            val_data,
            loss,
            optimizer,
            metrics,
            batch_size,
            epochs,
            early_stop_pat,
        )
        evaluate(model, history, val_data[0], val_data[1], outputs_path)
        model.save(os.path.join(objects_path, model_name))
    except Exception as e:
        logger.error("Exception in classification : {}".format(str(e)))
        return False
    return True
