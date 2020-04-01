"""
Python file containing methods that are useful in performing text processing
"""

import re
import unicodedata
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

from .ml import nlp, STOPWORDS
from .words import DATE_STOPWORDS, DIR_STOPWORDS, NUM_STOPWORDS, REL_STOPWORDS

STOPWORDS = STOPWORDS.union(DATE_STOPWORDS, DIR_STOPWORDS, NUM_STOPWORDS, REL_STOPWORDS)
ip_pattern = r"((((([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5]))([^0-9]|$))|(((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?([^A-Za-z0-9]|$)))"
yyyy_date_pattern = r"([1-2][0-9]{3}[-\.\/])((([1-9]|1[0-2]|0[1-9])[-\.\/]([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[^0-9])|(([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[-\.\/]([1-9]|0[1-9]|1[0-2])[^0-9]))"
m_d_y_data_pattern = r"[\s]?((([1-9]|0[1-9]|1[0-2])[-\.\/]([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1]))|(([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[-\.\/]([1-9]|1[0-2]|0[1-9])))[-\.\/]([1-2][0-9]{3}|[0-9]{2})"
timestamp_pattern_1 = r"([1-2][0-9]{3}[-\.\/])((([1-9]|1[0-2]|0[1-9])[-\.\/]([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[^0-9])|(([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[-\.\/]([1-9]|0[1-9]|1[0-2])))[\s]([0-9]|[0-1][0-9]|2[0-3]):([0-9]|[0-5][0-9]):([0-9]|[0-5][0-9])[^0-9]"
timestamp_pattern_2 = r"[\s]?((([1-9]|0[1-9]|1[0-2])[-\.\/]([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1]))|(([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[-\.\/]([1-9]|1[0-2]|0[1-9])))[-\.\/]([1-2][0-9]{3}|[0-9]{2})[\s]([0-9]|[0-1][0-9]|2[0-3]):([0-9]|[0-5][0-9]):([0-9]|[0-5][0-9])[^0-9]"
windows_file_path = r"[A-Za-z]:[\\][^\s]*"
unix_file_path = r""
url_pattern = r"(?:(?:http|https|url):\/\/|(?:www|ftp)\.)[^\s]+"
email_pattern = r"[\S]+@[^\s\.]+\.[\S]+"
html_tag = r"\<[^\>]*\>"
#     filename_pattern = r'[\w-]+\.+[A-za-z]{1,4}\s'


def count_words(text):
    return len(text.split())


def remove_repeating_words(words):
    if len(words) > 0:
        new_words = [words[0]]
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                continue
            else:
                new_words.append(words[i])
        return new_words
    else:
        return []


def remove_repeating_chars(text):
    words = text.split()
    for w in range(len(words)):
        word_lst = list(words[w])
        if len(words[w]) > 2:
            count = 0
            lenth = len(word_lst)
            i = 1
            while i < lenth:
                if i < lenth:
                    if word_lst[i - 1] != word_lst[i]:
                        count = 0
                    else:
                        count += 1
                    if count >= 2:
                        word_lst.pop(i)
                        i -= 1
                        lenth = len(word_lst)
                i += 1
        words[w] = "".join(word_lst)
    return " ".join(words).strip()


def remove_ly(word):
    if word.endswith("ly") and not word.endswith("ily"):
        word = word[:-2]
    return word


def perform_lemmatization(text):
    """
    Function to perform nlp operations like tokenization
    and performing lemmatization
    """
    tokens = nlp(text)
    token_list = []
    for token in tokens:
        word = token.text
        if token.lemma_ != "-PRON-":
            token_list.append(remove_ly(token.lemma_))
        else:
            token_list.append(remove_ly(word))
    token_list = remove_repeating_words(token_list)
    text = " ".join(token_list)
    return text


def keep_alpha_space(text):
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


def get_decontracted_form(text):
    """
    Function to expand words like won't, can't etc.
    """
    text = re.sub(r"won[\']t", "will not", text)
    text = re.sub(r"can[\']t", "can not", text)
    text = re.sub(r"shan[\']t", "shall not", text)
    text = re.sub(r"she ain[\']t", "she is not", text)
    text = re.sub(r"he ain[\']t", "he is not", text)
    text = re.sub(r"you ain[\']t", "you are not", text)
    text = re.sub(r"it ain[\']t", "it is not", text)
    text = re.sub(r"they ain[\']t", "they are not", text)
    text = re.sub(r"we ain[\']t", "we are not", text)
    text = re.sub(r"i ain[\']t", "i am not", text)
    text = re.sub(r"n[\']t", " not", text)
    text = re.sub(r"[\']re", " are", text)
    text = re.sub(r"[\']ll", " will", text)
    text = re.sub(r"[\']t", " it ", text)
    text = re.sub(r"[\']ve", " have", text)
    text = re.sub(r"[\']m", " am", text)
    text = re.sub(r"[\']em", " them", text)
    text = re.sub(r"[\']s", " is", text)
    return text


def remove_stopwords(text):
    words = text.split()
    new_text = ""
    for word in words:
        if not word in STOPWORDS:
            new_text += word + " "
    return new_text.strip()


def unicode_normalize(text):
    text = unicodedata.normalize("NFKD", text)
    return text


def lowercasing(text):
    return text.lower()


def general_regex(text):
    text = re.sub(html_tag, " ", text)
    text = re.sub(windows_file_path, " ", text)
    text = re.sub(url_pattern, " url ", text)
    # text = re.sub(filename_pattern, ' ', text)
    text = re.sub(email_pattern, " email ", text)
    text = re.sub(ip_pattern, " ip address ", text)
    text = re.sub(timestamp_pattern_1, " ", text)
    text = re.sub(timestamp_pattern_2, " ", text)
    text = re.sub(yyyy_date_pattern, " ", text)
    text = re.sub(m_d_y_data_pattern, " ", text)
    return text


def get_embeddings(data, embed_model):
    return list(data.map(lambda x: embed_model([x])[0].numpy()))


def get_word_vec(glove_filepath):
    word_vec = {}
    with open(glove_filepath, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype="float32")
            word_vec[word] = vec
    return word_vec


def get_tokenizer_object(sentences, num_words=20000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def get_text_sequences(tokenizer, sentences, seq_len=None):
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, maxlen=seq_len)


def get_embedding_matrix(
    glove_filepath, word_ind_dict, num_words, embed_dim, vocab_size
):
    word_vec = get_word_vec(glove_filepath)
    embedding_matrix = np.zeros((num_words, embed_dim))
    for word, ind in word_ind_dict.items():
        if ind < vocab_size:
            embed_vector = word_vec.get(word)
            if embed_vector is not None:
                embedding_matrix[ind] = embed_vector
    return embedding_matrix


def get_embedding_layer(num_words, embed_dim, embed_matrix, seq_len):
    return Embedding(
        input_dim=num_words,
        output_dim=embed_dim,
        weights=[embed_matrix],
        input_length=seq_len,
        trainable=False,
    )
