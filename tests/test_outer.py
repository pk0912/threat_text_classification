"""
Python file containing methods used for testing outer modules
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorizer as vec


class TestVectorizer:
    def test_vectorize(self):
        data = ["This is first sentence", "This is the second sentence"]
        vocab_size = 20000
        seq_len = 5
        tokenizer, vectors = vec.vectorize(vocab_size, seq_len, data)
        assert tokenizer is not None
        assert (vectors == [[0, 1, 2, 4, 3], [1, 2, 5, 6, 3]]).all()
