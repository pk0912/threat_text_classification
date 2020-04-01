"""
Python file containing methods used for testing utils modules
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.text_processing as tp


class TestTextProcessing:
    def test_count_words(self):
        assert tp.count_words(" word1 word2 ") == 2
        assert tp.count_words("") == 0

    def test_remove_repeating_words(self):
        assert tp.remove_repeating_words(["word1", "word2", "word2"]) == [
            "word1",
            "word2",
        ]
        assert tp.remove_repeating_words(
            ["word1", "w2", "w2", "w2", "word1", "w2"]
        ) == ["word1", "w2", "word1", "w2"]
        assert tp.remove_repeating_words(["w1", "w2", "w3"]) == ["w1", "w2", "w3"]
        assert tp.remove_repeating_words([]) == []

    def test_remove_repeating_chars(self):
        assert tp.remove_repeating_chars("wooord") == "woord"
        assert tp.remove_repeating_chars("woord") == "woord"
        assert tp.remove_repeating_chars("") == ""
        assert tp.remove_repeating_chars("word") == "word"
        assert tp.remove_repeating_chars("xxx") == "xx"
        assert (
            tp.remove_repeating_chars("tickets mtv movie awards scammmed")
            == "tickets mtv movie awards scammed"
        )

    def test_remove_ly(self):
        assert tp.remove_ly("beautifully") == "beautiful"
        assert tp.remove_ly("family") == "family"
        assert tp.remove_ly("") == ""
        assert tp.remove_ly("word") == "word"

    def test_perform_lemmatization(self):
        # TODO
        pass

    def test_keep_alpha_space(self):
        assert tp.keep_alpha_space("this is a sentence.") == "this is a sentence "
        assert (
            tp.keep_alpha_space("this is a sentence with numbers 123435")
            == "this is a sentence with numbers       "
        )
        assert (
            tp.keep_alpha_space("this is #trending!!! i don't know why though?")
            == "this is  trending    i don t know why though "
        )

    def test_get_decontracted_form(self):
        assert tp.get_decontracted_form("i won't.") == "i will not."
        assert tp.get_decontracted_form("i can't") == "i can not"
        assert tp.get_decontracted_form("i shan't") == "i shall not"
        assert (
            tp.get_decontracted_form("she ain't reading that.")
            == "she is not reading that."
        )
        assert tp.get_decontracted_form("he ain't going.") == "he is not going."
        assert (
            tp.get_decontracted_form("you ain't doing that") == "you are not doing that"
        )
        assert (
            tp.get_decontracted_form("it ain't getting done.")
            == "it is not getting done."
        )
        assert tp.get_decontracted_form("they ain't coming.") == "they are not coming."
        assert tp.get_decontracted_form("we ain't going.") == "we are not going."
        assert tp.get_decontracted_form("i ain't working.") == "i am not working."
        assert tp.get_decontracted_form("don't go") == "do not go"
        assert tp.get_decontracted_form("she isn't happy") == "she is not happy"
        assert tp.get_decontracted_form("we're going.") == "we are going."
        assert tp.get_decontracted_form("we'll do it.") == "we will do it."
        assert tp.get_decontracted_form("we've been there.") == "we have been there."
        assert tp.get_decontracted_form("i'm not going.") == "i am not going."
        assert tp.get_decontracted_form("get'em.") == "get them."
        assert tp.get_decontracted_form("he's not coming.") == "he is not coming."

    def test_remove_stopwords(self):
        assert tp.remove_stopwords("he is am not") == ""

    def test_unicode_normalize(self):
        assert type(tp.unicode_normalize("word word")) == str

    def test_general_regex(self):
        assert (
            tp.general_regex("192.168.0.34 https://www.google.com praveen@anlyz.io")
            == " ip address  url   email "
        )
        assert tp.general_regex(r"<html>This is html<\html>") == " This is html "

    def test_get_embeddings(self):
        # TODO
        pass

    def test_get_word_vec(self):
        assert (
            len(
                tp.get_word_vec(
                    "/Users/praveenkumar/Own_works/resources/nlp/glove/glove.6B.100d.txt"
                )
            )
            != 0
        )

    def test_get_tokenizer_object(self):
        assert (
            tp.get_tokenizer_object(
                ["This is first sentence", "This is second sentence"]
            )
            is not None
        )

    def test_get_text_sequences(self):
        sentences = ["This is first sentence", "This is second sentence"]
        tokenizer = tp.get_tokenizer_object(sentences)
        assert (
            tp.get_text_sequences(tokenizer, sentences) == [[1, 2, 4, 3], [1, 2, 5, 3]]
        ).all()
