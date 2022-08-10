from typing import List
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
import string

Q = 2  # Value length of q-grams for comparison function
allowed_chars = ["$", ".", ",", "%"]  # List of chars which will not be removed


def padding(text: str, q: int) -> str:
    """
    Add padding characters to given text -> #
    :param text: string with given text
    :param q: length of q-grams
    :return: string with added padding characters
    """
    pad_chars = str()
    i = 0

    while i < q - 1:
        pad_chars = pad_chars + "#"
        i += 1
    text = pad_chars + text + pad_chars

    return text


def create_grams(text: str, q: int, use_padding: bool) -> List:
    """
    Create q-grams from given text
    :param text: string with given text
    :param q: length of q-grams
    :param use_padding: True if padding should be used, else: False
    :return: List with created q-grams
    """
    if use_padding:
        text = padding(text=text, q=Q)

    q_grams = ["".join(k1) for k1 in list(ngrams(text, n=q))]

    return q_grams


def remove_stop_chars(text: str) -> str:
    """
    Method to remove punctuations and spaces
    :param text: string with given text
    :return: string with preprocessed text
    """
    punctuations = string.punctuation

    # just use punctuations that are not included in allowed_chars
    punctuations_used = "".join([i for i in punctuations if i not in allowed_chars])

    # remove punctuations from given text
    text = "".join([i for i in text if i not in punctuations_used])

    # remove spaces with length > 1 ("   " -> " ")
    text = " ".join(text.split())

    return text


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text with WordNetLemmatizer
    :param text: string with given text
    :return: string with lemmatized text
    """
    wordnet_lemmatizer = WordNetLemmatizer()
    text = wordnet_lemmatizer.lemmatize(text)

    return text


def preprocess_text_comparison(text: str) -> str:
    """
    Method to preprocess text for comparison
    :param text: string with given text
    :return: string with given text
    """
    text = text.lower()
    text = remove_stop_chars(text=text)
    text = lemmatize_text(text=text)

    return text
