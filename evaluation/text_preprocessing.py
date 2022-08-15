import string
from typing import List, Dict

from nltk.stem import WordNetLemmatizer

allowed_chars = ["$", ".", ",", "%"]  # List of chars which will not be removed


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


def preprocess_list_comparison(texts: List[str]) -> List[str]:
    """
    Method to preprocess a list of texts for comparison
    :param texts: List with texts as string
    :return: List with preprocessed texts
    """
    preprocessed_texts = list()
    for t in texts:
        preprocessed_texts.append(preprocess_text_comparison(t))

    return preprocessed_texts


def preprocess_extraction_data_comparison(data: List[Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
    """
    Method to preprocess texts in ground-truth / results datastructure for comparison
    :param data: given datastructure
    :return: preprocessed datastructure
    """
    for w in range(0, len(data)):
        for attribute, texts in data[w].items():
            if attribute != "category":
                data[w][attribute] = preprocess_list_comparison(texts=texts)

    return data


def preprocess_text_html(text: str) -> str:
    """
    Method to preprocess text for html
    :param text: string with given text
    :return: string with given text
    """
    if not text:
        return ""

    if isinstance(text, list):
        " ".join(text)

    # rules for replacement
    print(text)
    text = text.replace("&nbsp", " ")

    text = text.lower()
    text = remove_stop_chars(text=text)

    return text
