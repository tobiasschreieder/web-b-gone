import string
from typing import List, Dict

from nltk.stem import WordNetLemmatizer

allowed_chars = ["$", ".", ",", "%"]  # List of chars which will not be removed
stop_words = ["nbsp", "&nbsp"]


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

    # remove stopwords from given text
    for sw in stop_words:
        text = text.replace(sw, " ")

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


def split_char_numeric(text: str) -> str:
    """
    Split letters from numeric characters for each token in given string: "abc123" -> "abc 123"
    :param text: String with given text
    :return: new separated string
    """
    tokens = text.split(" ")
    new_text = text

    for token in tokens:
        for char in token:
            if char.isdigit():
                i = token.index(char)
                text_1 = token[:i]
                text_2 = token[i:]
                new_token = text_1 + " " + text_2
                new_text = new_text.replace(token, new_token)
                break
    return new_text


def preprocess_text_comparison(text: str) -> str:
    """
    Method to preprocess text for comparison
    :param text: string with given text
    :return: string with given text
    """
    text = text.lower()
    text = split_char_numeric(text=text)
    text = remove_stop_chars(text=text)
    text = lemmatize_text(text=text)

    return text


def preprocess_text_partial_match(text: str) -> str:
    """
    Method to preprocess text for partial match
    :param text: string with given text
    :return: string with given text
    """
    text = ''.join(char for char in text if char.isalnum())

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

    if not isinstance(text, str):
        text = " ".join(text)

    text = text.lower()
    text = remove_stop_chars(text=text)

    return text
