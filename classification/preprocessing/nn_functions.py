import json
import logging
from datetime import datetime, time
from typing import List, Dict
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np

from config import Config
from data_storage import GroundTruth, Category

cfg = Config.get()
log = logging.getLogger('categorize_prepare')
words = []
documents = []
ignore_words = ['?']
categories = [Category.AUTO, Category.BOOK, Category.CAMERA, Category.JOB, Category.MOVIE, Category.NBA_PLAYER,
              Category.RESTAURANT, Category.UNIVERSITY]

"""
    Prepare text for usage in neural network.
"""


def prepare_text_for_nn(self, training_data: List[str]):
    self.words = []
    self.documents = []
    self.ignore_words = ['?']
    stemmer = LancasterStemmer()
    # create our training data
    training = []
    output = []
    # create an empty array for our output
    # loop through each sentence in our training data
    for pattern in training_data:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern['text_all'])
        # add to our words list
        self.words.extend(w)
        # add to documents in our corpus
        documents.append((w, pattern['web_id']))

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
    words = list(set(words))

    print(len(documents), "documents")
    print(len(words), "unique stemmed words", words)

    ##todo
    # create a list of tokenized words for the pattern and also create a bag of words by using NLTK Lancaster Stemmer
    stemmer = LancasterStemmer()
    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * 8
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        training.append(bag)
        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[categories.index(doc[1])] = 1
        output.append(output_row)

    print("# words", len(words))


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    stemmer = LancasterStemmer()
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(self, sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = self.clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))

