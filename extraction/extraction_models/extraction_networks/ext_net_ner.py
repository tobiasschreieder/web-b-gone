from typing import List, Dict
from copy import copy
from pathlib import Path

import spacy
import numpy as np
from bs4 import BeautifulSoup
from bs4.element import Comment
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense

from classification.preprocessing import Category, Website
from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkNerV1(BaseExtractionNetwork):

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='NER_v1', description='Try to extract information with NER')
        self.nlp = spacy.load('en_core_web_sm')
        self.EMB_DIM = self.nlp.vocab.vectors_length
        self.MAX_LEN = 50

    def predict(self, web_ids: List[str]) -> List[Dict[str, List[str]]]:
        html_text = self.get_html_text(web_ids[0])
        # TODO: currently just predict first web_id
        website = Website.load(web_ids[0])
        attributes = website.truth.attributes
        attributes.pop('category')
        new_attributes = {}
        for attr, value in attributes.items():
            value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
            new_attributes[str(attr).upper()] = value_preprocessed

        pred_samples = self.html_text_to_BIO(html_text, new_attributes)

        X_pred, y_pred, schema = self.preprocess(pred_samples)

        self.load()
        y_probs = self.model.predict(X_pred)
        y_pred = np.argmax(y_probs, axis=-1)
        result = []
        for sentence, tag_pred in zip(pred_samples, y_pred):
            for (token, tag), index in zip(sentence, tag_pred):
                # result.append([(token, tag, schema[index])])
                result.append((token, schema[index]))

        # TODO: tranform result in correct format for return
        return result

    def train(self, web_ids: List[str]) -> None:
        epochs = 2
        batch_size = 32

        train_samples = []
        for web_id in web_ids:
            html_text = self.get_html_text(web_id)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')
            new_attributes = {}
            for attr, value in attributes.items():
                value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
                new_attributes[str(attr).upper()] = value_preprocessed

            train_samples += self.html_text_to_BIO(html_text, new_attributes)
            print(len(train_samples))

        X_train, y_train, schema_train = self.preprocess(train_samples)

        model = self.build_model(schema=schema_train)

        model.compile(optimizer='Adam',
                      loss='sparse_categorical_crossentropy',
                      metrics='accuracy')
        history = model.fit(X_train, y_train,
                            validation_split=0.2,
                            epochs=epochs,
                            batch_size=batch_size)
        self.model = model
        self.save()

    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def get_html_text(self, web_id):
        print(web_id)
        website = Website.load(web_id)
        with Path(website.file_path).open(encoding='utf-8') as htm_file:
            soup = BeautifulSoup(htm_file, features="html.parser")
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        line_list = []
        for line in [t.strip() for t in visible_texts]:
            if line:
                line = ' '.join(line.split())
                line_list.append(line)
        return line_list

    def html_text_to_BIO(self, text, attributes):
        bio_format = []
        for line in text:
            labels = ['O'] * len(copy(line).split(' '))
            line_list = line.split(" ")
            for attr, value in attributes.items():
                value_list = value.split(" ")
                if (value in line) and (value_list[0] in line_list):
                    start_index = line_list.index(value_list[0])
                    for i in range(len(value_list)):
                        labels[start_index + i] = attr
            bio_line = []
            for i in range(len(line_list)):
                bio_line.append((line_list[i], labels[i]))
            bio_format.append(bio_line)
        return bio_format

    def preprocess(self, samples):
        # schema = ['_'] + sorted({tag for sentence in samples for _, tag in sentence})
        schema = sorted({tag for sentence in samples for _, tag in sentence})
        print(schema)
        tag_index = {tag: index for index, tag in enumerate(schema)}
        X = np.zeros((len(samples), self.MAX_LEN, self.EMB_DIM), dtype=np.float32)
        y = np.zeros((len(samples), self.MAX_LEN), dtype=np.uint8)
        vocab = self.nlp.vocab
        for i, sentence in enumerate(samples):
            for j, (token, tag) in enumerate(sentence[:self.MAX_LEN]):
                X[i, j] = vocab.get_vector(token)
                y[i, j] = tag_index[tag]
        return X, y, schema

    def load_data(self, filename: str):
        with open(filename, 'r', encoding="utf8") as file:
            lines = [line[:-1].split() for line in file]
        samples, start = [], 0
        for end, parts in enumerate(lines):
            if not parts:
                sample = [(token, tag.split('-')[-1]) for token, tag in lines[start:end]]
                samples.append(sample)
                start = end + 1
        if start < end:
            samples.append(lines[start:end])
        return samples

    def build_model(self, schema, nr_filters=256):
        input_shape = (self.MAX_LEN, self.EMB_DIM)
        lstm = LSTM(nr_filters, return_sequences=True)
        bi_lstm = Bidirectional(lstm, input_shape=input_shape)
        tag_classifier = Dense(len(schema), activation='softmax')
        sequence_labeller = TimeDistributed(tag_classifier)
        return Sequential([bi_lstm, sequence_labeller])
