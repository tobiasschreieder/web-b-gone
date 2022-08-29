import logging
from typing import List, Dict

import numpy as np
import spacy
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from classification.preprocessing import Website
from extraction import ner_helper
from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkNerV2(BaseExtractionNetwork):
    """
    NER-Extraction with own LSTM-Model.
    Create our own LSTM-Model for extraction
    """

    log = logging.getLogger('ExtNet-NerV2')

    def __init__(self, name: str):
        super().__init__(name=name, version='NerV2',
                         description='Try to extract information with own NER-Model - trained with multiple attributes')
        self.nlp = spacy.load('en_core_web_md')
        self.EMB_DIM = self.nlp.vocab.vectors_length
        self.MAX_LEN = 50

    def predict(self, web_ids: List[str], k=3, **kwargs) -> List[Dict[str, List[str]]]:
        results = []
        for web_id in web_ids:
            html_text = ner_helper.get_html_text(web_id)
            website = Website.load(web_id)
            attributes_dict = website.truth.attributes
            attributes_dict.pop('category')
            attributes = []
            for attr in attributes_dict:
                attributes.append(attr)

            schema = ['_'] + sorted(attributes + ['O'])
            x_pred = self._preprocess_predict(html_text)

            self.load()
            y_probs = self.model.predict(x_pred)
            y_pred = np.argmax(y_probs, axis=-1)
            id_result = {}
            for attr in attributes:
                id_result[str(attr)] = []
            for sentence, tag_pred in zip(html_text, y_pred):
                in_tag = False
                current_token = ""
                sentence_list = sentence.split(" ")
                for token, tag in zip(sentence_list, tag_pred):
                    tag = schema[tag]
                    if tag not in ['_', 'O']:
                        if not in_tag:
                            in_tag = tag
                            current_token = str(token)
                        elif tag == in_tag:
                            current_token += " " + str(token)
                        elif not tag == in_tag:
                            id_result[in_tag].append(current_token)
                            current_token = ""
                            in_tag = tag
                    else:
                        if in_tag:
                            id_result[in_tag].append(current_token)
                            current_token = ""
                            in_tag = False
                if in_tag:
                    id_result[in_tag].append(current_token)

                for label in id_result:
                    if len(id_result[label]) > 3:
                        lst_sorted = sorted([ss for ss in set(id_result[label]) if len(ss) > 0 and ss.istitle()],
                                            key=id_result[label].count,
                                            reverse=True)
                        id_result[label] = lst_sorted[0:k]

            results.append(id_result)
        return results

    def train(self, web_ids: List[str], **kwargs) -> None:
        epochs = 3
        batch_size = 64

        train_samples = []
        for web_id in web_ids:
            html_text = ner_helper.get_html_text(web_id)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')
            new_attributes = {}
            for attr, value in attributes.items():
                if value:
                    value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
                    new_attributes[str(attr).upper()] = value_preprocessed

            train_samples += ner_helper.html_text_to_BIO(html_text, new_attributes)

        x_train, y_train, schema_train = self._preprocess_train(train_samples)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, shuffle=True)

        model = self._build_model(schema=schema_train)

        model.compile(optimizer='Adam',
                      loss='sparse_categorical_crossentropy',
                      metrics='accuracy')

        history = model.fit(x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x_valid, y_valid))
        self.model = model
        self.save()
        return history

    def _preprocess_train(self, samples):
        schema = ['_'] + sorted({tag for sentence in samples for _, tag in sentence})
        tag_index = {tag: index for index, tag in enumerate(schema)}
        x = np.zeros((len(samples), self.MAX_LEN, self.EMB_DIM), dtype=np.float32)
        y = np.zeros((len(samples), self.MAX_LEN), dtype=np.uint8)
        vocab = self.nlp.vocab
        for i, sentence in enumerate(samples):
            print(sentence)
            for j, (token, tag) in enumerate(sentence[:self.MAX_LEN]):
                x[i, j] = vocab.get_vector(token)
                y[i, j] = tag_index[tag]
        return x, y, schema

    def _preprocess_predict(self, html_text):
        x = np.zeros((len(html_text), self.MAX_LEN, self.EMB_DIM), dtype=np.float32)
        vocab = self.nlp.vocab
        for i, sentence in enumerate(html_text):
            for j, token in enumerate(sentence[:self.MAX_LEN]):
                x[i, j] = vocab.get_vector(token)
        return x

    def _build_model(self, schema, nr_filters=256):
        input_shape = (self.MAX_LEN, self.EMB_DIM)
        lstm = LSTM(nr_filters, return_sequences=True)
        bi_lstm = Bidirectional(lstm, input_shape=input_shape)
        tag_classifier = Dense(len(schema), activation='softmax')
        sequence_labeller = TimeDistributed(tag_classifier)
        return Sequential([bi_lstm, sequence_labeller])
