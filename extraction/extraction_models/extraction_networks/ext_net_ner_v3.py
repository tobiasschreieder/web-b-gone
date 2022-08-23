from typing import List, Dict

import numpy as np
import spacy
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense
from keras.models import Sequential

from extraction import nerHelper
from classification.preprocessing import Website
from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkNerV3(BaseExtractionNetwork):

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='NerV3',
                         description='Try to extract information with multiple own NER-Models - one Model for one Entity')
        self.nlp = spacy.load('en_core_web_sm')
        self.EMB_DIM = self.nlp.vocab.vectors_length
        self.MAX_LEN = 50

    def predict(self, web_ids: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        html_text = nerHelper.get_html_text(web_ids[0])
        results = []
        for web_id in web_ids:
            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')
            new_attributes = {}
            for attr, value in attributes.items():
                value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
                new_attributes[str(attr).upper()] = value_preprocessed

            pred_samples = nerHelper.html_text_to_BIO(html_text, new_attributes)

            X_pred, y_pred, schema = self.preprocess(pred_samples)

            self.load()
            y_probs = self.model.predict(X_pred)
            y_pred = np.argmax(y_probs, axis=-1)
            id_result = {}
            for attr in attributes:
                id_result[str(attr).upper()] = []
            for sentence, tag_pred in zip(pred_samples, y_pred):
                in_tag = False
                current_token = ""
                for (token, tag), index in zip(sentence, tag_pred):
                    if tag not in ['_', 'O']:
                        if not in_tag:
                            in_tag = tag
                            current_token = str(token)
                        elif tag == in_tag:
                            current_token += " " + str(token)
                        elif not tag == in_tag:
                            if current_token not in id_result[in_tag]:
                                id_result[in_tag].append(current_token)
                            current_token = ""
                            in_tag = tag
                    else:
                        if in_tag:
                            if current_token not in id_result[in_tag]:
                                id_result[in_tag].append(current_token)
                            current_token = ""
                            in_tag = False
                if in_tag:
                    if current_token not in id_result[in_tag]:
                        id_result[in_tag].append(current_token)
            results.append(id_result)
        return results

    def train(self, web_ids: List[str], **kwargs) -> None:
        epochs = 10
        batch_size = 32

        website = Website.load(web_ids[0])
        attributes = website.truth.attributes
        attributes.pop('category')

        for attr in attributes:
            attribute = attr.upper()
            print("TRAINING FOR ATTRIBUTE: ", attribute)

            train_samples = []
            for web_id in web_ids:
                html_text = nerHelper.get_html_text(web_id)

                website = Website.load(web_id)
                attributes = website.truth.attributes
                attributes.pop('category')
                new_attributes = {}
                value = attributes[attribute]
                value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
                new_attributes[str(attr).upper()] = value_preprocessed

                train_samples += nerHelper.html_text_to_BIO(html_text, new_attributes)

            X_train, y_train, schema_train = self.preprocess(train_samples)

            model = self.build_model(schema=schema_train)

            model.compile(optimizer='Adam',
                          loss='sparse_categorical_crossentropy',
                          metrics='accuracy')
            history = model.fit(X_train, y_train,
                                validation_split=0.2,
                                epochs=epochs,
                                batch_size=batch_size)

            #TODO: save every model for every entity correct!
            self.model = model
            self.save()

    def preprocess(self, samples):
        schema = ['_'] + sorted({tag for sentence in samples for _, tag in sentence})
        # schema = sorted({tag for sentence in samples for _, tag in sentence})
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
