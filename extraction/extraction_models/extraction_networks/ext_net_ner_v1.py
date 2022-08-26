import logging
import random
from typing import List, Dict

import spacy
from spacy.training.example import Example

from classification.preprocessing import Website
from extraction import nerHelper
from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkNerV1(BaseExtractionNetwork):

    model: spacy.Language
    log = logging.getLogger('ExtNet-NerV1')

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='NerV1',
                         description='Try to extract information with custom SPACY-Model')
        self.nlp = spacy.load('en_core_web_md')
        self.EMB_DIM = self.nlp.vocab.vectors_length
        self.MAX_LEN = 50

    def predict(self, web_ids: List[str], k=3, **kwargs) -> List[Dict[str, List[str]]]:
        self.load()

        result_list = []
        for web_id in web_ids:
            html_text = nerHelper.get_html_text(web_id)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')

            id_results = {}
            for attr in attributes:
                id_results[str(attr)] = []

            doc = self.model(' '.join(html_text))
            # doc = self.model(html_text)
            for ent in doc.ents:
                id_results[ent.label_].append(ent.text)

            for label in id_results:
                if len(id_results[label]) > 3:
                    lst_sorted = sorted([ss for ss in set(id_results[label]) if len(ss) > 0 and ss.istitle()],
                                        key=id_results[label].count,
                                        reverse=True)
                    id_results[label] = lst_sorted[0:k]
                    # id_results[label] = [max(set(id_results[label]), key=id_results[label].count)]

            result_list.append(id_results)

        return result_list

    def train(self, web_ids: List[str], **kwargs) -> None:
        epochs = 40
        batch_size = 32

        training_data = {'classes': [], 'annotations': []}
        for web_id in web_ids:
            html_text = nerHelper.get_html_text(web_id)
            # print("html_text: ", html_text)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')
            new_attributes = {}
            for attr, value in attributes.items():
                if value:
                    value_preprocessed = str(value[0]).replace('&nbsp;', ' ').strip()
                    new_attributes[str(attr)] = value_preprocessed
                else:
                    new_attributes[str(attr)] = []

            training_data['annotations'].append(nerHelper.html_text_to_spacy(html_text, new_attributes))

        nlp = spacy.blank("en")  # load a new spacy model

        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner')
        else:
            ner = nlp.get_pipe('ner')

        for attr in attributes:
            training_data['classes'].append(str(attr))
            self.log.debug("Attribute added: %s", attr)
            ner.add_label(attr)

        optimizer = nlp.begin_training()

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            for itn in range(epochs):
                random.shuffle(training_data['annotations'])
                losses = {}
                for batch in spacy.util.minibatch(training_data['annotations'], size=batch_size):
                    for content in batch:
                        # create Example
                        doc = nlp.make_doc(content['text'])
                        example = Example.from_dict(doc, content)
                        # Update the model
                        nlp.update([example], losses=losses, drop=0.3, sgd=optimizer)
                self.log.info('Iteration %s / %s with loss: %s', itn+1, epochs, losses)

        self.model = nlp
        self.save()

    def load(self) -> None:
        if not self.dir_path.exists():
            raise ValueError(f"The model '{self.name}' for version {self.version} doesn't exit.")
        self.model = spacy.load(self.dir_path)

    def save(self) -> None:
        if self.model is None:
            raise ValueError(f"No model to save. Model '{self.name}' for version {self.version} not set.")
        self.model.meta['name'] = self.name
        self.model.to_disk(self.dir_path.as_posix())
