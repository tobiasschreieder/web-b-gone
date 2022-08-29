import logging
import random
import re
from typing import List, Dict

import spacy
from bs4 import Comment, Tag
from spacy.training import Example
from tqdm import tqdm

from classification.preprocessing import Website
from extraction import ner_helper
from .base_extraction_network import BaseExtractionNetwork


def tag_filter_regex(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False

    if isinstance(element, Tag):
        reg = re.compile('nav')

        if 'class' in element.attrs.keys():
            if any(reg.search(html_class) for html_class in element.attrs['class']):
                return False

        if 'id' in element.attrs.keys():
            if reg.search(element.attrs['id']):
                return False

    return True


class ExtractionNetworkNerV1(BaseExtractionNetwork):
    """
    NER-Extraction with spaCy.
    Train a custom spacy model to extract text.
    """

    model: spacy.Language
    log = logging.getLogger('ExtNet-NerV1')

    def __init__(self, name: str):
        super().__init__(name=name, version='NerV1',
                         description='Try to extract information with custom SPACY-Model')
        self.nlp = spacy.load('en_core_web_md')
        self.EMB_DIM = self.nlp.vocab.vectors_length
        self.MAX_LEN = 50

    def predict(self, web_ids: List[str], k=3, **kwargs) -> List[Dict[str, List[str]]]:
        self.load()

        result_list = []
        for web_id in tqdm(web_ids, desc='NerV1 Prediction'):
            html_text = ner_helper.get_html_text(web_id, filter_method=tag_filter_regex)

            website = Website.load(web_id)
            attributes = website.truth.attributes
            attributes.pop('category')

            id_results = {}
            for attr in attributes:
                id_results[str(attr)] = []

            doc = self.model(' '.join(html_text))
            for ent in doc.ents:
                id_results[ent.label_].append(ent.text)

            for label in id_results:
                if len(id_results[label]) > 3:
                    lst_sorted = sorted([ss for ss in set(id_results[label]) if len(ss) > 0 and ss.istitle()],
                                        key=id_results[label].count,
                                        reverse=True)
                    id_results[label] = lst_sorted[0:k]

            result_list.append(id_results)

        return result_list

    def train(self, web_ids: List[str], **kwargs) -> None:
        epochs = 40
        batch_size = 32

        training_data = {'classes': [], 'annotations': []}
        attributes = dict()
        for web_id in tqdm(web_ids, desc='Preprocess web_ids'):
            html_text = ner_helper.get_html_text(web_id, filter_method=tag_filter_regex)

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

            training_data['annotations'].append(ner_helper.html_text_to_spacy(html_text, new_attributes))

        filtered_anno = []
        for i in range(len(training_data['annotations'])):
            ents = training_data['annotations'][i]['entities']
            filtered_anno.append({'entities': ner_helper.filter_spans(ents),
                                  'text': training_data['annotations'][i]['text']})
        training_data['annotations'] = filtered_anno

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
            self.log.info('Start with epoch training')
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
        """
        Load the spacy model from disk.

        :return: None
        :raises ValueError: if the model doesn't exist.
        """
        if not self.dir_path.exists():
            raise ValueError(f"The model '{self.name}' for version {self.version} doesn't exit.")
        self.model = spacy.load(self.dir_path)

    def save(self) -> None:
        """
        Saves the spacy model to disk.

        :return: None
        :raises ValueError: if the model is None (not trained).
        """
        if self.model is None:
            raise ValueError(f"No model to save. Model '{self.name}' for version {self.version} not set.")
        self.model.meta['name'] = self.name
        self.model.to_disk(self.dir_path.as_posix())
