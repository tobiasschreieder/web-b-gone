import logging
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from joblib import Parallel, delayed

import evaluation.text_preprocessing as tp
from classification.preprocessing import Category, Website
from .base_model import BaseExtractionModel
from ..structure_helper import simple_string_match, build_text_position_mapping, candidates_filter, find_best_candidate


class StructuredTemplate:
    log = logging.getLogger('StrucTemp')

    def __init__(self):
        self.web_ids_trained_on = []
        self.web_ids_faulty = []
        self.attributes = []
        self.positions = []
        self.text_info = []

    def add_faulty_id(self, web_id: str):
        self.web_ids_faulty.append(web_id)

    def get_web_ids(self) -> List[str]:
        return list({web_id for web_id in self.web_ids_trained_on if web_id not in self.web_ids_faulty})

    def add_attribute(self, attribute: str, position, text_info: List[int], web_id: str) -> None:
        self.attributes.append(attribute)
        self.positions.append(position)
        self.text_info.append(text_info)
        self.web_ids_trained_on.append(web_id)

    def get_from_attribute(self, attribute: str):
        if attribute in self.attributes:
            index = [i for i, x in enumerate(self.attributes) if x == attribute]
            all_attributes = []
            for i in index:
                all_attributes.append(
                    {'attribute': attribute,
                     'position': self.positions[i],
                     'text_info': self.text_info[i],
                     'web_id': self.web_ids_trained_on[i]}
                )
            return all_attributes
        else:
            return False

    def get_from_web_id(self, web_id: str):
        if web_id not in self.web_ids_trained_on:
            return False
        else:
            index = [i for i, x in enumerate(self.web_ids_trained_on) if x == web_id]
            all_attributes = []
            for i in index:
                all_attributes.append(
                    {'attribute': self.attributes[i],
                     'position': self.positions[i],
                     'text_info': self.text_info[i],
                     'web_id': web_id}
                )
            return all_attributes

    def get_from_web_id_and_attribute(self, web_id: str, attribute: str):
        if web_id not in self.web_ids_trained_on:
            return False
        else:
            index = [i for i, x in enumerate(self.web_ids_trained_on) if x == web_id]
            for i in index:
                if self.attributes[i] == attribute:
                    return {'attribute': self.attributes[i],
                            'position': self.positions[i],
                            'text_info': self.text_info[i],
                            'web_id': web_id}
        return False

    def get_all_attributes(self) -> List[Dict[str, Any]]:  # TODO
        all_attributes = []
        for index in range(len(self.attributes)):
            all_attributes.append(
                {'attribute': self.attributes[index],
                 'position': self.positions[index],
                 'text_info': self.text_info[index]}
            )
        return all_attributes

    def print(self):
        print("TEMPLATE")
        for e in self.get_all_attributes():
            print(e)
        print("---")


class StructuredTemplateExtractionModel(BaseExtractionModel):
    template: StructuredTemplate
    log = logging.getLogger('StrucTempExtModel')

    def __init__(self, category: Category):
        super().__init__(category)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Learn a structured template from the given website ids.

        :param web_ids: Website ids to learn the template from.
        :return: None
        """
        template = StructuredTemplate()
        for web_id in web_ids:
            self.log.debug(f'Start learning from web_id {web_id}')
            website = Website.load(web_id)
            with Path(website.file_path).open(encoding='utf-8') as htm_file:
                soup = BeautifulSoup(htm_file, features="html.parser")

            body = soup.find('body')
            # html_tree = build_html_tree(body, [])
            # print_html_tree(html_tree)
            text_position_mapping = build_text_position_mapping(body)

            attributes = website.truth.attributes
            if 'category' in attributes:
                attributes.pop('category')
            for key in attributes:
                self.log.debug(f'Find best match for attribute {key}')
                # TODO witch ground truth should be used, currently the first
                if len(attributes[key]) == 0:
                    template.add_faulty_id(web_id)
                    continue

                attributes[key] = tp.preprocess_text_html(attributes[key][0])
                best_match = 0
                best_position = None
                for mapping in text_position_mapping:
                    match = simple_string_match(mapping['text'], website.truth.attributes[key])
                    if match > best_match:
                        best_match = match
                        text_correct = str(website.truth.attributes[key]).split(" ")
                        text_found = str(mapping['text']).split(" ")
                        text_info = [0] * len(text_found)
                        for correct_word in text_correct:
                            if correct_word in text_found:
                                pos = text_found.index(correct_word)
                                text_info[pos] = 1

                        best_position = {'attribute': key,
                                         'position': mapping['position'],
                                         'text_info': text_info}
                if best_match == 0:
                    template.add_faulty_id(web_id)
                    continue

                template.add_attribute(attribute=best_position['attribute'],
                                       position=best_position['position'],
                                       text_info=best_position['text_info'],
                                       web_id=web_id)
        # TODO: cluster templates
        self.template = template

    def extract(self, web_ids: List[str], n_jobs: int = 1, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract information from websites using the structured template approach.

        :param web_ids: List of website ids for extraction
        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :return: Extracted information
        """

        def extract_web_id(web_id: str) -> Dict[str, List[str]]:
            website = Website.load(web_id)
            self.log.debug(f'Extract for web_id {web_id}')
            with Path(website.file_path).open(encoding='utf-8') as htm_file:
                soup = BeautifulSoup(htm_file, features="html.parser")

            body = soup.find('body')
            # html_tree = build_html_tree(body, [])
            # print_html_tree(html_tree)
            text_position_mapping = build_text_position_mapping(body)

            attributes = website.truth.attributes
            if 'category' in attributes:
                attributes.pop('category')

            candidates = []
            for key in attributes:
                if key in ['name', 'team']:
                    filter_category = 'Name'
                elif key in ['height', 'weight']:
                    filter_category = 'Number'
                else:
                    self.log.warning(f'Skipped attribute key {key}, no filter defined')
                    continue

                candidates.append({'attribute': key,
                                   'candidates': candidates_filter(filter_category, text_position_mapping)})
            best_cand = find_best_candidate(candidates, self.template)
            res = {}
            for att in best_cand:
                res[att['attribute']] = [att['candidate']['text']]
            return res

        if len(web_ids) > 20:
            with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
                return parallel(delayed(extract_web_id)(web_id) for web_id in web_ids)
        else:
            return [extract_web_id(web_id) for web_id in web_ids]
