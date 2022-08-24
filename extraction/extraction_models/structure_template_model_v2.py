import logging
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup
from joblib import Parallel, delayed

import evaluation.text_preprocessing as tp
from classification.preprocessing import Category, Website
from .base_model import BaseExtractionModel
from ..structure_helper_v2 import simple_string_match, build_text_position_mapping, \
    candidates_filter, get_longest_element, StructuredTemplate, sort_n_score_candidates, build_mapping


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
            # ToDo Comments are also recognized as text
            # text_position_mapping = build_mapping(body)

            attr_truth = website.truth.attributes
            for key in attr_truth:
                if key == 'category':
                    continue

                # self.log.debug(f'Find best match for attribute {key}')
                # TODO witch ground truth should be used, currently the longest
                if len(attr_truth[key]) == 0:
                    continue

                prep_text = str(tp.preprocess_text_html(get_longest_element(attr_truth[key])))
                best_match = 0
                best_position = None
                for mapping in text_position_mapping:
                    match = simple_string_match(mapping['text'], prep_text)
                    if match > best_match:
                        best_match = match
                        text_correct = prep_text.split(" ")
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
                    continue

                template.add_attribute(attribute=best_position['attribute'],
                                       position=best_position['position'],
                                       text_info=best_position['text_info'],
                                       web_id=web_id)
        # TODO: cluster templates
        self.template = template

    def extract(self, web_ids: List[str], n_jobs: int = -1, k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract information from websites using the structured template approach.

        :param web_ids: List of website ids for extraction
        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :param k: number of results per attribute, default: 3
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

            candidates = dict()
            for key in self.category.get_attribute_names():
                filter_category = Category.get_attr_type(key)
                candidates[key] = candidates_filter(filter_category, text_position_mapping)

            return sort_n_score_candidates(candidates, self.template, k=k)

        return [extract_web_id(web_id) for web_id in web_ids]
        # if len(web_ids) > 20:
        #     with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
        #         return parallel(delayed(extract_web_id)(web_id) for web_id in web_ids)
        # else:
        #     return [extract_web_id(web_id) for web_id in web_ids]
