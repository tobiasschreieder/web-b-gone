import logging
from typing import List, Dict

from classification.preprocessing import Category
from config import Config
from .base_model import BaseExtractionModel
from ..structure_tree_template import StructuredTreeTemplate

cfg = Config.get()

# TODO boilerplate with regex search after -nav nav nav-, ggf header
# TODO update build_mapping so no comments are returned


class StrucTempExtractionModelV3(BaseExtractionModel):
    template: StructuredTreeTemplate
    log = logging.getLogger('StrucTempExtModel V3')

    name: str

    def __init__(self, category: Category, name: str):
        super().__init__(category, 'StrucTemp_v3')
        self.name = name
        self.template = None
        self.dir_path = self.dir_path.joinpath(self.name)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Learn a structured template from the given website ids.

        :param web_ids: Website ids to learn the template from.
        :return: None
        """
        template = StructuredTreeTemplate()
        template.train(web_ids)
        self.template = template
        self.template.save(self.dir_path)

    def extract(self, web_ids: List[str], n_jobs: int = -2, k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract information from websites using the structured template approach.

        :param web_ids: List of website ids for extraction
        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :param k: number of results per attribute, default: 3
        :return: Extracted information
        """
        if self.template is None:
            self.template = StructuredTreeTemplate.load(self.dir_path)

        return self.template.extract(web_ids, self.category, k=k)

    def save(self) -> None:
        """
        TODO
        :return:
        """
        self.template.save(self.dir_path)

    def load(self) -> None:
        """
        TODO
        :return:
        """
        if self.template is None:
            self.template = StructuredTreeTemplate.load(self.dir_path)
