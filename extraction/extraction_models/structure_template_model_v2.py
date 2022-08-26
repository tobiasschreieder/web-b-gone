import logging
from typing import List, Dict

from classification.preprocessing import Category
from config import Config
from .base_model import BaseExtractionModel
from ..structure_helper_v2 import StructuredTemplate

cfg = Config.get()


class StrucTempExtractionModelV2(BaseExtractionModel):
    template: StructuredTemplate
    log = logging.getLogger('StrucTempExtModel V2')

    name: str

    def __init__(self, category: Category, name: str):
        super().__init__(category, 'StrucTemp_v2')
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
        template = StructuredTemplate()
        template.train(web_ids)
        self.template = template
        self.save()

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
        self.load()

        return self.template.extract(web_ids, self.category, k=k)
        # if len(web_ids) > 20:
        #     with Parallel(n_jobs=n_jobs, verbose=2) as parallel:
        #         data = parallel(delayed(extract_web_id)(web_id) for web_id in web_ids)
        # else:
        #     data = [extract_web_id(web_id) for web_id in web_ids]
        # return data

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
            self.template = StructuredTemplate.load(self.dir_path)
