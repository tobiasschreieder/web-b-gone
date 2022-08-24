import logging
from typing import List, Dict
from pathlib import Path

from classification.preprocessing import Category
from config import Config
from .base_model import BaseExtractionModel
from ..structure_helper_v2 import StructuredTemplate
from .extraction_networks.base_extraction_network import ExtractionNetwork

cfg = Config.get()


class CombinedExtractionModel(BaseExtractionModel):
    template: StructuredTemplate
    ner_network: ExtractionNetwork
    log = logging.getLogger('CombinedExtModel')

    dir_path: Path = cfg.working_dir.joinpath(Path('models/extraction/'))
    name: str

    def __init__(self, category: Category, name: str):
        super().__init__(category)
        self.name = name
        self.ner_network = ExtractionNetwork.get('NerV2')(name)
        self.template = None
        self.dir_path = self.dir_path.joinpath('strucTemp').joinpath(self.name)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Learn from the given website ids.

        :param web_ids: Website ids to learn the template from.
        :return: None
        """
        self.template = StructuredTemplate()

        # do Stuff

        self.ner_network.train(web_ids)

        # do Stuff

        self.ner_network.save()
        self.template.save(self.dir_path)

    def extract(self, web_ids: List[str], n_jobs: int = -1, k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract information from websites using the combined approach.

        :param web_ids: List of website ids for extraction
        :param n_jobs: the number of processes to use, if -1 use all,
            if < -1 use max_processes+1+n_jobs, example n_jobs = -2 -> use all processors except 1.
            see joblib.parallel.Parallel
        :param k: number of results per attribute, default: 3
        :return: Extracted information
        """
        self.ner_network.load()
        if self.template is None:
            self.template = StructuredTemplate.load(self.dir_path)

        # do Stuff
        pass