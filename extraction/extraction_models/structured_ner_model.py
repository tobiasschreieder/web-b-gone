import logging
from typing import List, Dict
from pathlib import Path

from classification.preprocessing import Category, Website
from config import Config
from .base_model import BaseExtractionModel
from .. import nerHelper
from ..structure_helper_v2 import StructuredTemplate
from .extraction_networks.base_extraction_network import ExtractionNetwork

cfg = Config.get()


class CombinedExtractionModel(BaseExtractionModel):
    template: StructuredTemplate
    ner_network: ExtractionNetwork
    log = logging.getLogger('CombinedExtModel')

    dir_path: Path = cfg.working_dir.joinpath(Path('models/extraction/'))
    ner_name: str
    struc_name: str

    def __init__(self, category: Category, ner_name: str, struc_name: str):
        super().__init__(category)
        self.ner_name = ner_name
        self.ner_name = 'all_doms'
        self.struc_name = struc_name
        self.struc_name = 'all_doms'
        self.ner_network = ExtractionNetwork.get('NerV1')(self.ner_name)
        self.template = None
        self.dir_path = self.dir_path.joinpath('strucTemp').joinpath(self.struc_name)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Method not implemented for CombinedExtractionModel.

        :param web_ids: Website ids to learn the template from.
        :return: None
        """
        raise NotImplementedError(f'CombinedExtractionModel needs pretrained models of '
                                  f'StructuredTemplate and ExtractionNetwork version: NerV2 to work. ')

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

        print(web_ids)
        ner_result = self.ner_network.predict(web_ids)
        print("NER:", ner_result)
        structure_result = self.template.extract(web_ids, self.category, k=k)
        print("STRUCTURE:", structure_result)

        # wenn bei mehr als 50% ein Overlap, dann nimm komplpett struc-Modell
        # score von structure-Modell wenn über 10 dann nimm alles von structure

        # wenn weniger, nimm immer den Overlap und vom Rest dann das NER-Modell
        pass
