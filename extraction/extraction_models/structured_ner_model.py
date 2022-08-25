import logging
from typing import List, Dict
from pathlib import Path

from classification.preprocessing import Category, Website
from config import Config
from .base_model import BaseExtractionModel
from .. import nerHelper
from ..structure_helper_v2 import StructuredTemplate
from .extraction_networks.base_extraction_network import ExtractionNetwork
from evaluation import text_preprocessing

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
        self.struc_name = struc_name
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

        ner_result = self.ner_network.predict(web_ids)
        structure_result = self.template.extract(web_ids, self.category, k=k, with_score=True)

        EPSILON = 12

        combine_result = []
        for id_result_ner, id_result_struct in zip(ner_result, structure_result):
            good_matches = 0
            id_result = {}
            for attr in id_result_ner:
                ner_result = id_result_ner[attr]
                structure_result = id_result_struct[attr]
                id_result[attr] = []
                for score, candiate_struc in structure_result:
                    if not candiate_struc or score < EPSILON:
                        continue
                    candiate_struc = text_preprocessing.preprocess_text_html(candiate_struc)
                    for candidate_ner in ner_result:
                        if not candidate_ner:
                            continue
                        candidate_ner = text_preprocessing.preprocess_text_html(candidate_ner)
                        if candidate_ner in candiate_struc or candiate_struc in candidate_ner:
                            good_matches += 1
                            id_result[attr].append(candiate_struc)
                            continue
                if not id_result[attr]:
                    id_result[attr].append(ner_result)

            if good_matches > len(id_result_ner)/2:
                new_dict = {}
                for attr in id_result_struct:
                    new_dict[attr] = []
                    for tupel in id_result_struct[attr]:
                        if tupel[1] and tupel[0] > EPSILON/2:
                            new_dict[attr].append(tupel[1])
                combine_result.append(new_dict)
            else:
                combine_result.append(id_result)

        return combine_result
