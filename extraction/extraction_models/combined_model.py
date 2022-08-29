import logging
from pathlib import Path
from typing import List, Dict

from classification.preprocessing import Category
from config import Config
from evaluation import text_preprocessing
from .base_model import BaseExtractionModel
from .extraction_networks.base_extraction_network import ExtractionNetwork
from ..structure_tree_template import StructuredTreeTemplate

cfg = Config.get()


class CombinedExtractionModel(BaseExtractionModel):
    """
    Combined Template-NER-Model.
    Combine StructuredTreeTemplate and NERV1 ExtractionNetwork
    """

    template: StructuredTreeTemplate
    ner_network: ExtractionNetwork
    log = logging.getLogger('CombinedExtModel')

    struc_path: Path = cfg.working_dir.joinpath('models/extraction/').joinpath('StrucTemp')

    def __init__(self, category: Category, ner_name: str, struc_name: str):
        """
        Create CombinedExtractionModel for a category.

        :param category: Category for extraction.
        :param ner_name: Name of the ExtractionNetwork NerV1
        :param struc_name: Name of the StructuredTreeTemplate
        """
        super().__init__(category, 'CombinedStrucNer')
        self.ner_network = ExtractionNetwork.get('NerV1')(ner_name)
        self.template = None
        self.struc_path = self.struc_path.joinpath(struc_name)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Method not implemented for CombinedExtractionModel.

        :param web_ids: Website ids to learn the template from.
        :return: None
        """
        raise NotImplementedError(f'CombinedExtractionModel needs pretrained models of '
                                  f'StructuredTemplate and ExtractionNetwork version: NerV1 to work. ')

    def extract(self, web_ids: List[str], k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract information from websites using the combined approach.

        :param web_ids: List of website ids for extraction
        :param k: number of results per attribute, default: 3
        :return: Extracted information
        """
        self.load()

        ner_result = self.ner_network.predict(web_ids)
        structure_result = self.template.extract(web_ids, self.category, k=k,
                                                 with_score=True, only_perfect_match=True)

        epsilon = 15

        combine_result = []
        for id_result_ner, id_result_struct in zip(ner_result, structure_result):
            good_matches = 0
            id_result = {}
            for attr in id_result_ner:
                ner_result = id_result_ner[attr]
                structure_result = id_result_struct[attr]
                id_result[attr] = []
                for score, candiate_struc in structure_result:
                    if not candiate_struc or score < epsilon:
                        continue
                    if isinstance(candiate_struc, str):
                        candiate_struc = text_preprocessing.preprocess_text_html(candiate_struc)
                    else:
                        continue
                    for candidate_ner in ner_result:
                        if not candidate_ner:
                            continue
                        if isinstance(candiate_struc, str):
                            candidate_ner = text_preprocessing.preprocess_text_html(candidate_ner)
                        else:
                            continue
                        if candidate_ner in candiate_struc or candiate_struc in candidate_ner:
                            good_matches += 1
                            id_result[attr].append(candiate_struc)
                            continue
                if not id_result[attr]:
                    id_result[attr] = ner_result

            combine_result.append(id_result)

        return combine_result

    def load(self) -> None:
        self.ner_network.load()
        if self.template is None:
            self.template = StructuredTreeTemplate.load(self.struc_path)
