from typing import List, Dict

from classification.preprocessing import Category
from .base_model import BaseExtractionModel
from .extraction_networks import ExtractionNetwork


class NeuralNetExtractionModel(BaseExtractionModel):

    network: ExtractionNetwork

    def __init__(self, category: Category, name: str, version: str):
        super().__init__(category, version)
        self.network = ExtractionNetwork.get(version)(name)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        self.network.train(web_ids, **kwargs)
        self.save()

    def extract(self, web_ids: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        self.load()
        return self.network.predict(web_ids, **kwargs)

    def save(self) -> None:
        """
        TODO
        :return:
        """
        self.network.save()

    def load(self) -> None:
        """
        TODO
        :return:
        """
        self.network.load()
