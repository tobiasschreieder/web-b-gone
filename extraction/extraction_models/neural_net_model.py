from typing import List, Dict

from classification.preprocessing import Category
from .base_model import BaseExtractionModel
from .extraction_networks import ExtractionNetwork


class NeuralNetExtractionModel(BaseExtractionModel):

    network: ExtractionNetwork

    def __init__(self, category: Category, name: str, version: str):
        """
        Create NeuralNetExtractionModel for a category.

        :param category: Category for extraction.
        :param name: Name of the ExtractionNetwork
        :param version: Version of the ExtractionNetwork
        """
        super().__init__(category, version)
        self.network = ExtractionNetwork.get(version)(name)
        self.dir_path = self.dir_path.joinpath(name)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, web_ids: List[str], **kwargs) -> None:
        self.network.train(web_ids, **kwargs)
        self.save()

    def extract(self, web_ids: List[str], k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        self.load()
        return self.network.predict(web_ids, **kwargs)

    def save(self) -> None:
        self.network.save()

    def load(self) -> None:
        self.network.load()
