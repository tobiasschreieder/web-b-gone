from typing import List, Dict

from classification.preprocessing import Category
from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkAutoV0(BaseExtractionNetwork):

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='AutoV0', description='Does nothing. Only for setup purpose.')

    def predict(self, web_ids: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        return [{key: [] for key in Category.BOOK.get_attribute_names()} for _ in web_ids]

    def train(self, web_ids: List[str], **kwargs) -> None:
        pass
