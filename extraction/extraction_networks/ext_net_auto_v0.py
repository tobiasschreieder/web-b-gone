from typing import List

from .base_extraction_network import BaseExtractionNetwork


class ExtractionNetworkAutoV0(BaseExtractionNetwork):

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='AutoV0', description='Does nothing. Only for setup purpose.')

    def predict(self, web_ids: List[str]) -> List[List[str]]:
        return [[] for _ in web_ids]

    def train(self, web_ids: List[str]) -> None:
        pass
