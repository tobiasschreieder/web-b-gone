import random
from typing import List

from .base_category_network import BaseCategoryNetwork
from ...preprocessing import Category


class CategoryNetworkV0(BaseCategoryNetwork):

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='v0', description='Does nothing. Only for setup purpose.')

    def predict(self, web_ids: List[str]) -> List[Category]:
        return [random.choice(Category) for _ in web_ids]

    def train(self, web_ids: List[str]) -> None:
        pass
