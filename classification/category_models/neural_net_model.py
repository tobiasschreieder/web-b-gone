from typing import List

from .base_model import BaseCategoryModel
from .category_networks import CategoryNetwork
from ..preprocessing import Category


class NeuralNetCategoryModel(BaseCategoryModel):
    """
    Classifies a website using a CategoryNetwork.
    """

    network: CategoryNetwork

    def __init__(self, name: str, version: str):
        super().__init__(version)
        self.network = CategoryNetwork.get(version)(name)
        self.dir_path = self.network.dir_path

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        self.network.load()
        return self.network.predict(web_ids)
