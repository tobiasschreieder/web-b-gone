from typing import List

from .base_model import BaseCategoryModel
from .category_networks import CategoryNetwork
from ..preprocessing import Category


class NeuralNetCategoryModel(BaseCategoryModel):

    network: CategoryNetwork

    def __init__(self, name: str, version: str):
        super().__init__()
        self.network = CategoryNetwork.get(version)(name)
        self.network.load()

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        return self.network.predict(web_ids)
