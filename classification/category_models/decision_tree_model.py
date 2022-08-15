import random
from typing import List, Any
from sklearn import tree

from .base_model import BaseCategoryModel
from .category_networks import CategoryNetwork
from ..preprocessing import Category


class DecisionTreeC5CategoryModel(BaseCategoryModel):

    network: CategoryNetwork

    def __init__(self, name: str, version: str):
        super().__init__()
        self.network = CategoryNetwork.get(version)(name)
        self.network.load()

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        """
        :param web_ids:
        :param kwargs:
        :return:
        """
        return self.network.predict(web_ids)