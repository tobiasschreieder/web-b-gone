import random
from typing import List, Any

from .base_model import BaseCategoryModel
from ..preprocessing import Category


class RandomCategoryModel(BaseCategoryModel):

    def __init__(self, seed: Any = None):
        self.seed = seed

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        if self.seed is not None:
            random.seed(self.seed)
        return [random.choice(Category) for _ in web_ids]
