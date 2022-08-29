import random
from typing import List, Any

from .base_model import BaseCategoryModel
from ..preprocessing import Category


class RandomCategoryModel(BaseCategoryModel):
    """
    Classifies a website using a randomizer.
    """

    def __init__(self, seed: Any = None):
        super().__init__('Random')
        self.seed = seed

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        if self.seed is not None:
            random.seed(self.seed)
        return [random.choice([cat for cat in Category]) for _ in web_ids]
