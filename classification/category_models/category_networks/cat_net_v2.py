import random
from typing import List
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np

import time

from .base_category_network import BaseCategoryNetwork
from ...preprocessing import Category


class CategoryNetworkV0(BaseCategoryNetwork):

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='v2', description='MLP Try.')

    def predict(self, web_ids: List[str]) -> List[Category]:
        return [random.choice(Category) for _ in web_ids]



    def train(self, web_ids: List[str]) -> None:

        pass
