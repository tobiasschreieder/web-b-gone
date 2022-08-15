import abc
from typing import List, Dict

from classification.preprocessing import Category


class BaseExtractionModel(abc.ABC):

    def __init__(self, category: Category, **kwargs):
        self.category = category

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        pass

    @abc.abstractmethod
    def extract(self, web_ids: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        pass
