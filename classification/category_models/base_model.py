import abc
from typing import List

from ..preprocessing import Category


class BaseCategoryModel(abc.ABC):

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        """
        TODO
        :param web_ids:
        :param kwargs:
        :return:
        """
        pass
