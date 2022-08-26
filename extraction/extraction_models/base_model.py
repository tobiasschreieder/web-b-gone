import abc
from pathlib import Path
from typing import List, Dict

from classification.preprocessing import Category
from config import Config

cfg = Config.get()


class BaseExtractionModel(abc.ABC):

    version: str
    dir_path: Path

    def __init__(self, category: Category, version: str, **kwargs):
        self.category = category
        self.version = version
        self.dir_path = cfg.working_dir.joinpath('models/extraction/').joinpath(version)
        self.dir_path.mkdir(parents=True, exist_ok=True)

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

    def save(self) -> None:
        """
        TODO
        :return:
        """
        pass

    def load(self) -> None:
        """
        TODO
        :return:
        """
        pass
