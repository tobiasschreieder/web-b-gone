import abc
from pathlib import Path
from typing import List

from config import Config
from ..preprocessing import Category

cfg = Config.get()


class BaseCategoryModel(abc.ABC):
    dir_path: Path = cfg.working_dir.joinpath(Path('models/classification/'))
    version: str

    def __init__(self, version: str, **kwargs):
        self.version = version
        self.dir_path = self.dir_path.joinpath(self.version)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        """
        Perform the classification of given web_ids.

        :param web_ids: List of web_ids for classification
        :param kwargs: additional parameter
        :return: List of predicted Categories
        """
        pass
