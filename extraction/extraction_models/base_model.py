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
        Train the model on given web_ids.

        :param web_ids: web_ids to train on.
        :param kwargs: additional parameters.
        :return: None
        """
        pass

    @abc.abstractmethod
    def extract(self, web_ids: List[str], k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract the attributes from the given Websites.
        Returns a dictionary for each website where the keys are the extracted attributes
        and the values are lists with the extracted text.

        :param k: number of extract values per attribute.
        :param web_ids: websites to extract attribute from.
        :param kwargs: additional parameters.
        :return: List of dictionaries with extracted text.
        """
        pass

    def save(self) -> None:
        """
        Save model to disk.

        :return: None
        """
        pass

    def load(self) -> None:
        """
         Load model from disk.

        :return: None
        """
        pass
