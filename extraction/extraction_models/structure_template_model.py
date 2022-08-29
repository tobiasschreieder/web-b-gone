import logging
from typing import List, Dict

from classification.preprocessing import Category
from config import Config
from .base_model import BaseExtractionModel
from ..structure_tree_template import StructuredTreeTemplate

cfg = Config.get()


class StrucTempExtractionModel(BaseExtractionModel):
    template: StructuredTreeTemplate
    log = logging.getLogger('StrucTempExtModel')

    name: str

    def __init__(self, category: Category, name: str):
        """
        Create StrucTempExtractionModel for a category.

        :param category: Category for extraction.
        :param name: Name of the StructuredTreeTemplate
        """
        super().__init__(category, 'StrucTemp')
        self.name = name
        self.template = None
        self.dir_path = self.dir_path.joinpath(self.name)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def train(self, web_ids: List[str], **kwargs) -> None:
        """
        Learn a structured template from the given website ids.

        :param web_ids: Website ids to learn the template from.
        :return: None
        """
        template = StructuredTreeTemplate()
        template.train(web_ids)
        self.template = template
        self.template.save(self.dir_path)

    def extract(self, web_ids: List[str], k: int = 3, **kwargs) -> List[Dict[str, List[str]]]:
        """
        Extract information from websites using the structured template approach.

        :param web_ids: List of website ids for extraction
        :param k: number of results per attribute, default: 3
        :return: Extracted information
        """
        if self.template is None:
            self.template = StructuredTreeTemplate.load(self.dir_path)

        return self.template.extract(web_ids, self.category, k=k)

    def save(self) -> None:
        """
        Save template to disk.

        :return: None
        """
        self.template.save(self.dir_path)

    def load(self) -> None:
        """
         Load template from disk.

        :return: None
        """
        if self.template is None:
            self.template = StructuredTreeTemplate.load(self.dir_path)
