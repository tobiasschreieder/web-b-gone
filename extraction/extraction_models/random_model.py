import random
from pathlib import Path
from typing import List, Any, Dict

from bs4 import BeautifulSoup

from classification.preprocessing import Website, Category
from .base_model import BaseExtractionModel


def _get_rdm_words(text: str) -> str:
    text_split = text.split(' ')
    length = random.randint(1, 10)
    pos = random.randint(0, len(text_split) - length)
    return ' '.join(text_split[pos:pos + length])


class RandomExtractionModel(BaseExtractionModel):
    """
    ExtractionModel extracting random words from the HTML.
    """

    def __init__(self, category: Category, seed: Any = None):
        """
        Create RandomExtractionModel for a category.

        :param category: Category for extraction.
        :param seed: Seed for the randomizer
        """
        super().__init__(category, 'rdm_model')
        self.seed = seed

    def extract(self, web_ids: List[str], **kwargs) -> List[Dict[str, List[str]]]:
        if self.seed is not None:
            random.seed(self.seed)

        result = []

        for web_id in web_ids:
            website = Website.load(web_id)
            with Path(website.file_path).open(encoding='utf-8') as htm_file:
                soup = BeautifulSoup(htm_file, features="html.parser")
                htm_text = soup.get_text(separator=' ')

                result.append({
                    key: [_get_rdm_words(htm_text) for _ in range(random.randint(0, 3))]
                    for key in self.category.get_attribute_names()
                })

        return result
