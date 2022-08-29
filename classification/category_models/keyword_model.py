from typing import List, Any

from bs4 import BeautifulSoup
from tqdm import tqdm

from .base_model import BaseCategoryModel
from ..preprocessing import Category, Website
from ..preprocessing.keyword_categorize import find_class


class KeywordModel(BaseCategoryModel):
    """
    Classifies a website using keywords.
    """

    def __init__(self, seed: Any = None):
        super().__init__('Keyword')
        self.seed = seed

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        categories = []
        for web_id in tqdm(web_ids):
            with Website.load(web_id).file_path.open(encoding='utf-8') as fp:
                text = ''.join(BeautifulSoup(fp, features="html.parser").get_text()).replace('\n', ' ')
                categories.append(find_class(text)[0])

        return categories
