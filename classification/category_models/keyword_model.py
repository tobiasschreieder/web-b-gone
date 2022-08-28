import random
from typing import List, Any

from tqdm import tqdm

from .base_model import BaseCategoryModel
from ..preprocessing import Category
from ..preprocessing.categorize_prepare import get_all_text_from_feature_list, create_feature_list
from ..preprocessing.keyword_categorize import find_class


class KeywordModel(BaseCategoryModel):

    def __init__(self, seed: Any = None):
        super().__init__('Keyword')
        self.seed = seed

    def classification(self, web_ids: List[str], **kwargs) -> List[Category]:
        # features = create_feature_list(web_ids)
        #
        # all_text_list = get_all_text_from_feature_list(features)
        # categories = []
        # for all_text in all_text_list:
        #     cat_pair = find_class(all_text)
        #     categories.append(cat_pair[0])

        categories = []
        for web_id in tqdm(web_ids):
            features = create_feature_list([web_id])
            all_text = get_all_text_from_feature_list(features)[0]
            categories.append(find_class(all_text)[0])

        return categories
