import logging
from typing import List, Dict


from config import Config
from data_storage import GroundTruth, Category

cfg = Config.get()
log = logging.getLogger('categorize_prepare')


def get_true_categorys( web_ids: List[str]) -> List[Category]:
    """
    Returns the corresponding list of groundtruth categories to the web id list input.
    """
    true_cats = []
    for id in web_ids:
        gt = GroundTruth.load(id)
        true_cats.append(gt.category)
    return true_cats

def create_feature_list(web_ids: List[str]):
        # TODO placeholder
    #text: n-gram with feature selection - what is already in preprocess?
    #html tags: try different weights for text in html tags
    #hyperlinks



