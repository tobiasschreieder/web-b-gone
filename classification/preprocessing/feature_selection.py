import logging
from typing import List, Dict


from config import Config
from data_storage import GroundTruth, Category

cfg = Config.get()
log = logging.getLogger('categorize_prepare')
# TODO PLACEHOLDER CLASS
# https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection

#def get_true_categorys( web_ids: List[str]) -> List[Category]:
"""
    Returns the corresponding list of groundtruth categories to the web id list input.
"""
"""
    true_cats = []
    for id in web_ids:
        gt = GroundTruth.load(id)
        true_cats.append(gt.category)
    return true_cats
"""



