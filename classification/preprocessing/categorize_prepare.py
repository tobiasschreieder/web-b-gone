import logging
from typing import List, Dict
from bs4 import BeautifulSoup
from pathlib import Path

from config import Config
from .keyword_categorize import find_class
from ..preprocessing import GroundTruth, Category, Website

cfg = Config.get()
log = logging.getLogger('categorize_prepare')


def get_true_categorys(web_ids: List[str]) -> List[Category]:
    """
    Returns the corresponding list of groundtruth categories to the web id list input.
    """
    true_cats = []
    for id in web_ids:
        gt = GroundTruth.load(id)
        true_cats.append(gt.category)
    return true_cats


class FeatureHolder:
    web_id: str
    html: str
    url: str
    head: str
    title: List[str]
    link: List[str]
    largest_content: str
    domain_name: str
    text_all: str


def create_feature_list(web_ids: List[str]) -> List[FeatureHolder]:
    feature_list = []
    # text: n-gram with feature selection - what is already in preprocess?
    # html tags: try different weights for text in html tags
    for web_id in web_ids:
        fh = FeatureHolder()
        fh.web_id = web_id
        website = Website.load(web_id)
        fh.url = website.url
        fh.domain_name = website.domain_name
        # https://beautiful-soup-4.readthedocs.io/en/latest/
        # soup = BeautifulSoup(html_doc, 'html.parser') #lxml faster alternative
        with Path(website.file_path).open(encoding='utf-8') as fp:
            soup = BeautifulSoup(fp)
            for link in soup.find_all('a'):
                fh.link.append(link.get('href'))
            for title in soup.find_all('title'):
                fh.title.append(title)
            fh.head = soup.head
            # soup.get_attribute_list()
            text_all = soup.get_text()
            fh.text_all = ' '.join(text_all)
            feature_list.append(fh)
    # hyperlinks
    return feature_list


def get_dict_from_feature_list(feature_list: List[FeatureHolder]) -> Dict:
    train_dict = {}
    for f in feature_list:
        train_dict.append({"web_id": f.web_id, "url": f.url, "head": f.head, "title": f.title,
                           "domain_name": f.domain_name, "html": f.html, "largest_content": f.largest_content,
                           "hyperlinks": f.link, "text_all": f.text_all})
    return train_dict


def get_dict_from_feature_list_inkl_keyword_result(feature_list: List[FeatureHolder]) -> Dict:
    train_dict = {}
    for f in feature_list:
        train_dict.append({"web_id": f.web_id, "url": f.url, "head": f.head, "title": f.title,
                           "domain_name": f.domain_name, "html": f.html, "largest_content": f.largest_content,
                           "hyperlinks": f.link, "text_all": f.text_all,
                           "keyword_result": find_class(f.html)})
    return train_dict
