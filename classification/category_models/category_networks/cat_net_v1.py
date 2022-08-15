import random
from typing import List
from sklearn import tree
import graphviz
from sklearn.tree import export_text

from config import Config

from .base_category_network import BaseCategoryNetwork
from ...preprocessing import Category, Website
from ...preprocessing.categorize_prepare import get_true_categorys, create_feature_list


class CategoryNetworkV1(BaseCategoryNetwork):
    clf: tree.DecisionTreeClassifier()

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='v1', description='DecisionTree cart.')
        ##*, criterion='gini', splitter='best', max_depth=3, min_samples_split=2, min_samples_leaf=1,
        # min_weight_fraction_leaf=0.0, max_features=None, random_state=0, max_leaf_nodes=None,
        # min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0
        clf = tree.DecisionTreeClassifier(max_depth=3, random_state=0)

    def predict(self, web_ids: List[str]) -> List[Category]:
        website_with_feauture = create_feature_list(web_ids)
        self.print_tree_text()
        return self.clf.predict(website_with_feauture)

    def train(self, web_ids: List[str]) -> None:
        website_with_feauture = create_feature_list(web_ids)
        self.clf.fit(website_with_feauture, get_true_categorys(web_ids))
        pass

    def plot(self):
        tree.plot_tree(self.clf)

    def export_plot_graphviz(self):
        dot_data = tree.export_graphviz(self.clf, out_file=Config.output_dir + '')
        graph = graphviz.Source(dot_data)
        graph.render("iris")

    def print_tree_text(self):
        tree_text = export_text(self.clf, feature_names=['web_id', 'html', 'url', 'head', 'title', 'link',
                                                         'largest_content', 'domain_name', 'text_all'])
        print(tree_text)

