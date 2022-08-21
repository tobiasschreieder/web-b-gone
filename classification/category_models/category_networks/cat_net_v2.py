import random
from typing import List
import collections
import pathlib

import tensorflow as tf

from keras import layers
from keras import losses
from keras import utils
from keras.layers import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

import time

from tensorflow.python.keras.metrics import binary_accuracy

from .base_category_network import BaseCategoryNetwork
from ...preprocessing import Category, text_vectorization
from ...preprocessing.categorize_prepare import get_dict_from_feature_list, create_feature_list, get_true_categorys, \
    get_all_text_from_feature_list


class CategoryNetworkV2(BaseCategoryNetwork):

    def __init__(self, name: str = "Categorize_text_all", **kwargs):
        super().__init__(name=name, version='v2', description='binary_vectorization on all_text Try.')

    def predict(self, web_ids: List[str]) -> List[Category]:
        """
        """
        self.load()
        feature_dict = get_all_text_from_feature_list(create_feature_list(web_ids))
        # loss, accuracy = self.model.predict(feature_dict)
        # print("Accuracy: {:2.2%}".format(binary_accuracy))
        return self.model.predict(feature_dict)


    def classification(self, web_ids: List[str]) -> List[Category]:
        return self.predict( web_ids)


    def train(self, web_ids: List[str]) -> None:
        feature_dict = get_all_text_from_feature_list(create_feature_list(web_ids))
        result_list = get_true_categorys(web_ids)
        text_vectorization.tf_vectorisation_binary(feature_dict["text_all"], result_list)
        self.model = self.build_model(text_vectorization.binary_vectorize_layer, text_vectorization.binary_model)
        self.save()
        # (number of inputs + 1 output)/2 = #hiddenLayers
        # sigmoid for classifikation as activation
        pass

    def build_model(self, binary_vectorize_layer, binary_model):
        export_model = tf.keras.Sequential(
            [binary_vectorize_layer, binary_model,
             layers.Activation('softmax')])

        export_model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy'])
        return export_model
