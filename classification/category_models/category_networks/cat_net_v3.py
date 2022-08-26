import random
from typing import List
import collections
import pathlib

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import tensorflow as tf

from keras import layers
from keras import losses
from keras import layers
from keras import losses
from keras.layers import TextVectorization, Dense
from keras.models import Sequential


from sklearn.model_selection import train_test_split
from tensorflow.python.keras.metrics import binary_accuracy

from .base_category_network import BaseCategoryNetwork
from ...preprocessing import Category, text_vectorization
from ...preprocessing.categorize_prepare import get_dict_from_feature_list, create_feature_list, get_true_categorys, \
    get_all_text_from_feature_list, get_dict_from_feature_list_inkl_keyword_result


tfds.disable_progress_bar()
class CategoryNetworkV3(BaseCategoryNetwork):

    def __init__(self, name: str = "Categorize_text_all", **kwargs):
        super().__init__(name=name, version='v3', description='Keras complete try.')

    def predict(self, web_ids: List[str]) -> List[Category]:
        """
        """
        self.load()  # self.load() later
        feature_dict = get_dict_from_feature_list_inkl_keyword_result(create_feature_list(web_ids), web_ids)
        X = pd.get_dummies(feature_dict.drop(['true_category', 'web_id'], axis=1))
        # loss, accuracy = self.model.predict(feature_dict)
        # print("Accuracy: {:2.2%}".format(binary_accuracy))
        return self.model.predict(X)


    def classification(self, web_ids: List[str]) -> List[Category]:
        return self.predict( web_ids)


    def train(self, web_ids: List[str]) -> None:
        #feature_dict = get_all_text_from_feature_list(create_feature_list(web_ids), web_ids)
        feature_dict = get_dict_from_feature_list_inkl_keyword_result(create_feature_list(web_ids), web_ids)
        result_list = get_true_categorys(web_ids)
        print(feature_dict)
        X_train = pd.get_dummies(feature_dict.drop(['true_category', 'web_id'], axis=1))
        y_train = feature_dict['true_category'].apply(lambda x: int(x))
        model = tf.keras.Sequential([layers.Dense(4)])
        model = Sequential()
        model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=9, activation='softmax'))  # softmax

        model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics='accuracy')
        model.fit(X_train, y_train, epochs=24, batch_size=32)
        self.model = model
        #self.model = self.build_model(text_vectorization.binary_vectorize_layer, text_vectorization.binary_model)
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
