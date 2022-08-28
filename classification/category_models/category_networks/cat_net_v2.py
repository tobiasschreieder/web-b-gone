from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
from keras import layers
from keras import losses
from keras.layers import TextVectorization
from tqdm import tqdm

from .base_category_network import BaseCategoryNetwork
from ...preprocessing import Category, Website

tfds.disable_progress_bar()


"""
Because of my previous problems, I'll keep this so I can always reset to this step or at least refer back 
to the minimum working version.
"""
class CategoryNetworkV2(BaseCategoryNetwork):
    EMBEDDING_DIM = 128
    MAX_TOKENS = 20000
    EPOCHS = 20

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, version='v2', description='Using text from html in a TextVectorizationLayer.')

    """
    Method loads the saved trained model and predicts the categories for the input web_ids.
    Preparation of the input generated by the web_ids is handled through the method 
    get_df_from_feature_list_inkl_keyword_result which returns a dataframe with the web_ids as index.
    """

    def predict(self, web_ids: List[str]) -> List[Category]:
        self.load()
        df_x = self.create_x(web_ids)
        X = tf.convert_to_tensor(df_x['text_all'])
        y_hat = self.model.predict(X)
        return [Category.get(int(np.argmax(y))) for y in y_hat]

    @staticmethod
    def create_x(web_ids: List[str], mute=False) -> pd.DataFrame:
        text_data = []
        cat_data = []
        id_data = []
        for web_id in tqdm(web_ids, desc='Preprocessing web_ids', disable=mute):
            entry = Website.load(web_id)
            with entry.file_path.open(encoding='utf-8') as fp:
                text_data.append(''.join(BeautifulSoup(fp, features="html.parser").get_text()).replace('\n', ' '))
                cat_data.append(int(entry.truth.category))
                id_data.append(web_id)
        df = pd.DataFrame(data={'text_all': text_data, 'true_category': cat_data, 'web_id': id_data})
        df.set_index('web_id', inplace=True)
        return df

    """
    Method is base method of a classificator method and forwards into the predict method.
    Preparation of the input generated by the web_ids is handled through the method 
    get_df_from_feature_list_inkl_keyword_result which returns a dataframe with the web_ids as index.
    The keras model built here, has 8 output classes (9 will be none for keyword based), uses softmax as the last layer.
    Optimizer is adam, SparseCategoricalCrossentropy, and the accuracy metrics.
    Training length here is only used for testing: epochs=24, batch_size=32
    """

    def train(self, web_ids: List[str]) -> None:
        # feature_dict = get_df_from_feature_list_str_features(create_feature_list(web_ids))
        df_x = self.create_x(web_ids)
        X = tf.convert_to_tensor(df_x['text_all'])
        Y = df_x.pop('true_category')
        vectorize_layer = TextVectorization(
            standardize='lower_and_strip_punctuation',
            max_tokens=self.MAX_TOKENS,
            output_mode='int',
            output_sequence_length=500)
        vectorize_layer.adapt(X)
        text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text')
        x = vectorize_layer(text_input)
        x = layers.Embedding(self.MAX_TOKENS + 1, self.EMBEDDING_DIM)(x)
        x = layers.Dropout(0.5)(x)

        # Conv1D + global max pooling
        x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
        x = layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3)(x)
        x = layers.GlobalMaxPooling1D()(x)

        # We add a vanilla hidden layer:
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        predictions = layers.Dense(8, activation='softmax', name='predictions')(x)

        model = tf.keras.Model(text_input, predictions)

        model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics='accuracy')
        model.fit(X, Y, epochs=self.EPOCHS, batch_size=32)
        self.model = model
        self.save()
        pass
