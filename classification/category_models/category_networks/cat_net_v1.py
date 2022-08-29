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


class CategoryNetworkV1(BaseCategoryNetwork):
    """
    Categorize a website using a keras neural net model and the full website text.
    """
    EMBEDDING_DIM = 128
    MAX_TOKENS = 20000
    EPOCHS = 20

    def __init__(self, name: str):
        super().__init__(name=name, version='v1', description='Using text from html in a TextVectorizationLayer.')

    def predict(self, web_ids: List[str]) -> List[Category]:
        self.load()
        df_x = self.create_x(web_ids)
        x = tf.convert_to_tensor(df_x['text_all'])
        y_hat = self.model.predict(x)
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

    def train(self, web_ids: List[str]) -> None:
        df_x = self.create_x(web_ids)
        x = tf.convert_to_tensor(df_x['text_all'])
        y = df_x.pop('true_category')
        vectorize_layer = TextVectorization(
            standardize='lower_and_strip_punctuation',
            max_tokens=self.MAX_TOKENS,
            output_mode='int',
            output_sequence_length=500)
        vectorize_layer.adapt(x)
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
        model.fit(x, y, epochs=self.EPOCHS, batch_size=32)
        self.model = model
        self.save()
        pass
