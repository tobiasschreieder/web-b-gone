import logging

import numpy as np
import pandas as pd

from config import Config

import tensorflow as tf

from keras import layers
from keras import losses
from keras.layers import TextVectorization

cfg = Config.get()
log = logging.getLogger('text_vectorization')

"""
    
"""
binary_model = tf.keras.Sequential([layers.Dense(4)])
VOCAB_SIZE = 10000

binary_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='binary')  # bag-of-words
MAX_SEQUENCE_LENGTH = 250
int_model = tf.keras.Sequential([layers.Dense(4)])
int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

def binary_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return binary_vectorize_layer(text), label


def int_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return int_vectorize_layer(text), label


AUTOTUNE = tf.data.AUTOTUNE


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def tf_vectorisation_binary(train_data_list : pd.DataFrame, raw_val_ds): #raw_val_ds, raw_test_ds
    #train_data_list = train_data_list[["text_all", "true_category"]]
    #https://notebook.community/mitdbg/modeldb/client/workflows/demos/tf-text-classification
    train_text = tf.data.Dataset.from_tensor_slices((np.asarray(train_data_list["text_all"].astype('str')), np.asarray(train_data_list["true_category"].astype('str'))))
    binary_vectorize_layer.adapt(train_text)
    raw_val_ds = tf.data.Dataset.from_tensor_slices((np.asarray(raw_val_ds["text_all"].astype('str')), np.asarray(raw_val_ds["true_category"].astype('str'))))
    binary_vectorize_layer.adapt(raw_val_ds)

    binary_train_ds = train_text.map(binary_vectorize_text)
    binary_val_ds = raw_val_ds.map(binary_vectorize_text)
    #binary_test_ds = raw_test_ds.map(binary_vectorize_text)

    binary_train_ds = configure_dataset(binary_train_ds)
    binary_val_ds = configure_dataset(binary_val_ds)
    #binary_test_ds = configure_dataset(binary_test_ds)

    binary_model = tf.keras.Sequential([layers.Dense(4)])

    binary_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    history = binary_model.fit(binary_train_ds,validation_data=binary_val_ds, epochs=5)

"""
    history = binary_model.fit(
        binary_train_ds, validation_data=binary_val_ds, epochs=10)
    print("Linear model on binary vectorized data:")
    print(binary_model.summary())
    binary_loss, binary_accuracy = binary_model.evaluate(binary_test_ds)
    print("Binary model accuracy: {:2.2%}".format(binary_accuracy)) """



def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(num_labels)
    ])
    return model


def tf_vectorisation_int(train_text, raw_train_ds, raw_val_ds, raw_test_ds):
    int_vectorize_layer.adapt(train_text)

    int_train_ds = raw_train_ds.map(int_vectorize_text)
    int_val_ds = raw_val_ds.map(int_vectorize_text)
    int_test_ds = raw_test_ds.map(int_vectorize_text)

    int_train_ds = configure_dataset(int_train_ds)
    int_val_ds = configure_dataset(int_val_ds)
    int_test_ds = configure_dataset(int_test_ds)

    # `vocab_size` is `VOCAB_SIZE + 1` since `0` is used additionally for padding.
    int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4)
    int_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)
    print("ConvNet model on int vectorized data:")
    print(int_model.summary())
    int_loss, int_accuracy = int_model.evaluate(int_test_ds)
    print("Int model accuracy: {:2.2%}".format(int_accuracy))



