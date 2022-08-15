from typing import List, Dict, Type

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from classification.category_models import BaseCategoryModel, NeuralNetCategoryModel
from classification.preprocessing import Category, GroundTruth, Website

AVERAGE = "macro"  # determines the type of averaging performed on the data, choose from "micro", "macro", "weighted"


def evaluate_classification(model_cls_classification: Type[BaseCategoryModel],
                            train_test_split: float,
                            max_size: int = -1, **model_kwargs) -> Dict[str, float]:
    """
    Evaluate a given classification model
    :param model_cls_classification: Classification model which should be used
    :param train_test_split: Specify proportion of train data
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param model_kwargs:
    :return: Dictionary with calculated metric scores
    """
    # Load and split data
    web_ids: List[str] = Website.get_website_ids(max_size=max_size, rdm_sample=True, seed='eval_class')
    split_index = int(len(web_ids) * train_test_split)
    train_ids = web_ids[:split_index]
    test_ids = web_ids[split_index:]

    # Classification
    model_classification: BaseCategoryModel

    if model_cls_classification == NeuralNetCategoryModel:
        model_classification: NeuralNetCategoryModel = model_cls_classification(**model_kwargs)
        model_classification.network.train(train_ids)
    else:
        model_classification = model_cls_classification(**model_kwargs)

    results_classification = classification_metrics(model_classification.classification(web_ids=test_ids),
                                                    [GroundTruth.load(web_id).category for web_id in test_ids])

    return results_classification


def format_data_classification(data: List[Category]) -> List[str]:
    """
    Format test data to List of Strings (Category.name)
    :param data: given List of data
    :return: formatted data
    """
    formatted_data = list()
    for d in data:
        formatted_data.append(d.name)

    return formatted_data


def create_confusion_matrix(pred: List[str], truth: List[str]) -> np.ndarray:
    """
    Create Confusion-Matrix
    :param truth: List with names of predicted categories
    :param pred: List with names of ground truth categories
    :return: Confusion-Matrix as Array
    """
    conf = confusion_matrix(y_true=truth, y_pred=pred)

    return conf


def classification_metrics(pred: List[Category], truth: List[Category]) -> Dict[str, float]:
    """
    Calculate Recall, Precision and F1 for classification model
    :param pred: List with predicted categories
    :param truth: List with ground-truth categories
    :return: Results as Dict
    """
    pred = format_data_classification(data=pred)
    truth = format_data_classification(data=truth)

    recall = round(recall_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)
    precision = round(precision_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)
    f1 = round(f1_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)

    results = {"recall": recall, "precision": precision, "f1": f1}

    return results
