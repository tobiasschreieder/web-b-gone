from typing import List, Dict, Type, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from classification.category_models import BaseCategoryModel, NeuralNetCategoryModel
from classification.preprocessing import Category, GroundTruth, Website

AVERAGE = "macro"  # determines the type of averaging performed on the data, choose from "micro", "macro", "weighted"


def evaluate_classification(model_cls_classification: Type[BaseCategoryModel],
                            train_test_split: float,
                            max_size: int = -1, **model_kwargs) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a given classification model
    :param model_cls_classification: Classification model which should be used
    :param train_test_split: Specify proportion of train data [0; 1]
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param model_kwargs:
    :return: Dictionary with calculated metric scores
    """
    # Load and split data
    train_ids: List[str]
    test_ids: List[str]
    train_ids, test_ids = split_data(train_test_split=train_test_split, split_type="website", max_size=max_size)

    # Classification
    model_classification: BaseCategoryModel

    if model_cls_classification == NeuralNetCategoryModel:
        model_classification: NeuralNetCategoryModel = model_cls_classification(**model_kwargs)
        model_classification.network.train(train_ids)
    else:
        model_classification = model_cls_classification(**model_kwargs)

    # out of sample prediction
    if len(test_ids) != 0:
        results_classification_test = classification_metrics(model_classification.classification(web_ids=test_ids),
                                                        [GroundTruth.load(web_id).category for web_id in test_ids])
    else:
        results_classification_test = {"recall": None, "precision": None, "f1": None}

    # in sample prediction
    if len(train_ids) != 0:
        results_classification_train = classification_metrics(model_classification.classification(web_ids=train_ids),
                                                        [GroundTruth.load(web_id).category for web_id in train_ids])
    else:
        results_classification_train = {"recall": None, "precision": None, "f1": None}

    results = {"out of sample": results_classification_test, "in sample": results_classification_train}

    return results


def split_data(train_test_split: float, split_type: str, max_size: int = -1,
               seed: str = "eval_class") -> Tuple[List[str], List[str]]:
    """
    Method to split dataset with defined split-type
    :param train_test_split: Specify proportion of train data [0; 1]
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param seed: String with seed
    :return: Tuple with train-ids and test-ids
    """
    train_ids = list()
    test_ids = list()

    if split_type == "website":
        web_ids: List[str] = Website.get_website_ids(max_size=max_size, rdm_sample=True, seed=seed)

        split_index = int(len(web_ids) * train_test_split)
        train_ids = web_ids[:split_index]
        test_ids = web_ids[split_index:]

    elif split_type == "domain":
        categories = [cat for cat in Category]

        cat_size = int(max_size / len(categories))
        for cat in categories:
            domains = Website.get_all_domains(category=cat)
            split_index = int(len(domains) * train_test_split)
            train_domains = domains[:split_index]
            test_domains = domains[split_index:]

            train_ids += Website.get_website_ids(max_size=int(cat_size*train_test_split), rdm_sample=True, seed=seed, categories=cat,
                                                 domains=train_domains)

            test_ids += Website.get_website_ids(max_size=int(cat_size*(1-train_test_split)), rdm_sample=True, seed=seed, categories=cat,
                                                domains=test_domains)

    else:
        return [], []

    return train_ids, test_ids


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
