from typing import List, Dict, Type, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from classification.category_models import BaseCategoryModel, NeuralNetCategoryModel
from classification.preprocessing import Category, GroundTruth, Website

AVERAGE = "macro"  # determines the type of averaging performed on the data, choose from "micro", "macro", "weighted"


def evaluate_classification(model_cls_classification: Type[BaseCategoryModel],
                            train_test_split: float,
                            max_size: int = -1,
                            split_type: str = "website",
                            save_results: bool = True, **model_kwargs) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a given classification model
    :param model_cls_classification: Classification model which should be used
    :param train_test_split: Specify proportion of train data [0; 1]
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param save_results: Boolean -> Set to False if no MD-File should be created
    :param model_kwargs:
    :return: Dictionary with calculated metric scores
    """
    # Load and split data
    train_ids: List[str]
    test_ids: List[str]
    train_ids, test_ids = split_data(train_test_split=train_test_split, split_type=split_type, max_size=max_size)

    # Classification
    model_classification: BaseCategoryModel

    if model_cls_classification == NeuralNetCategoryModel:
        model_classification: NeuralNetCategoryModel = model_cls_classification(**model_kwargs)
        model_classification.network.train(web_ids=train_ids)
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

    # combine results
    results = {"out of sample": results_classification_test, "in sample": results_classification_train}

    # save results as MD-File
    if save_results:
        parameters = {"Model": model_cls_classification, "Data-split": split_type,
                      "Size dataset": max_size, "Train-Test-Split": train_test_split, "Averaging method": AVERAGE}
        create_md_file(results=results, parameters=parameters)

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

            train_ids += Website.get_website_ids(max_size=int(cat_size*train_test_split), rdm_sample=True, seed=seed,
                                                 categories=cat, domains=train_domains)

            test_ids += Website.get_website_ids(max_size=int(cat_size*(1-train_test_split)), rdm_sample=True, seed=seed,
                                                categories=cat, domains=test_domains)

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


def create_md_file(results: Dict[str, Dict[str, float]], parameters: Dict[str, str], name: str = "",
                   path: str = "working/"):
    """
    Create MD-File for extraction results
    :param results: Dictionary with calculated results from extraction model
    :param parameters: Dictionary with all parameter that should be listed in file
    :param name: String with name of considered model
    :param path: String with path to save MD-File
    """
    # Header
    text = list()
    text.append("# Evaluation Classification")
    if len(parameters) != 0:
        text.append("## Parameters:")
        for k, v in parameters.items():
            text.append("* " + str(k) + ": " + str(v))

    # In-sample prediction
    text.append("## In-sample Prediction:")
    text.append("| Metric | Result |")
    text.append("|---|---|")
    text.append("| Recall | " + str(results["in sample"]["recall"]) + " |")
    text.append("| Precision | " + str(results["in sample"]["precision"]) + " |")
    text.append("| F1 | " + str(results["in sample"]["f1"]) + " |")

    # Out-of-sample prediction
    text.append("## Out-of-sample Prediction:")
    text.append("| Metric | Result |")
    text.append("|---|---|")
    text.append("| Recall | " + str(results["out of sample"]["recall"]) + " |")
    text.append("| Precision | " + str(results["out of sample"]["precision"]) + " |")
    text.append("| F1 | " + str(results["out of sample"]["f1"]) + " |")

    # Specify path and file-name
    save_name = "classification_results"
    if name != "":
        save_name += "_" + name
    save_name = path + save_name + ".md"

    # Save MD-File
    with open(save_name, 'w') as f:
        for item in text:
            f.write("%s\n" % item)

