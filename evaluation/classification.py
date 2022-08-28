from pathlib import Path
from typing import List, Dict, Type, Tuple
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, ConfusionMatrixDisplay

from classification.category_models import BaseCategoryModel, NeuralNetCategoryModel
from classification.preprocessing import Category, GroundTruth, Website

log = logging.getLogger('Classification')

AVERAGE = "macro"  # determines the type of averaging performed on the data, choose from "micro", "macro", "weighted"
SEED = "eval_class"  # Seed to Load data sample


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
    train_ids, test_ids = split_data(train_test_split=train_test_split, split_type=split_type, max_size=max_size, )

    # Classification
    model_classification: BaseCategoryModel

    if model_cls_classification == NeuralNetCategoryModel:
        model_classification: NeuralNetCategoryModel = model_cls_classification(**model_kwargs)
        model_classification.network.train(web_ids=train_ids)
    else:
        model_classification = model_cls_classification(**model_kwargs)

    path = model_classification.dir_path

    # Out of sample prediction
    if len(test_ids) != 0:
        results_classification_test = classification_metrics(model_classification.classification(web_ids=test_ids),
                                                             [GroundTruth.load(web_id).category for web_id in test_ids],
                                                             path=path, create_conf=True)
    else:
        results_classification_test = {"recall": None, "precision": None, "f1": None}

    # In sample prediction
    if len(train_ids) != 0:
        results_classification_train = classification_metrics(model_classification.classification(web_ids=train_ids),
                                                              [GroundTruth.load(web_id).category for web_id in
                                                               train_ids], path=path)
    else:
        results_classification_train = {"recall": None, "precision": None, "f1": None}

    # Combine results
    results = {"out of sample": results_classification_test, "in sample": results_classification_train}

    # Save results as MD-File
    if save_results:
        parameters = {"Model": model_cls_classification, "Data-split": split_type,
                      "Size dataset": max_size, "Train-Test-Split": train_test_split, "Averaging method": AVERAGE}

        create_md_file(results=results, parameters=parameters, path=path)

    return results


def split_data(train_test_split: float, split_type: str, max_size: int = -1) -> Tuple[List[str], List[str]]:
    """
    Method to split dataset with defined split-type
    :param train_test_split: Specify proportion of train data [0; 1]
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :return: Tuple with train-ids and test-ids
    """
    train_ids = list()
    test_ids = list()

    if split_type == "website":
        web_ids: List[str] = Website.get_website_ids(max_size=max_size, rdm_sample=True, seed=SEED)

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

            train_ids += Website.get_website_ids(max_size=int(cat_size*train_test_split), rdm_sample=True, seed=SEED,
                                                 categories=cat, domains=train_domains)

            test_ids += Website.get_website_ids(max_size=int(cat_size*(1-train_test_split)), rdm_sample=True, seed=SEED,
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


def create_confusion_matrix(pred: List[str], truth: List[str], path: Path):
    """
    Create and save Confusion-Matrix
    :param truth: List with names of predicted categories
    :param pred: List with names of ground truth categories
    :param path: Path to save MD-File
    """
    ConfusionMatrixDisplay.from_predictions(y_pred=pred, y_true=truth, xticks_rotation=45, normalize="true",
                                            values_format=".1g")

    name = path.joinpath("conf_matrix.png")
    plt.subplots_adjust(top=0.95, bottom=0.25, left=0, right=1.0)
    plt.savefig(fname=name, format="png")

    try:
        plt.savefig(fname=name, format="png")
        log.info(f'conf_matrix saved to {path}')
    except FileNotFoundError:
        plt.savefig(fname="working/conf_matrix.png", format="png")
        log.info("FileNotFoundError: conf_matrix.png saved at /working")


def classification_metrics(pred: List[Category], truth: List[Category], path: Path, create_conf: bool = False, ) \
        -> Tuple[Dict[str, float], np.array]:
    """
    Calculate Recall, Precision and F1 for classification model
    :param pred: List with predicted categories
    :param truth: List with ground-truth categories
    :param path: Path to save conf_matrix
    :param create_conf: Boolean to select if confusion matrix should be saved
    :return: Results as Dict
    """
    pred = format_data_classification(data=pred)
    truth = format_data_classification(data=truth)

    recall = round(recall_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)
    precision = round(precision_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)
    f1 = round(f1_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)

    results = {"recall": recall, "precision": precision, "f1": f1}

    if create_conf:
        create_confusion_matrix(pred=pred, truth=truth, path=path)

    return results


def create_md_file(results: Dict[str, Dict[str, float]], parameters: Dict[str, str], name: str = "",
                   path: Path = Path("working/")):
    """
    Create MD-File for extraction results
    :param results: Dictionary with calculated results from extraction model
    :param parameters: Dictionary with all parameter that should be listed in file
    :param name: String with name of considered model
    :param path: Path to save MD-File
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
    save_name = path.joinpath(save_name + ".md")

    # Save MD-File
    try:
        with open(save_name, 'w', encoding='utf-8') as f:
            for item in text:
                f.write("%s\n" % item)
        log.info(f'results saved to {path}')
    except FileNotFoundError:
        with open("working/classification_results.md", 'w') as f:
            for item in text:
                f.write("%s\n" % item)
        log.info("FileNotFoundError: classification_results.md saved at /working")