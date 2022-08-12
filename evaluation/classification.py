from typing import List, Dict, Callable, Type
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from classification.category_models import BaseCategoryModel, NeuralNetCategoryModel
from classification.preprocessing import Category, GroundTruth, Website

AVERAGE = "macro"  # determines the type of averaging performed on the data, choose from "micro", "macro", "weighted"


def evaluate_model(model_cls: Type[BaseCategoryModel],
                   metrik: Callable[[List[Category], List[Category]], Dict[str, float]],
                   train_test_split: float,
                   max_size: int = -1, **model_kwargs):  # TODO Adjust method
    """
    Evaluate a given model
    :param model_cls:
    :param metrik:
    :param train_test_split:
    :param max_size:
    :param model_kwargs:
    :return:
    """
    web_ids: List[str] = Website.get_website_ids(max_size=max_size, rdm_sample=True, seed='eval_class')
    split_index = int(len(web_ids)*train_test_split)
    train_ids = web_ids[:split_index]
    test_ids = web_ids[split_index:]

    model: BaseCategoryModel

    if model_cls == NeuralNetCategoryModel:
        model: NeuralNetCategoryModel = model_cls(**model_kwargs)
        model.network.train(train_ids)
    else:
        model = model_cls(**model_kwargs)

    results_classification = metrik(model.classification(test_ids),
                                    [GroundTruth.load(web_id).category for web_id in test_ids])

    return results_classification


def format_test_data(data: List[Category]) -> List[str]:
    """
    Format test data to List of Strings (Category.name)
    :param data: given List of data
    :return: formatted data
    """
    formatted_data = list()
    for d in data:
        formatted_data.append(d.name)

    return formatted_data


def create_confusion_matrix(truth: List[str], pred: List[str]):  # TODO: specify return
    """
    Create Confusion-Matrix
    :param truth: List with names of predicted categories
    :param pred: List with names of ground truth categories
    :return: Confusion-Matrixx as Array
    """
    conf = confusion_matrix(y_true=truth, y_pred=pred)
    return conf


def classification_metrics(pred: List[Category], truth: List[Category]) -> Dict[str, float]:
    """
    Calculate Recall, Precision and F1 for classification model
    :param pred: List with predicted categories
    :param truth: List with ground truth categories
    :return: Results as Dict
    """
    pred = format_test_data(data=pred)
    truth = format_test_data(data=truth)

    recall = round(recall_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)
    precision = round(precision_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)
    f1 = round(f1_score(y_true=truth, y_pred=pred, average=AVERAGE), 4)

    results = {"recall": recall, "precision": precision, "f1": f1}

    return results


def extraction_metrics(pred: List[Category], truth: List[Category]) -> float:
    return 0.0
