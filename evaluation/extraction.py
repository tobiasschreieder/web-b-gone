from typing import List, Dict, Type

from classification.preprocessing import Category, GroundTruth, Website
from evaluation import comparison, text_preprocessing
from extraction.extraction_models import BaseExtractionModel


def evaluate_extraction(model_cls_extraction: Type[BaseExtractionModel],
                        category: Category,
                        train_test_split: float,
                        max_size: int = -1, **model_kwargs) -> Dict[str, float]:
    """
    Evaluate a given extraction model
    :param model_cls_extraction: Extraction model which should be used
    :param category: Category which should be evaluated
    :param train_test_split: Specify proportion of train data
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param model_kwargs:
    :return: Dictionary with calculated metric scores
    """
    # Load and split data
    web_ids: List[str] = Website.get_website_ids(categories=category, max_size=max_size, rdm_sample=True,
                                                 seed='eval_class')
    split_index = int(len(web_ids)*train_test_split)
    train_ids = web_ids[:split_index]
    test_ids = web_ids[split_index:]

    # Extraction
    model_extraction: BaseExtractionModel
    model_extraction = model_cls_extraction(category, **model_kwargs)
    model_extraction.train(train_ids)

    results_extraction = extraction_metrics(model_extraction.extract(web_ids=test_ids),
                                            [GroundTruth.load(web_id).attributes for web_id in test_ids])

    return results_extraction


def format_data_extraction(data: List[Dict[str, List[str]]]) -> List[List[str]]:
    """
    Format test data to List of Strings
    :param data: given datastructure
    :return: formatted data
    """
    formatted_data = list()
    for website in data:
        for attribute, text in website.items():
            if attribute != "category":
                formatted_data.append(text)

    return formatted_data


def extraction_metrics(pred: List[Dict[str, List[str]]], truth: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Calculate Exact_Match and F1 for extraction model
    :param pred: Predictions
    :param truth: Ground-truth
    :return: Results as Dict
    """
    pred = text_preprocessing.preprocess_extraction_data_comparison(data=pred)
    truth = text_preprocessing.preprocess_extraction_data_comparison(data=truth)

    pred = format_data_extraction(data=pred)
    truth = format_data_extraction(data=truth)

    exact_match = comparison.exact_match(truth=truth, pred=pred)
    f1 = comparison.f1(truth=truth, pred=pred)

    results = {"exact_match": exact_match, "f1": f1}

    return results
