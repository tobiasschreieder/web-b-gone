from typing import List, Dict, Type, Tuple

from classification.preprocessing import Category, GroundTruth, Website
from evaluation import comparison, text_preprocessing
from extraction.extraction_models import BaseExtractionModel


def evaluate_extraction(model_cls_extraction: Type[BaseExtractionModel],
                        category: Category,
                        train_test_split: float,
                        max_size: int = -1,
                        split_type: str = "website", **model_kwargs) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a given extraction model
    :param model_cls_extraction: Extraction model which should be used
    :param category: Category which should be evaluated
    :param train_test_split: Specify proportion of train data
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param model_kwargs:
    :return: Dictionary with calculated metric scores
    """
    # Load and split data
    train_ids: List[str]
    test_ids: List[str]
    train_ids, test_ids = split_data(category=category, train_test_split=train_test_split, split_type=split_type,
                                     max_size=max_size)

    # Extraction
    model_extraction: BaseExtractionModel
    model_extraction = model_cls_extraction(category, **model_kwargs)
    model_extraction.train(train_ids)

    # out of sample prediction
    if len(test_ids) != 0:
        results_extraction_test = extraction_metrics(model_extraction.extract(web_ids=test_ids),
                                                [GroundTruth.load(web_id).attributes for web_id in test_ids])
    else:
        results_extraction_test = {"exact_match": None, "f1": None}

    # in sample prediction
    if len(train_ids) != 0:
        results_extraction_train = extraction_metrics(model_extraction.extract(web_ids=train_ids),
                                                     [GroundTruth.load(web_id).attributes for web_id in train_ids])
    else:
        results_extraction_train = {"exact_match": None, "f1": None}

    results = {"out of sample": results_extraction_test, "in sample": results_extraction_train}

    return results


def split_data(category: Category, train_test_split: float, split_type: str, max_size: int = -1,
               seed: str = "eval_class") -> Tuple[List[str], List[str]]:
    """
    Method to split dataset with defined split-type
    :param category: Category that should be considered
    :param train_test_split: Specify proportion of train data [0; 1]
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param seed: String with seed
    :return: Tuple with train-ids and test-ids
    """
    train_ids = list()
    test_ids = list()

    if split_type == "website":
        web_ids: List[str] = Website.get_website_ids(max_size=max_size, categories=category, rdm_sample=True, seed=seed)

        split_index = int(len(web_ids) * train_test_split)
        train_ids = web_ids[:split_index]
        test_ids = web_ids[split_index:]

    elif split_type == "domain":
        domains = Website.get_all_domains(category=category)
        split_index = int(len(domains) * train_test_split)
        train_domains = domains[:split_index]
        test_domains = domains[split_index:]

        train_ids += Website.get_website_ids(max_size=int(max_size * train_test_split), rdm_sample=True, seed=seed,
                                             categories=category, domains=train_domains)

        test_ids += Website.get_website_ids(max_size=int(max_size * (1 - train_test_split)), rdm_sample=True,
                                            seed=seed, categories=category, domains=test_domains)

    else:
        return [], []

    return train_ids, test_ids


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
