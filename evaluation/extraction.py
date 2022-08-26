import copy
from pathlib import Path
from typing import List, Dict, Type, Tuple
import logging

from classification.preprocessing import Category, GroundTruth, Website
from evaluation import comparison, text_preprocessing, dataset
from extraction.extraction_models import BaseExtractionModel
from extraction.extraction_models.structured_ner_model import CombinedExtractionModel

log = logging.getLogger('Extraction')

SEED = "eval_class"  # Seed to Load data sample


def evaluate_extraction(model_cls_extraction: Type[BaseExtractionModel],
                        category: Category,
                        train_test_split: float,
                        max_size: int = -1,
                        split_type: str = "website",
                        save_results: bool = True, **model_kwargs) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a given extraction model
    :param model_cls_extraction: Extraction model which should be used
    :param category: Category which should be evaluated
    :param train_test_split: Specify proportion of train data
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param save_results: Boolean -> Set to False if no MD-File should be created
    :param model_kwargs:
    :return: Dictionary with calculated metric scores
    """
    # Load and split data
    train_ids: List[str]
    test_ids: List[str]
    train_ids, test_ids = split_data(category=category, train_test_split=train_test_split, split_type=split_type,
                                     max_size=max_size)

    model_extraction: BaseExtractionModel
    model_extraction = model_cls_extraction(category, **model_kwargs)
    # if model_cls_extraction != CombinedExtractionModel:
    #     model_extraction.train(train_ids)

    # Out of sample prediction
    prediction_oos = list()
    if len(test_ids) != 0:
        prediction_oos = model_extraction.extract(web_ids=test_ids)
        results_extraction_test = extraction_metrics(prediction_oos,
                                                     [GroundTruth.load(web_id).attributes for web_id in test_ids])
    else:
        results_extraction_test = {"exact_match_top_1": None, "exact_match_top_3": None,
                                   "f1_top_1": None, "f1_top_3": None}

    # In sample prediction
    if len(train_ids) != 0:
        prediction_is = model_extraction.extract(web_ids=train_ids)
        results_extraction_train = extraction_metrics(prediction_is,
                                                      [GroundTruth.load(web_id).attributes for web_id in train_ids])
    else:
        results_extraction_train = {"exact_match_top_1": None, "exact_match_top_3": None,
                                    "f1_top_1": None, "f1_top_3": None}

    # Combine results
    results = {"out of sample": results_extraction_test, "in sample": results_extraction_train}

    # Save results
    if save_results:
        # Save results as MD-File
        parameters = {"Model": model_cls_extraction, "Category": category, "Data-split": split_type,
                      "Size dataset": max_size, "Train-Test-Split": train_test_split, "Seed": SEED}
        for k, v in model_kwargs.items():
            parameters.setdefault(str(k).capitalize(), str(v))

        path = model_extraction.dir_path
        create_md_file(results=results, parameters=parameters, path=path)

        # EDA of prediction
        if len(prediction_oos) != 0:
            dataset.exploratory_data_analysis(name="prediction", path=path, data=prediction_oos, category=category)

        # Wrong Predictions
        preprocessed_prediction = text_preprocessing.preprocess_extraction_data_comparison(
            data=copy.deepcopy(prediction_oos))
        preprocessed_ground_truth = text_preprocessing.preprocess_extraction_data_comparison(
            data=[GroundTruth.load(web_id).attributes for web_id in test_ids])

        save_wrong_results(pred=preprocessed_prediction, truth=preprocessed_ground_truth, path=path)

    return results


def split_data(category: Category, train_test_split: float, split_type: str,
               max_size: int = -1) -> Tuple[List[str], List[str]]:
    """
    Method to split dataset with defined split-type
    :param category: Category that should be considered
    :param train_test_split: Specify proportion of train data [0; 1]
    :param split_type: String to define Split-Type, Choose between "website" and "domain"
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    :return: Tuple with train-ids and test-ids
    """
    train_ids = list()
    test_ids = list()

    if split_type == "website":
        web_ids: List[str] = Website.get_website_ids(max_size=max_size, categories=category, rdm_sample=True, seed=SEED)

        split_index = int(len(web_ids) * train_test_split)
        train_ids = web_ids[:split_index]
        test_ids = web_ids[split_index:]

    elif split_type == "domain":
        domains = Website.get_all_domains(category=category)
        split_index = int(len(domains) * train_test_split)
        train_domains = domains[:split_index]
        test_domains = domains[split_index:]

        train_ids += Website.get_website_ids(max_size=int(max_size * train_test_split), rdm_sample=True, seed=SEED,
                                             categories=category, domains=train_domains)

        test_ids += Website.get_website_ids(max_size=int(max_size * (1 - train_test_split)), rdm_sample=True,
                                            seed=SEED, categories=category, domains=test_domains)

    else:
        return [], []

    return train_ids, test_ids


def format_data_extraction(data: List[Dict[str, List[str]]], attribute_name: str = "all") -> List[List[str]]:
    """
    Format test data to List of Strings
    :param attribute_name: String with name of attribute that should only be considered -> "all": all attributes used
    :param data: given datastructure
    :return: formatted data
    """
    formatted_data = list()
    if attribute_name == "all":
        for website in data:
            for attribute, text in website.items():
                if attribute != "category":
                    formatted_data.append(text)

    else:
        for website in data:
            for attribute, text in website.items():
                if attribute != "category" and attribute == attribute_name:
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

    # Overall Metrics
    pred_overall = format_data_extraction(data=pred)
    truth_overall = format_data_extraction(data=truth)

    exact_match_top_1 = comparison.exact_match(truth=truth_overall, pred=pred_overall, top_k=1)
    f1_top_1 = comparison.f1(truth=truth_overall, pred=pred_overall, top_k=1)

    exact_match_top_3 = comparison.exact_match(truth=truth_overall, pred=pred_overall, top_k=3)
    f1_top_3 = comparison.f1(truth=truth_overall, pred=pred_overall, top_k=3)

    results_overall = {"exact_match_top_1": exact_match_top_1, "exact_match_top_3": exact_match_top_3,
                       "f1_top_1": f1_top_1, "f1_top_3": f1_top_3}

    # Metrics per attribute
    results_attribute = dict()
    for attribute in truth[0]:
        if attribute != "category":
            pred_attribute = format_data_extraction(data=pred, attribute_name=attribute)
            truth_attribute = format_data_extraction(data=truth, attribute_name=attribute)

            exact_match_top_1_attribute = comparison.exact_match(truth=truth_attribute, pred=pred_attribute, top_k=1)
            f1_top_1_attribute = comparison.f1(truth=truth_attribute, pred=pred_attribute, top_k=1)

            exact_match_top_3_attribute = comparison.exact_match(truth=truth_attribute, pred=pred_attribute, top_k=3)
            f1_top_3_attribute = comparison.f1(truth=truth_attribute, pred=pred_attribute, top_k=3)

            results_attribute.setdefault(attribute, {"exact_match_top_1": exact_match_top_1_attribute,
                                                     "exact_match_top_3": exact_match_top_3_attribute,
                                                     "f1_top_1": f1_top_1_attribute, "f1_top_3": f1_top_3_attribute})

    results = {"overall": results_overall, "attribute": results_attribute}

    return results


def create_md_file(results: Dict[str, Dict[str, float]], parameters: Dict[str, str], name: str = "",
                   path: Path = Path("working/")):
    """
    Create MD-File for extraction results
    :param results: Dictionary with calculated results from extraction model
    :param parameters: Dictionary with all parameter that should be listed in file
    :param name: String with name of considered model
    :param path: Path with path to save MD-File
    """
    path = Path(path)

    # Header
    text = list()
    text.append("# Evaluation Extraction")
    if len(parameters) != 0:
        text.append("## Parameters:")
        for k, v in parameters.items():
            text.append("* " + str(k) + ": " + str(v))

    # Overall prediction
    text.append("## Overall Prediction: ")

    # In-sample prediction
    text.append("### In-sample Prediction:")
    text.append("| Metric | Top 1 | Top 3 |")
    text.append("|---|---|---|")
    text.append("| Exact Match | " + str(results["in sample"]["overall"]["exact_match_top_1"]) + " | " +
                str(results["in sample"]["overall"]["exact_match_top_3"]) + " |")
    text.append("| F1 | " + str(results["in sample"]["overall"]["f1_top_1"]) + " | " +
                str(results["in sample"]["overall"]["f1_top_3"]) + " |")

    # Out-of-sample prediction
    text.append("### Out-of-sample Prediction:")
    text.append("| Metric | Top 1 | Top 3 |")
    text.append("|---|---|---|")
    text.append("| Exact Match | " + str(results["out of sample"]["overall"]["exact_match_top_1"]) + " | " +
                str(results["out of sample"]["overall"]["exact_match_top_3"]) + " |")
    text.append("| F1 | " + str(results["out of sample"]["overall"]["f1_top_1"]) + " | " +
                str(results["out of sample"]["overall"]["f1_top_3"]) + " |")

    # Attribute prediction
    for attribute in results["in sample"]["attribute"]:
        text.append("## Attribute Prediction: " + attribute.capitalize())

        # In-sample prediction
        text.append("### In-sample Prediction:")
        text.append("| Metric | Top 1 | Top 3 |")
        text.append("|---|---|---|")
        text.append(
            "| Exact Match | " + str(results["in sample"]["attribute"][attribute]["exact_match_top_1"]) + " | " +
            str(results["in sample"]["attribute"][attribute]["exact_match_top_3"]) + " |")
        text.append("| F1 | " + str(results["in sample"]["attribute"][attribute]["f1_top_1"]) + " | " +
                    str(results["in sample"]["attribute"][attribute]["f1_top_3"]) + " |")

        # Out-of-sample prediction
        text.append("### Out-of-sample Prediction:")
        text.append("| Metric | Top 1 | Top 3 |")
        text.append("|---|---|---|")
        text.append(
            "| Exact Match | " + str(results["out of sample"]["attribute"][attribute]["exact_match_top_1"]) + " | " +
            str(results["out of sample"]["attribute"][attribute]["exact_match_top_3"]) + " |")
        text.append("| F1 | " + str(results["out of sample"]["attribute"][attribute]["f1_top_1"]) + " | " +
                    str(results["out of sample"]["attribute"][attribute]["f1_top_3"]) + " |")

    # Specify path and file-name
    save_name = "extraction_results"
    if name != "":
        save_name += "_" + name
    save_name = path.joinpath(save_name + ".md")

    # Save MD-File
    try:
        with open(save_name, 'w') as f:
            for item in text:
                f.write("%s\n" % item)
        log.info(f'results saved to {path}')
    except FileNotFoundError:
        with open("working/extraction_results.md", 'w') as f:
            for item in text:
                f.write("%s\n" % item)
        log.info("FileNotFoundError: extraction_results.md saved at /working")


def save_wrong_results(pred: List[Dict[str, List[str]]], truth: List[Dict[str, List[str]]],
                       name: str = "", path: Path = Path("working/")):
    """
    Save wrong extractions as MD-File
    :param pred: Predictions
    :param truth: Ground-truth
    :param name: String with name of considered model
    :param path: Path with path to save MD-File
    """
    wrong_result_list = list()
    wrong_result_list.append("# Wrong Extractions")
    wrong_result_list.append("| Attribute | Prediction | Ground Truth |")
    wrong_result_list.append("|---|---|---|")

    for i in range(0, len(truth)):
        for attribute, text in truth[i].items():
            if attribute != "category":
                if len(text) > 0 and len(pred[i][attribute]) > 0:
                    # print(attribute, text[0], pred[i][attribute][0])
                    if text[0] != pred[i][attribute][0]:
                        wrong_result_list.append(" | " + str(attribute) + " | " + str(pred[i][attribute][0]) + " | " +
                                                 str(text[0]) + " | ")

    # Specify path and file-name
    save_name = "extraction_mistakes"
    if name != "":
        save_name += "_" + name
    save_name = path.joinpath(save_name + ".md")

    # Save MD-File
    try:
        with open(save_name, 'w') as f:
            for item in wrong_result_list:
                f.write("%s\n" % item)
        log.info(f'results saved to {path}')
    except FileNotFoundError:
        with open("working/extraction_mistakes.md", 'w') as f:
            for item in wrong_result_list:
                f.write("%s\n" % item)
        log.info("FileNotFoundError: extraction_mistakes.md saved at /working")
