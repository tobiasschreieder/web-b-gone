from pathlib import Path
from typing import List, Dict
import copy
import logging

from classification.preprocessing import GroundTruth, Website, Category

log = logging.getLogger('Extraction')


def exploratory_data_analysis(max_size: int = -1, name: str = "", path: Path = "working/", data=None, category=None):
    """
    Run exploratory data analysis
    :param category: Category of given data
    :param data: List with dataset that contains extraction results of a model
    :param path: String with path to save MD-File
    :param name: String with name of considered dataset
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    """
    path = Path(path)

    # Load dataset
    if not data:
        web_ids: List[str] = Website.get_website_ids(max_size=max_size, rdm_sample=True, seed='eval_class')
        ground_truth = [GroundTruth.load(web_id).attributes for web_id in web_ids]
    else:
        prediction = copy.deepcopy(data)
        ground_truth = append_category(data=prediction, category=category)

    # Create datastructures to save later calculated metrics
    dictionary = get_dictionary(ground_truth=ground_truth)
    count_category_ids = count_websites_per_category(ground_truth=ground_truth)

    # Calculate metrics
    length_dict = average_length_ground_truth(ground_truth=ground_truth, dictionary=copy.deepcopy(dictionary),
                                              count_category_ids=count_category_ids)
    token_dict = average_tokens_ground_truth(ground_truth=ground_truth, dictionary=copy.deepcopy(dictionary),
                                             count_category_ids=count_category_ids)
    solution_dict = average_solutions_ground_truth(ground_truth=ground_truth, dictionary=copy.deepcopy(dictionary),
                                                   count_category_ids=count_category_ids)
    missing_dict = average_missing_ground_truth(ground_truth=ground_truth, dictionary=copy.deepcopy(dictionary),
                                                count_category_ids=count_category_ids)

    # Save metrics as MD-File
    eda = [length_dict, token_dict, solution_dict, missing_dict]
    create_eda_md_table(eda=eda, name=name, path=path)


def append_category(data: List[Dict[str, List[str]]], category: Category) -> List[Dict[str, List[str]]]:
    """
    Append category to datastructure of model prediction
    :param data: List with datastructure of model prediction
    :param category: Category to append
    :return: expanded List
    """
    for i in range(0, len(data)):
        data[i].setdefault("category", str(category))

    return data


def get_dictionary(ground_truth: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Create Dictionary to save later calculated results
    :param ground_truth: List with Ground-truth elements
    :return: Dictionary with attributes and counters for each category
    """
    dictionary = dict()
    for gt in ground_truth:
        category = gt["category"]
        if category not in dictionary:
            dictionary.setdefault(category, dict())
        for attribute, value in gt.items():
            if attribute != "category" and attribute not in dictionary[category]:
                dictionary[category].setdefault(attribute, 0.0)

    return dictionary


def count_websites_per_category(ground_truth: List[Dict[str, List[str]]]) -> Dict[str, int]:
    """
    Method to count websites per category for given ground-truth
    :param ground_truth: List with Ground-truth elements
    :return: Dictionary with categories and number of websites
    """
    count_category_ids = dict()
    for gt in ground_truth:
        category = gt["category"]
        if category not in count_category_ids:
            count_category_ids.setdefault(category, 1)
        else:
            count_category_ids[category] += 1

    return count_category_ids


def average_length_ground_truth(ground_truth: List[Dict[str, List[str]]], dictionary: Dict[str, Dict[str, float]],
                                count_category_ids: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of chars for ground-truth string per category and attribute
    :param count_category_ids: Dictionary with number of websites per category
    :param dictionary: Datastructure that is used to save calculated results
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    length_dict = dictionary

    for gt in ground_truth:
        for attribute, value in gt.items():
            category = gt["category"]
            if attribute != "category":
                length_value = 0.0
                for v in value:
                    length_value += len(v)
                if len(value) != 0:
                    length_value = length_value / len(value)
                else:
                    length_value = 0
                length_dict[category][attribute] += length_value

    for category in length_dict:
        length = count_category_ids[category]
        for attribute in length_dict[category]:
            length_dict[category][attribute] = round(length_dict[category][attribute] / length, 2)

    return length_dict


def average_tokens_ground_truth(ground_truth: List[Dict[str, List[str]]], dictionary: Dict[str, Dict[str, float]],
                                count_category_ids: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of tokens (words) for ground-truth string per category and attribute
    :param count_category_ids: Dictionary with number of websites per category
    :param dictionary: Datastructure that is used to save calculated results
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    token_dict = dictionary

    for gt in ground_truth:
        for attribute, value in gt.items():
            category = gt["category"]
            if attribute != "category":
                count_tokens = 0.0
                for v in value:
                    count_tokens += len(v.split())
                if len(value) != 0:
                    count_tokens = count_tokens / len(value)
                else:
                    count_tokens = 0
                token_dict[category][attribute] += count_tokens

    for category in token_dict:
        length = count_category_ids[category]
        for attribute in token_dict[category]:
            token_dict[category][attribute] = round(token_dict[category][attribute] / length, 2)

    return token_dict


def average_solutions_ground_truth(ground_truth: List[Dict[str, List[str]]], dictionary: Dict[str, Dict[str, float]],
                                   count_category_ids: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of different solutions in ground-truth per category and attribute
    :param count_category_ids: Dictionary with number of websites per category
    :param dictionary: Datastructure that is used to save calculated results
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    solution_dict = dictionary

    for gt in ground_truth:
        for attribute, value in gt.items():
            category = gt["category"]
            if attribute != "category":
                solution_dict[category][attribute] += len(value)

    for category in solution_dict:
        length = count_category_ids[category]
        for attribute in solution_dict[category]:
            solution_dict[category][attribute] = round(solution_dict[category][attribute] / length, 2)

    return solution_dict


def average_missing_ground_truth(ground_truth: List[Dict[str, List[str]]], dictionary: Dict[str, Dict[str, float]],
                                 count_category_ids: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of missing ground-truth per category and attribute
    :param count_category_ids: Dictionary with number of websites per category
    :param dictionary: Datastructure that is used to save calculated results
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    missing_dict = dictionary

    for gt in ground_truth:
        for attribute, value in gt.items():
            category = gt["category"]
            if attribute != "category":
                missing_value = False
                if len(value) == 0:
                    missing_value = True
                else:
                    for v in value:
                        if v == "":
                            missing_value = True

                if missing_value:
                    missing_dict[category][attribute] += 1

    for category in missing_dict:
        length = count_category_ids[category]
        for attribute in missing_dict[category]:
            missing_dict[category][attribute] = round(missing_dict[category][attribute] / length, 2)

    return missing_dict


def create_eda_md_table(eda: List[Dict[str, Dict[str, float]]], name: str, path: Path):
    """
    Create Markdown File with Table of EDA in /working
    :param path: String with path to save MD-File
    :param name: String with name of considered dataset
    :param eda: List with all calculated exploratory data analysis results
    """
    path = Path(path)

    # Header
    text = list()
    text.append("# Exploratory Data Analysis")

    # Description of measures
    text.append("**Average Length** = Average number of characters / length of solution calculated for each attribute "
                "and category.\n")
    text.append("**Average Tokens** = Average number of tokens (words) calculated for each attribute and category.\n")
    text.append("**Average Solutions** = Average number of existing solutions calculated for each attribute and "
                "category.\n")
    text.append("**Average Missing** = Average proportion of missing solutions calculated for each attribute and "
                "category.\n")
    text.append("\n")

    # Create MD-Tables for each category
    for category in eda[0]:
        text.append("## " + "Category: " + category.upper() + "\n")
        text.append("| Attribute | Average Length | Average Tokens | Average Solutions | Average Missing |")
        text.append("|---|---|---|---|---|")

        for attribute in eda[0][category]:
            column = "| " + str(attribute) + " | "
            column += str(eda[0][category][attribute]) + " | "
            column += str(eda[1][category][attribute]) + " | "
            column += str(eda[2][category][attribute]) + " | "
            column += str(eda[3][category][attribute]) + " | "
            text.append(column)

        text.append("\n")

    # Specify path and file-name
    save_name = "eda"
    if name != "":
        save_name += "_" + name
    save_name = path.joinpath(save_name + ".md")

    # Save MD-File
    try:
        with open(save_name, 'w') as f:
            for item in text:
                f.write("%s\n" % item)
        log.info(f'eda saved to {path}')
    except FileNotFoundError:
        with open("working/eda_results.md", 'w') as f:
            for item in text:
                f.write("%s\n" % item)
        log.info("FileNotFoundError: eda_results.md saved at /working")
