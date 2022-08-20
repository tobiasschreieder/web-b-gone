from typing import List, Dict

from classification.preprocessing import GroundTruth, Website


def exploratory_data_analysis(max_size: int = -1):
    """
    Run exploratory data analysis
    :param max_size: Size of sample which should be used, -1 -> all data will be used
    """
    web_ids: List[str] = Website.get_website_ids(max_size=max_size, rdm_sample=True, seed='eval_class')
    ground_truth = [GroundTruth.load(web_id).attributes for web_id in web_ids]

    length_dict = average_length_ground_truth(ground_truth=ground_truth)
    token_dict = average_tokens_ground_truth(ground_truth=ground_truth)
    solution_dict = average_solutions_ground_truth(ground_truth=ground_truth)
    missing_dict = average_missing_ground_truth(ground_truth=ground_truth)

    eda = [length_dict, token_dict, solution_dict, missing_dict]

    create_eda_md_table(eda=eda)

    # ToDo: further processing of the results deepcopy


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


def average_length_ground_truth(ground_truth: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of chars for ground-truth string per category and attribute
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    length_dict = get_dictionary(ground_truth=ground_truth)

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

    count_category_ids = count_websites_per_category(ground_truth=ground_truth)

    for category in length_dict:
        length = count_category_ids[category]
        for attribute in length_dict[category]:
            length_dict[category][attribute] = round(length_dict[category][attribute] / length, 2)

    return length_dict


def average_tokens_ground_truth(ground_truth: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of tokens (words) for ground-truth string per category and attribute
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    token_dict = get_dictionary(ground_truth=ground_truth)

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

    count_category_ids = count_websites_per_category(ground_truth=ground_truth)

    for category in token_dict:
        length = count_category_ids[category]
        for attribute in token_dict[category]:
            token_dict[category][attribute] = round(token_dict[category][attribute] / length, 2)

    return token_dict


def average_solutions_ground_truth(ground_truth: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of different solutions in ground-truth per category and attribute
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    solution_dict = get_dictionary(ground_truth=ground_truth)

    for gt in ground_truth:
        for attribute, value in gt.items():
            category = gt["category"]
            if attribute != "category":
                solution_dict[category][attribute] += len(value)

    count_category_ids = count_websites_per_category(ground_truth=ground_truth)

    for category in solution_dict:
        length = count_category_ids[category]
        for attribute in solution_dict[category]:
            solution_dict[category][attribute] = round(solution_dict[category][attribute] / length, 2)

    return solution_dict


def average_missing_ground_truth(ground_truth: List[Dict[str, List[str]]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average number of missing ground-truth per category and attribute
    :param ground_truth: List with Ground-truth elements
    :return: Results saved in dictionary structure
    """
    ground_truth = ground_truth
    missing_dict = get_dictionary(ground_truth=ground_truth)

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

    count_category_ids = count_websites_per_category(ground_truth=ground_truth)

    for category in missing_dict:
        length = count_category_ids[category]
        for attribute in missing_dict[category]:
            missing_dict[category][attribute] = round(missing_dict[category][attribute] / length, 2)

    return missing_dict


def create_eda_md_table(eda: List[Dict[str, Dict[str, float]]]):
    """
    Create Markdown File with Table of EDA in /working
    :param eda: List with all calculated exploratory data analysis results
    """
    text = list()
    text.append("# Exploratory Data Analysis")
    text.append("\n")

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

    with open('working/exploratory_data_analysis.md', 'w') as f:
        for item in text:
            f.write("%s\n" % item)

