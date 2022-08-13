from typing import List
from collections import Counter


def exact_match(truth: List[List[str]], pred: List[List[str]]) -> float:
    """
    Method to calculate Exact-Match Score
    :param truth: Ground-truth
    :param pred: Predictions
    :return: Exact-Match Score
    """
    total = 0

    for i in range(0, len(truth)):
        predicted_answer = pred[i]
        if predicted_answer is not None and exact_match_single(truth=truth, predicted_answer=predicted_answer,
                                                               iterator=i):
            total += 1

    return round(total / len(truth), 4)


def exact_match_single(truth: List[List[str]], predicted_answer: List[str], iterator: int) -> bool:
    """
    Check if there is an Exact-Match for a given pair of predicted answer and ground-truth answers
    :param truth: Ground-truth
    :param predicted_answer: List with predicted_answers -> just first answer will be used
    :param iterator: Integer which represents current iterator
    :return: Boolean, True if Exact-Match and False if no Exact-Match
    """
    if len(predicted_answer) != 0:
        predicted_answer = predicted_answer[0]
    else:
        predicted_answer = ""

    for answer in truth[iterator]:
        if answer == predicted_answer:
            return True

    return False


def f1(truth: List[List[str]], pred: List[List[str]]) -> float:
    """
    Method to calculate F1-Score
    :param truth: Ground-truth
    :param pred: Predictions
    :return: F1-Score
    """
    total = 0

    for i in range(0, len(truth)):
        predicted_answer = pred[i]
        if predicted_answer is not None:
            total += f1_single(truth=truth, predicted_answer=predicted_answer, iterator=i)

    return round(total / len(truth), 4)


def f1_single(truth: List[List[str]], predicted_answer: List[str], iterator: int) -> bool:
    """
    Calculate F1-Score for a given pair of predicted answer and ground-truth answers
    :param truth: Ground-truth
    :param predicted_answer: List with predicted_answers -> just first answer will be used
    :param iterator: Integer which represents current iterator
    :return: F1-Score
    """
    f1 = 0

    if len(predicted_answer) != 0:
        predicted_answer = predicted_answer[0]
    else:
        predicted_answer = ""

    predicted_answer_tokens = Counter(predicted_answer.split())
    num_predicted_answer_tokens = sum(predicted_answer_tokens.values())

    for answer in truth[iterator]:
        answer_tokens = Counter(answer.split())
        num_answer_tokens = sum(answer_tokens.values())
        num_same = sum((predicted_answer_tokens & answer_tokens).values())

        if num_same == 0:
            continue

        precision = 1.0 * num_same / num_predicted_answer_tokens
        recall = 1.0 * num_same / num_answer_tokens
        f1 = max(2 * precision * recall / (precision + recall), f1)

    return f1
