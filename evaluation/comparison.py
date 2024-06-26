from collections import Counter
from typing import List

from evaluation import text_preprocessing


def exact_match(truth: List[List[str]], pred: List[List[str]], top_k: int = 1) -> float:
    """
    Method to calculate Top-k Exact-Match Score
    :param truth: Ground-truth
    :param pred: Predictions
    :param top_k: Integer >= 1 to define how many solutions should be considered
    :return: Exact-Match Score
    """
    total = 0

    for i in range(0, len(truth)):
        predicted_answer = pred[i]
        if predicted_answer is not None:
            for j in range(0, top_k):
                if len(predicted_answer) > j:
                    if exact_match_single(truth=truth, predicted_answer=predicted_answer, iterator=i, k=j):
                        total += 1
                        break

    return round(total / len(truth), 4)


def exact_match_single(truth: List[List[str]], predicted_answer: List[str], iterator: int, k: int = 0) -> bool:
    """
    Check if there is an Exact-Match for a given pair of predicted answer and ground-truth answers
    :param truth: Ground-truth
    :param predicted_answer: List with predicted_answers -> just first answer will be used
    :param iterator: Integer which represents current iterator
    :param k: Integer to select prediction
    :return: Boolean, True if Exact-Match and False if no Exact-Match
    """
    if len(predicted_answer) != 0:
        predicted_answer = predicted_answer[k]
    else:
        predicted_answer = ""

    for answer in truth[iterator]:
        if answer == predicted_answer:
            return True

    return False


def f1(truth: List[List[str]], pred: List[List[str]], top_k: int = 1) -> float:
    """
    Method to calculate Top-k F1-Score
    :param truth: Ground-truth
    :param pred: Predictions
    :param top_k: Integer >= 1 to define how many solutions should be considered
    :return: F1-Score
    """
    total = 0

    for i in range(0, len(truth)):
        predicted_answer = pred[i]
        if predicted_answer is not None:
            top_k_results = [0]
            for j in range(0, top_k):
                if len(predicted_answer) > j:
                    top_k_results.append(f1_single(truth=truth, predicted_answer=predicted_answer, iterator=i, k=j))
            total += max(top_k_results)

    return round(total / len(truth), 4)


def f1_single(truth: List[List[str]], predicted_answer: List[str], iterator: int, k: int = 0) -> bool:
    """
    Calculate F1-Score for a given pair of predicted answer and ground-truth answers
    :param truth: Ground-truth
    :param predicted_answer: List with predicted_answers -> just first answer will be used
    :param iterator: Integer which represents current iterator
    :param k: Integer to select prediction
    :return: F1-Score
    """
    f1_score = 0

    if len(predicted_answer) != 0:
        predicted_answer = predicted_answer[k]
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
        f1_score = max(2 * precision * recall / (precision + recall), f1_score)

    return f1_score


def partial_match(truth: List[List[str]], pred: List[List[str]], top_k: int = 1) -> float:
    """
    Method to calculate Top-k Partial-Match Score
    :param truth: Ground-truth
    :param pred: Predictions
    :param top_k: Integer >= 1 to define how many solutions should be considered
    :return: Exact-Match Score
    """
    total = 0

    for i in range(0, len(truth)):
        predicted_answer = pred[i]
        if predicted_answer is not None:
            for j in range(0, top_k):
                if len(predicted_answer) > j:
                    if partial_match_single(truth=truth, predicted_answer=predicted_answer, iterator=i, k=j):
                        total += 1
                        break

    return round(total / len(truth), 4)


def partial_match_single(truth: List[List[str]], predicted_answer: List[str], iterator: int, k: int = 0) -> bool:
    """
    Check if there is an Partial-Match for a given pair of predicted answer and ground-truth answers
    :param truth: Ground-truth
    :param predicted_answer: List with predicted_answers -> just first answer will be used
    :param iterator: Integer which represents current iterator
    :param k: Integer to select prediction
    :return: Boolean, True if Exact-Match and False if no Exact-Match
    """
    if len(predicted_answer) != 0:
        predicted_answer = text_preprocessing.preprocess_text_partial_match(text=predicted_answer[k])
    else:
        predicted_answer = ""

    for answer in truth[iterator]:
        truth_answer = text_preprocessing.preprocess_text_partial_match(text=answer)
        lengths = [len(truth_answer), len(predicted_answer)]
        if min(lengths) > 0:
            if min(lengths) / max(lengths) > 0.5:
                if truth_answer in predicted_answer or predicted_answer in truth_answer:
                    return True

    return False
