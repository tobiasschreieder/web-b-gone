from typing import List, Callable, Type

from classification.category_models import BaseCategoryModel, NeuralNetCategoryModel
from classification.preprocessing import Category, GroundTruth, Website


def evaluate_classification(model_cls: Type[BaseCategoryModel],
                            metrik: Callable[[List[Category], List[Category]], float],
                            train_test_split: float, **model_kwargs):
    web_ids: List[str] = Website.get_website_ids(rdm_sample=True, seed='eval_class')
    split_index = int(len(web_ids)*train_test_split)
    train_ids = web_ids[:split_index]
    test_ids = web_ids[split_index:]

    model: BaseCategoryModel

    if model_cls == NeuralNetCategoryModel:
        model: NeuralNetCategoryModel = model_cls(**model_kwargs)
        model.network.train(train_ids)
    else:
        model = model_cls(**model_kwargs)

    eval_val = metrik(model.classification(test_ids), [GroundTruth.load(web_id) for web_id in test_ids])


def sample_classification_metrik(results: List[Category], truth: List[Category]) -> float:
    return 0
