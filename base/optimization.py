import abc
import random

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
    auc,
    roc_curve,
    precision_recall_curve,
)

from base.basemodel import BaseModel
from base.basemodel import ModelFactory
from base.data import Data, TrainTestSplit
from base.evaluation import Result


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train(self, model: BaseModel, data: Data, optimizer_config) -> Result:
        raise NotImplementedError


class Tester(abc.ABC):
    @abc.abstractmethod
    def test(self, model: BaseModel, data: Data) -> Result:
        raise NotImplementedError


def cross_validation(
    k: int,
    data_size: int,
    train_test_spliter: TrainTestSplit,
    model_factory: ModelFactory,
    trainer: Trainer,
    tester: Tester,
    optimization_config,
    device,
) -> Result:

    subsets = dict()
    subset_size = int(data_size / k)
    remain = set(range(0, data_size))
    for i in range(k - 1):
        subsets[i] = random.sample(remain, subset_size)
        remain = remain.difference(subsets[i])
    subsets[k - 1] = remain

    result = Result()
    for i in range(k):
        print(f"\nLogging Info - Fold {i + 1} >>>>>>>>>>>>>>\n")

        indices = set(range(0, data_size))
        test_indices = list(subsets[i])
        train_indices = list(indices.difference(subsets[i]))
        print("test_indices:", test_indices)
        print("train_indices:", train_indices)
        print()

        train_data, test_data = train_test_spliter.split(train_indices, test_indices)

        model = model_factory.make_model()
        trainer.train(
            model=model,
            data=train_data,
            optimizer_config=optimization_config,
            device=device,
        )
        test_result = tester.test(model=model, data=test_data)
        print(f"\nLogging Info - Fold {i + 1} Result :", test_result.get_result())
        model.destroy()

        result.add(test_result)

    result.divide(k=k)

    print(
        f"\nLogging Info - {k} fold result: avg_auc: {result.auc}, avg_acc: {result.acc}, avg_f1: {result.f1}, avg_aupr: {result.aupr}\n"
    )

    return result


def get_prediction_results(y_test, y_predict, threshold):
    result = Result()

    binary_predict = np.where(np.array(y_predict) >= threshold, 1, 0).tolist()

    result.acc = accuracy_score(y_test, binary_predict)
    result.f1 = f1_score(y_test, binary_predict, average="macro", zero_division=1)
    result.recall = recall_score(
        y_test, binary_predict, average="macro", zero_division=1
    )
    result.precision = precision_score(
        y_test, binary_predict, average="macro", zero_division=1
    )
    result.mcc = matthews_corrcoef(y_test, binary_predict)

    precision, recall, thresholds = precision_recall_curve(y_test, y_predict)
    numerator = 2 * recall * precision
    denominator = recall + precision
    f1_scores = np.divide(
        numerator, denominator, out=np.zeros_like(denominator), where=(denominator != 0)
    )
    result.max_f1 = np.max(f1_scores)

    fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
    result.auc = auc(fpr, tpr)
    result.tpr = tpr
    result.fpr = fpr

    return result
