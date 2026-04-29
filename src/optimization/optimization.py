import os.path
import random

import numpy as np
import torch
from torch import nn

from base.data import Data
from base.evaluation import Result
from base.optimization import Trainer, Tester
from base.optimization import get_prediction_results
from src.config import OptimizerConfig, MODEL_SAVED_DIR
from src.models.sequence_models import TransformerLSTMModel


def batch_optimize(
    batch_size, data, loss_function, model, optimizer, optimizer_config,
):
    for epoch in range(optimizer_config.n_epoch):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        size = len(data.X)
        model.model.train()

        shuffle = [i for i in range(size)]
        random.shuffle(shuffle)

        temp = 0
        round_size = size - (size % batch_size) - batch_size
        for j in range(0, round_size, batch_size):
            indices = shuffle[j : j + batch_size]

            X = []
            for i in indices:
                X.append(data.X[i])
            max_length = max([len(s) for s in X])

            y = (
                torch.full((batch_size, max_length), 2)
                .long()
                .to(optimizer_config.device)
            )
            for k, i in enumerate(indices):
                region = data.y[i]
                y[k, : len(region)] = torch.tensor(region, dtype=torch.long).reshape(-1)

            loss, temp = predict_error(X, loss_function, model, temp, y)
            backpropagation(loss, optimizer)

            if j % 256 == 0 and j > 0:
                loss, current = temp / (256 / batch_size), j
                temp = 0
                print(f"loss: {loss:>7f}  [{current:>5d}/{round_size:>5d}]")

        if optimizer_config.save:
            if epoch % 5 == 0 or epoch == optimizer_config.n_epoch - 1:
                m = os.path.join(MODEL_SAVED_DIR, model.model_config.model_name + ".pth")
                torch.save(model.model.state_dict(), m)


def predict_error(X, loss_function, model, temp, y):
    pred = model.model(X)
    loss = loss_function(pred.reshape(-1, 2), y.reshape(-1))
    temp += loss.item()
    return loss, temp


def backpropagation(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class IDRTrainer(Trainer):
    def train(
        self,
        model: TransformerLSTMModel,
        data: Data,
        optimizer_config: OptimizerConfig,
        threshold=0.5,
    ) -> Result:
        loss_function = nn.CrossEntropyLoss(
            ignore_index=2,
            weight=torch.tensor(
                [
                    optimizer_config.first_class_weight,
                    1 - optimizer_config.first_class_weight,
                ]
            ).to(optimizer_config.device),
        )
        optimizer = torch.optim.Adam(model.model.parameters(), lr=optimizer_config.lr)
        batch_size = optimizer_config.batch_size

        batch_optimize(
            batch_size, data, loss_function, model, optimizer, optimizer_config,
        )

        model.model.eval()
        predictions = [model.predict(sequence) for sequence in data.X]

        total_labels = []
        total_predictions = []
        for k in range(len(data.y)):
            labels = data.y[k]
            predicted = predictions[k]
            total_labels += labels
            total_predictions += predicted

        return get_prediction_results(total_labels, total_predictions, threshold)


class IDRTester(Tester):
    def test(self, model: TransformerLSTMModel, data: Data, threshold=0.5) -> Result:

        model.model.eval()
        predictions = [model.predict(sequence) for sequence in data.X]
        binary_predictions = [
            np.where(np.array(pred) >= threshold, 1, 0).tolist() for pred in predictions
        ]

        total_labels = []
        total_predictions = []
        total_binary_predictions = []
        sequence_results = []
        for k in range(len(data.y)):
            labels = data.y[k]
            predicted = predictions[k]
            binary_predicted = binary_predictions[k]

            total_labels += labels
            total_predictions += predicted
            total_binary_predictions += binary_predicted

            result = get_prediction_results(
                y_test=labels, y_predict=binary_predicted, threshold=threshold
            )
            sequence_results.append(result)

        result = get_prediction_results(total_labels, total_predictions, threshold)
        result.sub_results = sequence_results

        return result
