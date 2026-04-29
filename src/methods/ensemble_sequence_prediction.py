import numpy as np
from base.basemodel import BaseModel


def predict_sequence(
    model: BaseModel, sequence, max_length=400, alpha=3 / 4
):

    if len(sequence) > max_length:
        y_predict = np.zeros(len(sequence))
        sub_pred = np.zeros(len(sequence))

        y_predict[:max_length] = predict_sequence(
            model, sequence[:max_length], max_length=max_length, alpha=alpha
        )
        cut_point = int(max_length * (1 - alpha))
        sub_pred[cut_point:] = predict_sequence(
            model, sequence[cut_point:], max_length=max_length, alpha=alpha
        )
        y_predict[cut_point:max_length] = (
            sub_pred[cut_point:max_length] + y_predict[cut_point:max_length]
        ) / 2

        y_predict[max_length:] = sub_pred[max_length:]

        return y_predict

    return np.array(model.predict([sequence]))
