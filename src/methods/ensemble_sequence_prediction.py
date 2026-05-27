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


def predict_proba_sequence(model, sequence, max_length: int = 400, alpha: float = 0.75):
    """Sliding-window inference returning per-residue P(disorder)."""
    if len(sequence) <= max_length:
        p = model.predict_proba([sequence])
        return np.array(p if isinstance(p, list) else [p])
    y = np.zeros(len(sequence))
    weight = np.zeros(len(sequence))
    start = 0
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        p = model.predict_proba([sequence[start:end]])
        chunk = np.array(p if isinstance(p, list) else [p])
        y[start:end] += chunk
        weight[start:end] += 1.0
        if end == len(sequence):
            break
        start += int(max_length * alpha)
    return y / np.maximum(weight, 1.0)
