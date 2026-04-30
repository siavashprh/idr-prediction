import numpy as np


def pseudo_label(
    sequences: list[str],
    model,
    threshold: float = 0.5,
) -> list[list[int]]:
    """Run a predictor on decoded sequences and return binary disorder labels.

    Args:
        sequences: List of amino acid strings.
        model: Any object with predict_proba(X: list) -> list[float].
        threshold: P(disorder) cutoff for positive label.

    Returns:
        List of per-residue binary label lists, one per sequence.
    """
    labels = []
    for seq in sequences:
        proba = model.predict_proba([list(seq)])
        proba_arr = np.array(proba if isinstance(proba, list) else [proba])
        binary = (proba_arr >= threshold).astype(int).tolist()
        labels.append(binary)
    return labels
