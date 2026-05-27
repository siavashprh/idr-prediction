"""Shared helpers for train_baseline.py, train_gan.py, train_augmented.py."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ModelConfig
from src.methods.ensemble_sequence_prediction import predict_proba_sequence
from src.models.sequence_models import TransformerLSTMModel


def build_model(cfg: DictConfig, device: str) -> TransformerLSTMModel:
    model_config = ModelConfig()
    model_config.model_name = cfg.model.name
    return TransformerLSTMModel(
        pre_model_name=cfg.model.pre_model,
        device=device,
        input_dim=cfg.model.input_dim,
        linear_hidden_dim=cfg.model.linear_hidden_dim,
        num_heads=cfg.model.num_heads,
        num_blocks=cfg.model.num_blocks,
        dropout=cfg.model.dropout,
        model_config=model_config,
        with_lstm=cfg.model.with_lstm,
        lstm_n_layers=cfg.model.lstm_n_layers,
    )


def evaluate_on_caid(model: TransformerLSTMModel, caid: dict, device: str) -> dict:
    model.model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_proba: list[float] = []

    for seq, labels in zip(caid["sequences"], caid["disorder region"]):
        with torch.no_grad():
            proba = predict_proba_sequence(model, list(seq))
        all_true.extend(labels)
        all_pred.extend((proba >= 0.5).astype(int).tolist())
        all_proba.extend(proba.tolist())

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    proba_arr = np.array(all_proba)

    return {
        "f1": f1_score(true_arr, pred_arr, average="macro", zero_division=1),
        "precision": precision_score(true_arr, pred_arr, average="macro", zero_division=1),
        "recall": recall_score(true_arr, pred_arr, average="macro", zero_division=1),
        "auc": roc_auc_score(true_arr, proba_arr),
        "aupr": average_precision_score(true_arr, proba_arr),
    }


def log_run(cfg_path: str, model_name: str, elapsed_min: float, extra: dict) -> Path:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    log = {
        "config": cfg_path,
        "commit": commit,
        "duration_min": round(elapsed_min, 1),
        **extra,
    }
    log_dir = Path("data_repository/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_name}_{int(time.time())}.json"
    log_path.write_text(json.dumps(log, indent=2))
    return log_path
