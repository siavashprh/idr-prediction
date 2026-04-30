"""
Baseline training script.

Usage:
    python -m src.train.train_baseline configs/baseline.yaml
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.config import ModelConfig, OptimizerConfig
from src.data.data import IDPData, get_caid_data, get_disprot_cut_data
from src.methods.ensemble_sequence_prediction import predict_sequence
from src.models.sequence_models import TransformerLSTMModel
from src.optimization.optimization import IDRTrainer


def load_config(path: str):
    return OmegaConf.load(path)


def build_model(cfg, device: str) -> TransformerLSTMModel:
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
    sequences = caid["sequences"]
    true_labels = caid["disorder region"]

    all_true: list[int] = []
    all_pred: list[int] = []
    all_proba: list[float] = []

    for seq, labels in zip(sequences, true_labels):
        # Use sliding-window predict for sequences longer than the training cut
        pred_array = predict_sequence(model, list(seq), max_length=400, alpha=0.75)
        pred_binary = (pred_array >= 0.5).astype(int).tolist()

        # For probabilities we need a parallel proba sliding window
        proba_array = _predict_proba_sequence(model, list(seq), max_length=400, alpha=0.75)

        all_true.extend(labels)
        all_pred.extend(pred_binary)
        all_proba.extend(proba_array.tolist())

    all_true_arr = np.array(all_true)
    all_pred_arr = np.array(all_pred)
    all_proba_arr = np.array(all_proba)

    return {
        "f1": f1_score(all_true_arr, all_pred_arr, average="macro", zero_division=1),
        "precision": precision_score(all_true_arr, all_pred_arr, average="macro", zero_division=1),
        "recall": recall_score(all_true_arr, all_pred_arr, average="macro", zero_division=1),
        "auc": roc_auc_score(all_true_arr, all_proba_arr),
    }


def _predict_proba_sequence(
    model: TransformerLSTMModel, sequence: list, max_length: int = 400, alpha: float = 0.75
) -> np.ndarray:
    """Sliding-window variant of predict_proba for long sequences."""
    if len(sequence) <= max_length:
        proba = model.predict_proba([sequence])
        return np.array(proba if isinstance(proba, list) else [proba])

    y = np.zeros(len(sequence))
    weight = np.zeros(len(sequence))

    start = 0
    while start < len(sequence):
        end = min(start + max_length, len(sequence))
        chunk = sequence[start:end]
        proba = model.predict_proba([chunk])
        chunk_arr = np.array(proba if isinstance(proba, list) else [proba])
        y[start:end] += chunk_arr
        weight[start:end] += 1.0
        if end == len(sequence):
            break
        start += int(max_length * alpha)

    return y / np.maximum(weight, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(OmegaConf.to_yaml(cfg))

    # --- Data ---
    print("Loading training data...")
    cut = get_disprot_cut_data()
    train_data = IDPData(cut["sequences"], cut["disorder region"])
    print(f"Training sequences: {len(train_data.X)}")

    print("Loading CAID data...")
    caid = get_caid_data()
    print(f"CAID sequences: {len(caid['sequences'])}")

    # --- Model ---
    print("Building model...")
    model = build_model(cfg, device)

    # --- Train ---
    optimizer_cfg = OptimizerConfig()
    optimizer_cfg.lr = cfg.training.lr
    optimizer_cfg.batch_size = cfg.training.batch_size
    optimizer_cfg.n_epoch = cfg.training.n_epoch
    optimizer_cfg.device = device
    optimizer_cfg.save = args.save_checkpoint
    optimizer_cfg.first_class_weight = cfg.training.first_class_weight

    trainer = IDRTrainer()
    print("\nTraining...")
    t0 = time.time()
    train_result = trainer.train(model=model, data=train_data, optimizer_config=optimizer_cfg)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")
    print(f"Train F1: {train_result.f1:.4f}")

    # --- Evaluate on CAID ---
    print("\nEvaluating on CAID...")
    caid_metrics = evaluate_on_caid(model, caid, device)
    print(f"\nCAID results:")
    for k, v in caid_metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Save checkpoint if requested ---
    if args.save_checkpoint:
        from src.config import MODEL_SAVED_DIR
        ckpt_path = Path(MODEL_SAVED_DIR) / f"{cfg.model.name}.pth"
        torch.save(model.model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    # --- Log run ---
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    log = {
        "config": args.config,
        "commit": commit,
        "device": device,
        "duration_min": round(elapsed / 60, 1),
        "train_f1": round(train_result.f1, 4),
        "caid": {k: round(v, 4) for k, v in caid_metrics.items()},
    }
    log_path = Path("data_repository/log") / f"{cfg.model.name}_{int(t0)}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nRun log: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
