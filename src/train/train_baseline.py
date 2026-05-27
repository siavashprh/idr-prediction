"""
Baseline training script.

Usage:
    python -m src.train.train_baseline configs/baseline.yaml [--save-checkpoint]
"""

import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.config import OptimizerConfig
from src.data.data import IDPData, get_caid_data, get_disprot_cut_data
from src.models.sequence_models import TransformerLSTMModel
from src.optimization.optimization import IDRTrainer
from src.train.utils import build_model, evaluate_on_caid, log_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(OmegaConf.to_yaml(cfg))

    print("Loading training data...")
    cut = get_disprot_cut_data()
    train_data = IDPData(cut["sequences"], cut["disorder region"])
    print(f"Training sequences: {len(train_data.X)}")

    print("Loading CAID data...")
    caid = get_caid_data()
    print(f"CAID sequences: {len(caid['sequences'])}")

    print("Building model...")
    model = build_model(cfg, device)

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

    if args.save_checkpoint:
        from src.config import MODEL_SAVED_DIR
        ckpt_path = Path(MODEL_SAVED_DIR) / f"{cfg.model.name}.pth"
        torch.save(model.model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("\nEvaluating on CAID...")
    caid_metrics = evaluate_on_caid(model, caid, device)
    print("\nCAID results:")
    for k, v in caid_metrics.items():
        print(f"  {k}: {v:.4f}")

    log_path = log_run(
        cfg_path=args.config,
        model_name=cfg.model.name,
        elapsed_min=elapsed / 60,
        extra={
            "train_f1": round(train_result.f1, 4),
            "caid": {k: round(v, 4) for k, v in caid_metrics.items()},
        },
    )
    print(f"\nRun log: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
