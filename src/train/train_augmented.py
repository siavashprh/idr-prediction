"""
Augmented training script: Transformer-LSTM on real DisProt + GAN synthetic data.

Prerequisites:
    data_repository/processed/synthetic_disprot.json  (run train_gan first)

Usage:
    python -m src.train.train_augmented configs/augmented.yaml
"""

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.config import OptimizerConfig
from src.data.data import IDPData, get_caid_data, get_disprot_cut_data
from src.optimization.optimization import IDRTrainer
from src.train.utils import build_model, evaluate_on_caid, log_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    syn_path = Path(cfg.data.synthetic_path)
    if not syn_path.exists():
        raise FileNotFoundError(
            f"Synthetic data not found: {syn_path}\n"
            "Run train_gan first:\n"
            "  python -m src.train.train_gan configs/gan.yaml"
        )

    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(cfg))

    print("Loading real training data...")
    real = get_disprot_cut_data()

    print("Loading synthetic training data...")
    with open(syn_path) as f:
        synthetic = json.load(f)

    all_seqs = real["sequences"] + synthetic["sequences"]
    all_labels = real["disorder region"] + synthetic["disorder region"]
    train_data = IDPData(all_seqs, all_labels)
    print(f"  Real chunks      : {len(real['sequences'])}")
    print(f"  Synthetic chunks : {len(synthetic['sequences'])}")
    print(f"  Total            : {len(train_data.X)}")

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
    optimizer_cfg.save = False
    optimizer_cfg.first_class_weight = cfg.training.first_class_weight

    trainer = IDRTrainer()
    print("\nTraining...")
    t0 = time.time()
    train_result = trainer.train(model=model, data=train_data, optimizer_config=optimizer_cfg)
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")
    print(f"Train F1: {train_result.f1:.4f}")

    ckpt_dir = Path("data_repository/ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{cfg.model.name}.pth"
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
            "n_real": len(real["sequences"]),
            "n_synthetic": len(synthetic["sequences"]),
            "caid": {k: round(v, 4) for k, v in caid_metrics.items()},
        },
    )
    print(f"\nRun log: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
