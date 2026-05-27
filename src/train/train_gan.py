"""
GAN training script: trains unconditional per-token embedding GAN on DisProt.

Prerequisites:
    data_repository/ckpt/baseline_transformer_lstm.pth  (run train_baseline first)

Usage:
    python -m src.train.train_gan configs/gan.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.augmentation.dataset import build_synthetic_dataset
from src.augmentation.decoder import EmbeddingDecoder
from src.augmentation.embeddings import extract_embeddings
from src.augmentation.gan import Discriminator, Generator
from src.augmentation.pseudo_label import pseudo_label
from src.config import ModelConfig
from src.models.sequence_models import TransformerLSTMModel
from src.train.utils import log_run

_BASELINE_CKPT = Path("data_repository/ckpt/baseline_transformer_lstm.pth")
_BASELINE_SHAPE = dict(
    pre_model_name="Rostlab/prot_bert_bfd",
    input_dim=1024,
    linear_hidden_dim=64,
    num_heads=4,
    num_blocks=2,
    dropout=0.6,
    with_lstm=True,
    lstm_n_layers=2,
)


def _load_baseline(device: str) -> TransformerLSTMModel:
    model_config = ModelConfig()
    model_config.model_name = "baseline_transformer_lstm"
    baseline = TransformerLSTMModel(device=device, model_config=model_config, **_BASELINE_SHAPE)
    baseline.load()
    baseline.model.eval()
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    if not _BASELINE_CKPT.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {_BASELINE_CKPT}\n"
            "Run train_baseline first:\n"
            "  python -m src.train.train_baseline configs/baseline.yaml --save-checkpoint"
        )

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(OmegaConf.to_yaml(cfg))

    print("Loading baseline model...")
    baseline = _load_baseline(device)
    print("Baseline loaded.")

    emb_path = Path(cfg.data.embeddings_path)
    if emb_path.exists():
        print(f"Loading cached embeddings from {emb_path}")
        embeddings = torch.load(emb_path, map_location="cpu")
    else:
        print("Extracting embeddings (runs once, ~10 min on RTX 3050)...")
        embeddings = extract_embeddings(
            baseline.model.pre_model,
            baseline.model.pre_tokenizer,
            device,
            str(emb_path),
        )

    all_embs = torch.cat(embeddings, dim=0).float()
    print(f"Total residue embeddings: {all_embs.shape[0]:,}  dim={all_embs.shape[1]}")

    G = Generator(latent_dim=cfg.gan.latent_dim).to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.gan.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.gan.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    batch_size = cfg.gan.batch_size
    n_epoch = cfg.gan.n_epoch
    n = len(all_embs)
    g_losses: list[float] = []
    d_losses: list[float] = []
    t_start = time.time()

    for epoch in range(n_epoch):
        G.train()
        D.train()
        perm = torch.randperm(n)
        epoch_g: list[float] = []
        epoch_d: list[float] = []

        for start in range(0, n - batch_size, batch_size):
            real = all_embs[perm[start : start + batch_size]].to(device)
            bs = real.size(0)

            z = torch.randn(bs, cfg.gan.latent_dim, device=device)
            fake = G(z).detach()
            r_lab = torch.full((bs, 1), cfg.gan.real_label_smooth, device=device)
            f_lab = torch.zeros(bs, 1, device=device)
            d_loss = bce(D(real), r_lab) + bce(D(fake), f_lab)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            z = torch.randn(bs, cfg.gan.latent_dim, device=device)
            g_loss = bce(D(G(z)), torch.ones(bs, 1, device=device))
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            epoch_g.append(g_loss.item())
            epoch_d.append(d_loss.item())

        g_losses.append(float(np.mean(epoch_g)))
        d_losses.append(float(np.mean(epoch_d)))
        elapsed = (time.time() - t_start) / 60
        print(
            f"Epoch {epoch+1}/{n_epoch}  "
            f"G={g_losses[-1]:.4f}  D={d_losses[-1]:.4f}  ({elapsed:.1f} min)"
        )

    ckpt_dir = Path("data_repository/ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), ckpt_dir / "gan_generator.pth")
    torch.save(D.state_dict(), ckpt_dir / "gan_discriminator.pth")
    print("Generator and discriminator checkpoints saved.")

    print(f"\nGenerating {cfg.data.n_synthetic_chunks} synthetic sequences...")
    build_synthetic_dataset(
        n_chunks=cfg.data.n_synthetic_chunks,
        chunk_len=cfg.data.chunk_len,
        generator=G,
        decoder=EmbeddingDecoder(),
        model=baseline,
        device=device,
        save_path=cfg.data.synthetic_path,
        latent_dim=cfg.gan.latent_dim,
    )
    print(f"Synthetic dataset saved to {cfg.data.synthetic_path}")

    print("\nDisorder rate sanity check (100 generated sequences)...")
    G.eval()
    decoder = EmbeddingDecoder()
    sample_seqs: list[list[str]] = []
    with torch.no_grad():
        for _ in range(100):
            z = torch.randn(50, cfg.gan.latent_dim, device=device)
            sample_seqs.append(decoder.decode_sequence(G(z).cpu()))
    sample_labels = pseudo_label(sample_seqs, baseline)
    total_res = sum(len(l) for l in sample_labels)
    disordered = sum(sum(l) for l in sample_labels)
    disorder_rate = disordered / total_res if total_res else 0.0
    print(f"Generated disorder rate: {100 * disorder_rate:.1f}%  (real DisProt: ~16%)")
    print("A rate << 16% confirms the unconditional GAN is order-biased (expected).")

    total_elapsed = (time.time() - t_start) / 60
    log_path = log_run(
        cfg_path=args.config,
        model_name="gan",
        elapsed_min=total_elapsed,
        extra={
            "final_g_loss": round(g_losses[-1], 4),
            "final_d_loss": round(d_losses[-1], 4),
            "n_synthetic_chunks": cfg.data.n_synthetic_chunks,
            "disorder_rate_pct": round(100 * disorder_rate, 1),
        },
    )
    print(f"\nRun log: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
