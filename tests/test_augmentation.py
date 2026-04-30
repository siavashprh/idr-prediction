"""Unit tests for GAN augmentation pipeline."""

import json
import pytest
import torch
import torch.nn as nn

from src.augmentation.gan import Generator, Discriminator


class TestGenerator:
    def test_output_shape(self):
        g = Generator(latent_dim=128)
        z = torch.randn(4, 128)
        out = g(z)
        assert out.shape == (4, 1024)

    def test_output_in_tanh_range(self):
        g = Generator(latent_dim=128)
        z = torch.randn(100, 128)
        out = g(z)
        assert float(out.min()) >= -1.0
        assert float(out.max()) <= 1.0

    def test_custom_latent_dim(self):
        g = Generator(latent_dim=64)
        z = torch.randn(2, 64)
        out = g(z)
        assert out.shape == (2, 1024)


class TestDiscriminator:
    def test_output_shape(self):
        d = Discriminator()
        x = torch.randn(4, 1024)
        out = d(x)
        assert out.shape == (4, 1)

    def test_output_in_sigmoid_range(self):
        d = Discriminator()
        x = torch.randn(100, 1024)
        out = d(x)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0


from src.augmentation.decoder import EmbeddingDecoder, AA_VOCAB


class TestEmbeddingDecoder:
    def test_vocab_size(self):
        assert len(AA_VOCAB) == 20

    def test_decode_returns_valid_aa(self):
        decoder = EmbeddingDecoder()
        emb = torch.randn(1024)
        aa = decoder.decode(emb)
        assert aa in AA_VOCAB

    def test_decode_sequence_correct_length(self):
        decoder = EmbeddingDecoder()
        embs = torch.randn(10, 1024)
        seq = decoder.decode_sequence(embs)
        assert len(seq) == 10

    def test_decode_sequence_valid_aas(self):
        decoder = EmbeddingDecoder()
        embs = torch.randn(50, 1024)
        seq = decoder.decode_sequence(embs)
        assert all(aa in AA_VOCAB for aa in seq)

    def test_decoder_is_frozen(self):
        decoder = EmbeddingDecoder()
        for param in decoder.parameters():
            assert not param.requires_grad


import numpy as np
from src.augmentation.pseudo_label import pseudo_label


class MockPredictor:
    """Minimal stand-in for TransformerLSTMModel in tests."""
    def predict_proba(self, X: list) -> list:
        return [0.3] * len(X[0])


class TestPseudoLabel:
    def test_output_length_matches_input(self):
        model = MockPredictor()
        seqs = ["ACDEF", "MSTFP"]
        labels = pseudo_label(seqs, model)
        assert len(labels) == 2
        assert len(labels[0]) == 5
        assert len(labels[1]) == 5

    def test_labels_are_binary(self):
        model = MockPredictor()
        seqs = ["ACDEFGHIKL"]
        labels = pseudo_label(seqs, model)
        assert all(v in (0, 1) for v in labels[0])

    def test_high_proba_labelled_disordered(self):
        class HighProbaModel:
            def predict_proba(self, X):
                return [0.9] * len(X[0])
        labels = pseudo_label(["ACDE"], HighProbaModel())
        assert all(v == 1 for v in labels[0])

    def test_low_proba_labelled_ordered(self):
        class LowProbaModel:
            def predict_proba(self, X):
                return [0.1] * len(X[0])
        labels = pseudo_label(["ACDE"], LowProbaModel())
        assert all(v == 0 for v in labels[0])


from src.augmentation.dataset import build_synthetic_dataset
from src.data.data import IDPData


class TestBuildSyntheticDataset:
    def test_sequence_label_length_match(self, tmp_path):
        data = build_synthetic_dataset(
            n_chunks=3,
            chunk_len=10,
            generator=Generator(),
            decoder=EmbeddingDecoder(),
            model=MockPredictor(),
            device="cpu",
            save_path=str(tmp_path / "synthetic.json"),
        )
        for seq, labels in zip(data["sequences"], data["disorder region"]):
            assert len(seq) == len(labels), (
                f"seq len {len(seq)} != labels len {len(labels)}"
            )

    def test_correct_number_of_chunks(self, tmp_path):
        data = build_synthetic_dataset(
            n_chunks=5,
            chunk_len=8,
            generator=Generator(),
            decoder=EmbeddingDecoder(),
            model=MockPredictor(),
            device="cpu",
            save_path=str(tmp_path / "synthetic.json"),
        )
        assert len(data["sequences"]) == 5
        assert len(data["disorder region"]) == 5

    def test_json_saved_and_loadable(self, tmp_path):
        save_path = str(tmp_path / "synthetic.json")
        build_synthetic_dataset(
            n_chunks=3,
            chunk_len=10,
            generator=Generator(),
            decoder=EmbeddingDecoder(),
            model=MockPredictor(),
            device="cpu",
            save_path=save_path,
        )
        with open(save_path) as f:
            data = json.load(f)
        idp = IDPData(data["sequences"], data["disorder region"])
        assert len(idp.X) == 3
        assert len(idp.y) == 3

    def test_sequences_contain_valid_aas(self, tmp_path):
        data = build_synthetic_dataset(
            n_chunks=3,
            chunk_len=10,
            generator=Generator(),
            decoder=EmbeddingDecoder(),
            model=MockPredictor(),
            device="cpu",
            save_path=str(tmp_path / "synthetic.json"),
        )
        valid = set(AA_VOCAB)
        for seq in data["sequences"]:
            assert all(aa in valid for aa in seq), f"Invalid AA in: {seq}"
