"""Tests for src/train/utils.py shared training helpers."""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.train.utils import build_model, evaluate_on_caid, log_run


class TestBuildModel:
    def _cfg(self, name="test_model"):
        return OmegaConf.create({
            "model": {
                "name": name,
                "pre_model": "Rostlab/prot_bert_bfd",
                "input_dim": 1024,
                "linear_hidden_dim": 64,
                "num_heads": 4,
                "num_blocks": 2,
                "dropout": 0.6,
                "with_lstm": True,
                "lstm_n_layers": 2,
            }
        })

    def test_constructs_with_correct_hyperparams(self):
        with patch("src.train.utils.TransformerLSTMModel") as MockModel:
            build_model(self._cfg(), "cpu")
        kwargs = MockModel.call_args.kwargs
        assert kwargs["device"] == "cpu"
        assert kwargs["input_dim"] == 1024
        assert kwargs["linear_hidden_dim"] == 64
        assert kwargs["num_heads"] == 4
        assert kwargs["num_blocks"] == 2
        assert kwargs["dropout"] == 0.6
        assert kwargs["with_lstm"] is True
        assert kwargs["lstm_n_layers"] == 2

    def test_sets_model_name_from_cfg(self):
        with patch("src.train.utils.TransformerLSTMModel") as MockModel:
            build_model(self._cfg("my_model"), "cpu")
        model_config = MockModel.call_args.kwargs["model_config"]
        assert model_config.model_name == "my_model"


class TestEvaluateOnCaid:
    def _caid(self):
        return {
            "sequences": ["ACDEF"] * 3,
            "disorder region": [[0, 1, 0, 1, 0]] * 3,
        }

    def _mock_model(self):
        m = MagicMock()
        m.model = MagicMock()
        return m

    def test_returns_required_keys(self):
        proba = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        with patch("src.train.utils.predict_proba_sequence", return_value=proba):
            result = evaluate_on_caid(self._mock_model(), self._caid(), "cpu")
        assert set(result.keys()) == {"f1", "precision", "recall", "auc", "aupr"}

    def test_values_in_unit_interval(self):
        proba = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        with patch("src.train.utils.predict_proba_sequence", return_value=proba):
            result = evaluate_on_caid(self._mock_model(), self._caid(), "cpu")
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_calls_model_eval(self):
        model = self._mock_model()
        proba = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        with patch("src.train.utils.predict_proba_sequence", return_value=proba):
            evaluate_on_caid(model, self._caid(), "cpu")
        model.model.eval.assert_called_once()

    def test_perfect_predictions_yield_f1_one(self):
        proba = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        with patch("src.train.utils.predict_proba_sequence", return_value=proba):
            result = evaluate_on_caid(self._mock_model(), self._caid(), "cpu")
        assert result["f1"] == pytest.approx(1.0)
        assert result["auc"] == pytest.approx(1.0)


class TestLogRun:
    def test_writes_json_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("subprocess.check_output", return_value=b"abc1234\n"):
            path = log_run("configs/baseline.yaml", "test_model", 5.3, {"train_f1": 0.42})
        assert path.exists()

    def test_json_has_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("subprocess.check_output", return_value=b"abc1234\n"):
            path = log_run("configs/baseline.yaml", "test_model", 5.3, {"train_f1": 0.42})
        data = json.loads(path.read_text())
        assert data["config"] == "configs/baseline.yaml"
        assert data["commit"] == "abc1234"
        assert data["duration_min"] == 5.3
        assert data["train_f1"] == 0.42

    def test_log_filename_contains_model_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("subprocess.check_output", return_value=b"abc1234\n"):
            path = log_run("configs/baseline.yaml", "my_model", 1.0, {})
        assert "my_model" in path.name

    def test_creates_log_dir_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("subprocess.check_output", return_value=b"abc1234\n"):
            path = log_run("configs/baseline.yaml", "m", 1.0, {})
        assert path.exists()


class TestTrainGanPrerequisite:
    def test_fails_with_clear_error_when_baseline_ckpt_missing(self, tmp_path, monkeypatch):
        """train_gan.main() raises FileNotFoundError if baseline checkpoint is absent."""
        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["train_gan", "configs/gan.yaml"]):
            from src.train.train_gan import main
            with pytest.raises(FileNotFoundError, match="Baseline checkpoint not found"):
                main()


class TestTrainAugmentedPrerequisite:
    _MINIMAL_CFG = """\
model:
  name: test_augmented
  pre_model: Rostlab/prot_bert_bfd
  input_dim: 1024
  linear_hidden_dim: 64
  num_heads: 4
  num_blocks: 2
  dropout: 0.6
  with_lstm: true
  lstm_n_layers: 2
training:
  lr: 0.0003
  batch_size: 16
  n_epoch: 1
  first_class_weight: 0.1
  device: cpu
data:
  cut_length: 400
  alpha: 0.5
  synthetic_path: data_repository/processed/synthetic_disprot.json
"""

    def test_fails_with_clear_error_when_synthetic_data_missing(self, tmp_path, monkeypatch):
        """train_augmented.main() raises FileNotFoundError if synthetic data is absent."""
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "augmented_test.yaml"
        cfg_path.write_text(self._MINIMAL_CFG)
        with patch("sys.argv", ["train_augmented", str(cfg_path)]):
            from src.train.train_augmented import main
            with pytest.raises(FileNotFoundError, match="Synthetic data not found"):
                main()
