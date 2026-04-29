import os
from base.config import Config

RAW_DATA_DIR = os.getcwd() + "/data_repository/raw"
PROCESSED_DATA_DIR = os.getcwd() + "/data_repository/processed"
LOG_DIR = os.getcwd() + "/data_repository/log"
MODEL_SAVED_DIR = os.getcwd() + "/data_repository/ckpt"
BENCHMARK_DIR = os.getcwd() + "/data_repository/benchmarks"

DATASETS = {1: "DisProtDatabase.json"}
BENCHMARKS = {1: "CAID", 2: "CASP10"}

CAID_BENCHMARK = os.path.join(BENCHMARK_DIR, BENCHMARKS[1], "disprot-disorder.txt")

DISPROT_DATABASE_FILE = os.path.join(RAW_DATA_DIR, DATASETS[1])

ALL_DATA_TEMPLATE = "{dataset}_all_data.json"
CUT_DATA_TEMPLATE = "{dataset}_cut_data.json"


class ModelConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = None

    def get_configuration(self):
        return {
            "model_name": self.model_name,
        }

    def get_summary(self):
        return {
            "model_name": self.model_name,
        }


class OptimizerConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.lr = None  # learning rate
        self.batch_size = None
        self.n_epoch = None
        self.device = "cuda"
        self.save = False
        self.first_class_weight = 0.1

    def get_configuration(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
            "first class weight": self.first_class_weight,
        }

    def get_summary(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epoch": self.n_epoch,
            "first class weight": self.first_class_weight,
        }
