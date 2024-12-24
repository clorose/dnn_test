# path: ~/Develop/dnn_test/src/config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class ExperimentConfig:
    # 기본 설정
    tag: str
    hidden_sizes: List[int]
    dropout_rates: List[float]

    # 모델 구조 설정
    use_batch_norm: bool = False

    # 학습 설정
    batch_size: int = 32
    epochs: int = 300
    learning_rate: float = 0.001

    # 옵티마이저 설정
    optimizer: str = "adam"
    optimizer_config: Dict[str, Any] = None

    # 신규 추가: 분류 설정
    classification_type: str = "binary"

    # 정규화 설정
    l2_lambda: float = 0.01
    label_smoothing: float = 0.0

    # 학습률 스케줄링
    use_cosine_scheduler: bool = False
    use_reduce_lr: bool = False
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.1
    min_lr: float = 1e-6

    # 조기 종료 설정
    use_early_stopping: bool = False
    early_stopping_patience: int = 20

    # 클래스 가중치 설정
    use_class_weights: bool = False
    class_weights: Optional[Dict[int, float]] = None

    # 모니터링 설정
    monitor_metric: str = "val_loss"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if "optimizer_config" not in config_dict:
            config_dict["optimizer_config"] = {
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
            }

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "hidden_sizes": self.hidden_sizes,
            "dropout_rates": self.dropout_rates,
            "use_batch_norm": self.use_batch_norm,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "optimizer_config": self.optimizer_config,
            "classification_type": self.classification_type,
            "l2_lambda": self.l2_lambda,
            "label_smoothing": self.label_smoothing,
            "use_cosine_scheduler": self.use_cosine_scheduler,
            "use_reduce_lr": self.use_reduce_lr,
            "reduce_lr_patience": self.reduce_lr_patience,
            "reduce_lr_factor": self.reduce_lr_factor,
            "min_lr": self.min_lr,
            "use_early_stopping": self.use_early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "use_class_weights": self.use_class_weights,
            "class_weights": self.class_weights,
            "monitor_metric": self.monitor_metric,
        }

    def copy(self):
        """Creates a deep copy of the configuration"""
        from copy import deepcopy

        return deepcopy(self)
