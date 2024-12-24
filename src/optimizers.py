# path: ~/Develop/dnn_test/src/optimizers.py
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, SGD
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)
import math


class ModelOptimizer:
    def __init__(self):
        self.current_config = {
            "hidden_sizes": [128, 256, 512, 512, 256, 128],
            "dropout_rates": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "learning_rate": 0.001,
            "use_batch_norm": False,
            "optimizer": "adam",
            "optimizer_config": {
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
                "weight_decay": 0.01,
                "momentum": 0.9,
                "rho": 0.9,
            },
            "classification_type": "binary",
            "l2_lambda": 0.01,
            "label_smoothing": 0.0,
            "use_cosine_scheduler": False,
            "use_reduce_lr": False,
            "reduce_lr_patience": 10,
            "reduce_lr_factor": 0.1,
            "use_early_stopping": False,
            "early_stopping_patience": 20,
            "batch_size": 32,
            "epochs": 300,
            "use_class_weights": False,
            "class_weights": None,
        }

    def get_optimizer(self):
        """설정된 옵티마이저 반환"""
        opt_name = self.current_config["optimizer"].lower()
        lr = self.current_config["learning_rate"]
        config = self.current_config["optimizer_config"]

        if opt_name == "adam":
            return Adam(
                learning_rate=lr,
                beta_1=config["beta_1"],
                beta_2=config["beta_2"],
                epsilon=config["epsilon"],
            )
        elif opt_name == "adamw":
            return AdamW(
                learning_rate=lr,
                weight_decay=config["weight_decay"],
                beta_1=config["beta_1"],
                beta_2=config["beta_2"],
            )
        elif opt_name == "rmsprop":
            return RMSprop(learning_rate=lr, rho=config["rho"])
        elif opt_name == "sgd":
            return SGD(learning_rate=lr, momentum=config["momentum"])
        else:
            return Adam(learning_rate=lr)

    def get_callbacks(self, epochs):
        """콜백 함수들 반환"""
        callbacks = []

        if self.current_config["use_cosine_scheduler"]:

            def cosine_decay(epoch):
                return (
                    0.5
                    * (1 + math.cos(math.pi * epoch / epochs))
                    * self.current_config["learning_rate"]
                )

            callbacks.append(LearningRateScheduler(cosine_decay))

        if self.current_config["use_reduce_lr"]:
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.current_config["reduce_lr_factor"],
                patience=self.current_config["reduce_lr_patience"],
                min_lr=1e-6,
                verbose=1,
            )
            callbacks.append(reduce_lr)

        if self.current_config["use_early_stopping"]:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.current_config["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            )
            callbacks.append(early_stopping)

        return callbacks

    def get_compile_kwargs(self):
        """모델 컴파일을 위한 인자들 반환"""
        kwargs = {"optimizer": self.get_optimizer(), "metrics": ["accuracy"]}

        if self.current_config.get("classification_type", "binary") == "binary":
            kwargs["loss"] = "binary_crossentropy"
        else:
            kwargs["loss"] = "sparse_categorical_crossentropy"

        return kwargs
