# 표준 라이브러리
import os
import logging
from typing import Tuple, Dict

# 써드파티 라이브러리
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    History,
    ReduceLROnPlateau,
    EarlyStopping,
)
from tensorflow.keras.regularizers import l1_l2


class ModelBuilder:
    def __init__(self, config: Dict, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def build_model(self) -> Sequential:
        """모델 구조 정의"""
        model = Sequential()

        # 첫 번째 레이어
        layer_args = {"units": 64, "activation": "relu", "input_dim": 48}
        if self.config.get("use_regularization"):
            layer_args["kernel_regularizer"] = l1_l2(
                l1=self.config.get("l1_reg", 1e-5), l2=self.config.get("l2_reg", 1e-4)
            )
        model.add(Dense(**layer_args))

        if self.config.get("use_batch_norm"):
            model.add(BatchNormalization())
        if self.config.get("use_dropout"):
            model.add(Dropout(self.config.get("dropout_rate", 0.3)))

        # 히든 레이어들
        layer_sizes = [128, 256, 256, 128]
        for size in layer_sizes:
            layer_args = {"units": size, "activation": "relu"}
            if self.config.get("use_regularization"):
                layer_args["kernel_regularizer"] = l1_l2(
                    l1=self.config.get("l1_reg", 1e-6),
                    l2=self.config.get("l2_reg", 1e-5),
                )
            model.add(Dense(**layer_args))

            if self.config.get("use_batch_norm"):
                model.add(BatchNormalization())
            if self.config.get("use_dropout"):
                model.add(Dropout(self.config.get("dropout_rate", 0.2)))

        # 출력 레이어
        model.add(Dense(self.config["nb_classes"], activation="softmax"))

        opt = Adam(self.config["learning_rate"], clipnorm=1.0)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        self.logger.info("Model built and compiled successfully")
        return model

    def create_callbacks(self, output_dir: str, timestamp: str) -> list:
        """콜백 함수 생성"""
        weights_dir = os.path.join(output_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                os.path.join(weights_dir, f"model_checkpoint_{timestamp}"),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                save_format="tf",
            ),
            History(),
        ]

        # Learning Rate Scheduling이 설정된 경우
        if self.config.get("use_lr_schedule"):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.config.get("lr_factor", 0.2),
                    patience=self.config.get("lr_patience", 5),
                    min_lr=self.config.get("min_lr", 1e-6),
                    verbose=1,
                )
            )

        # Early Stopping이 설정된 경우
        if self.config.get("use_early_stopping"):
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.get("early_stopping_patience", 10),
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        self.logger.info(f"Created callbacks with checkpoint dir: {weights_dir}")
        return callbacks

    def train_model(
        self, model: Sequential, X_train, Y_train, callbacks: list
    ) -> History:
        """모델 학습"""
        history = model.fit(
            X_train,
            Y_train,
            verbose=2,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            validation_split=0.15,
            shuffle=True,
            callbacks=callbacks,
        )

        self.logger.info("Model training completed")
        return history
