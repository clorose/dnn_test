# path: ~/Develop/dnn_test/src/experiment.py
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2


class ModelExperiment:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.histories = {}
        self.model = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.result_dir = os.path.join(
            project_root, "runs", f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        os.makedirs(self.result_dir, exist_ok=True)

    def create_model(self, optimizer_config):
        """최적화 설정을 적용한 모델 생성"""
        model = tf.keras.Sequential()

        # 입력층
        model.add(
            Dense(
                optimizer_config.current_config["hidden_sizes"][0],
                input_dim=self.X_train.shape[1],
                kernel_regularizer=l2(optimizer_config.current_config["l2_lambda"]),
            )
        )
        if optimizer_config.current_config["use_batch_norm"]:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(optimizer_config.current_config["dropout_rates"][0]))

        # 은닉층들
        for i in range(1, len(optimizer_config.current_config["hidden_sizes"])):
            model.add(
                Dense(
                    optimizer_config.current_config["hidden_sizes"][i],
                    kernel_regularizer=l2(optimizer_config.current_config["l2_lambda"]),
                )
            )
            if optimizer_config.current_config["use_batch_norm"]:
                model.add(BatchNormalization())
            model.add(Activation("relu"))
            model.add(Dropout(optimizer_config.current_config["dropout_rates"][i]))

        # 출력층
        if optimizer_config.current_config["classification_type"] == "binary":
            model.add(Dense(1, activation="sigmoid"))
        else:
            model.add(Dense(3, activation="softmax"))

        # 컴파일
        compile_kwargs = optimizer_config.get_compile_kwargs()
        model.compile(**compile_kwargs)

        return model

    def run_experiment(self, name, optimizer_config):
        """실험 실행 및 결과 저장"""
        self.model = self.create_model(optimizer_config)

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.1,
            epochs=optimizer_config.current_config["epochs"],
            batch_size=optimizer_config.current_config["batch_size"],
            callbacks=optimizer_config.get_callbacks(
                optimizer_config.current_config["epochs"]
            ),
            shuffle=True,
            verbose=1,
        )

        test_score = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        self.histories[name] = {
            "history": history.history,
            "config": optimizer_config.current_config.copy(),
            "test_score": test_score,
        }

        self.save_results(name, history)

        return history

    def save_results(self, name, history):
        """실험 결과 저장"""
        experiment_dir = os.path.join(self.result_dir, name)
        os.makedirs(experiment_dir, exist_ok=True)

        # 학습 히스토리 저장
        history_dict = {
            "accuracy": history.history["accuracy"],
            "val_accuracy": history.history["val_accuracy"],
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
            "max_accuracy": float(max(history.history["accuracy"])),
            "min_loss": float(min(history.history["loss"])),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_epoch": int(
                history.history["val_accuracy"].index(
                    max(history.history["val_accuracy"])
                )
            ),
        }

        with open(os.path.join(experiment_dir, "metrics.json"), "w") as f:
            json.dump(history_dict, f, indent=4)

        # 학습 곡선 저장
        self.plot_learning_curves(history, name, experiment_dir)

        # 모델 저장
        self.model.save(os.path.join(experiment_dir, "model.keras"))

    def plot_learning_curves(self, history, name, save_dir):
        """학습 곡선 그리기"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 정확도 그래프
        ax1.plot(history.history["accuracy"], label="Training")
        ax1.plot(history.history["val_accuracy"], label="Validation")
        ax1.set_title(f"{name} - Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # 손실 그래프
        ax2.plot(history.history["loss"], label="Training")
        ax2.plot(history.history["val_loss"], label="Validation")
        ax2.set_title(f"{name} - Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curves.png"))
        plt.close()

    def run_optimization(self, config):
        """최적화 실행"""

        self.model = self.create_model(config)

        history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.1,
            epochs=config.current_config["epochs"],
            batch_size=config.current_config["batch_size"],
            callbacks=config.get_callbacks(config.current_config["epochs"]),
            shuffle=True,
            verbose=1,
        )

        val_loss = min(history.history["val_loss"])
        val_accuracy = max(history.history["val_accuracy"])

        # 모든 중요 지표 반환
        return {
            "best_val_loss": val_loss,
            "best_val_accuracy": val_accuracy,
            "final_val_loss": history.history["val_loss"][-1],
            "final_val_accuracy": history.history["val_accuracy"][-1],
            "history": history.history,
        }
