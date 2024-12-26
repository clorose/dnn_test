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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


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

        # 예측 수행
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = (
            (y_pred > 0.5).astype(int)
            if self.y_test.ndim == 1
            else np.argmax(y_pred, axis=1)
        )

        # Confusion Matrix 계산 및 정규화 (0으로 나누기 방지)
        cm = confusion_matrix(self.y_test, y_pred_classes)
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
        cm_normalized = cm.astype("float") / row_sums[:, np.newaxis]

        # ROC-AUC 계산 (이진분류인 경우만)
        roc_auc = roc_auc_score(self.y_test, y_pred) if self.y_test.ndim == 1 else None

        # 1. metrics.json에 모든 정보 저장 (training_history.json 대체)
        metrics_dict = {
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
            # 새로운 테스트 메트릭 추가
            "test_metrics": {
                "f1_score": float(
                    f1_score(
                        self.y_test, y_pred_classes, average="weighted", zero_division=0
                    )
                ),
                "precision": float(
                    precision_score(
                        self.y_test, y_pred_classes, average="weighted", zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        self.y_test, y_pred_classes, average="weighted", zero_division=0
                    )
                ),
                "confusion_matrix": {
                    "raw": cm.tolist(),
                    "normalized": cm_normalized.tolist(),
                },
            },
        }

        if roc_auc is not None:
            metrics_dict["test_metrics"]["roc_auc"] = float(roc_auc)

        with open(os.path.join(experiment_dir, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # 분류 보고서 저장
        report = classification_report(self.y_test, y_pred_classes, zero_division=0)
        with open(os.path.join(experiment_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # 시각화
        plt.figure(figsize=(15, 5))

        # 1. Confusion Matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # 2. Learning Curves
        plt.subplot(1, 3, 2)
        plt.plot(history.history["accuracy"], label="Train")
        plt.plot(history.history["val_accuracy"], label="Val")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # 3. ROC Curve (이진분류인 경우만)
        plt.subplot(1, 3, 3)
        if roc_auc is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "results_summary.png"))
        plt.close()

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

        return {
            "best_val_loss": val_loss,
            "best_val_accuracy": val_accuracy,
            "final_val_loss": history.history["val_loss"][-1],
            "final_val_accuracy": history.history["val_accuracy"][-1],
            "history": history.history,
        }

    def _calculate_trend(self, values, min_change=0.001):
        """최근 추세 분석"""
        if len(values) < 2:
            return "insufficient_data"

        changes = np.diff(values)
        improving = np.sum(changes > min_change)
        declining = np.sum(changes < -min_change)
        stable = np.sum(np.abs(changes) <= min_change)

        if improving > declining and improving > stable:
            return "improving"
        elif declining > improving and declining > stable:
            return "declining"
        else:
            return "stable"

    def _detect_plateau(self, values, window=5, threshold=0.001):
        """성능 정체 시점 감지"""
        if len(values) < window:
            return None

        for i in range(window, len(values)):
            recent_vals = values[i - window : i]
            if np.std(recent_vals) < threshold:
                return i
        return None

    def _check_early_convergence(self, history, window_size=10):
        """학습의 조기 수렴 여부 확인"""
        val_acc = history["val_accuracy"]
        if len(val_acc) < window_size:
            return False

        recent_improvement = max(val_acc[-window_size:]) - min(val_acc[-window_size:])
        return recent_improvement < 0.001

    def _calculate_stability(self, history, window_size=10):
        """학습 안정성 계산"""
        if len(history["val_accuracy"]) < window_size:
            return "insufficient_data"

        recent_val_acc = history["val_accuracy"][-window_size:]
        std_dev = np.std(recent_val_acc)

        if std_dev < 0.01:
            return "very_stable"
        elif std_dev < 0.03:
            return "stable"
        elif std_dev < 0.05:
            return "moderately_stable"
        else:
            return "unstable"

    def _calculate_overfitting_metrics(self, history):
        """과적합 관련 메트릭 계산"""
        train_acc = history["accuracy"]
        val_acc = history["val_accuracy"]

        gaps = np.array(train_acc) - np.array(val_acc)
        significant_gap_epoch = np.where(gaps > 0.05)[0]
        overfitting_start = (
            significant_gap_epoch[0] if len(significant_gap_epoch) > 0 else None
        )

        return {
            "overfitting_start_epoch": (
                int(overfitting_start) if overfitting_start is not None else None
            ),
            "max_accuracy_gap": float(max(gaps)),
            "final_accuracy_gap": float(gaps[-1]),
            "overfitting_severity": (
                "high" if max(gaps) > 0.1 else "medium" if max(gaps) > 0.05 else "low"
            ),
        }

    def _calculate_convergence_speed(self, history, threshold=0.95):
        """수렴 속도 분석"""
        max_acc = max(history["accuracy"])
        target_acc = max_acc * threshold

        for epoch, acc in enumerate(history["accuracy"]):
            if acc >= target_acc:
                return {
                    "epochs_to_95_percent_max": epoch + 1,
                    "initial_learning_rate": acc / (epoch + 1) if epoch > 0 else None,
                }

        return {"epochs_to_95_percent_max": None, "initial_learning_rate": None}

    def _calculate_improvement_rate(self, values, window=5):
        """개선 속도 계산"""
        if len(values) < window:
            return None

        improvements = np.diff(values)
        avg_improvement = np.mean(improvements)

        return float(avg_improvement)

    def _identify_learning_phases(self, history, threshold=0.01):
        """학습 단계 식별"""
        improvements = np.diff(history["accuracy"])

        rapid_learning = np.sum(improvements > threshold)
        slow_learning = np.sum(
            (improvements <= threshold) & (improvements > threshold / 10)
        )
        plateau = np.sum(improvements <= threshold / 10)

        return {
            "rapid_learning_epochs": int(rapid_learning),
            "slow_learning_epochs": int(slow_learning),
            "plateau_epochs": int(plateau),
            "primary_phase": (
                "rapid"
                if rapid_learning > slow_learning and rapid_learning > plateau
                else (
                    "slow"
                    if slow_learning > rapid_learning and slow_learning > plateau
                    else "plateau"
                )
            ),
        }

    def _analyze_stability_phases(self, history, window=5):
        """안정성 단계 분석"""
        val_acc = np.array(history["val_accuracy"])

        stability_windows = []
        current_stability = "unknown"
        phase_length = 0

        for i in range(window, len(val_acc)):
            std_dev = np.std(val_acc[i - window : i])

            if std_dev < 0.01:
                new_stability = "stable"
            elif std_dev < 0.03:
                new_stability = "moderate"
            else:
                new_stability = "unstable"

            if new_stability != current_stability:
                if current_stability != "unknown":
                    stability_windows.append(
                        {
                            "phase": current_stability,
                            "length": phase_length,
                            "start_epoch": i - phase_length,
                            "end_epoch": i,
                        }
                    )
                current_stability = new_stability
                phase_length = 1
            else:
                phase_length += 1

        # 마지막 phase 추가
        if phase_length > 0:
            stability_windows.append(
                {
                    "phase": current_stability,
                    "length": phase_length,
                    "start_epoch": len(val_acc) - phase_length,
                    "end_epoch": len(val_acc),
                }
            )

        return {
            "stability_phases": stability_windows,
            "dominant_phase": (
                max(stability_windows, key=lambda x: x["length"])["phase"]
                if stability_windows
                else None
            ),
        }
