# path: ~/Develop/dnn_test/src/first/evaluation_utils.py
# 표준 라이브러리
import os
import logging
from datetime import datetime
from typing import Dict, Tuple, Any

# 써드파티 라이브러리
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# 로컬 모듈
from visualization_utils import plot_training_history, plot_evaluation_metrics


def setup_logger(output_dir: str, timestamp: str) -> logging.Logger:
    """로깅 설정을 초기화합니다."""
    logger = logging.getLogger(f"evaluation_{timestamp}")
    logger.setLevel(logging.INFO)

    # 파일 핸들러 설정
    log_file = os.path.join(output_dir, f"evaluation_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포매터 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_output_directory(
    base_path: str, timestamp: str = None
) -> Tuple[str, logging.Logger]:
    """결과물을 저장할 디렉토리를 생성합니다."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")

    output_dir = os.path.join(base_path, f"result_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 하위 디렉토리 생성
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger = setup_logger(output_dir, timestamp)
    logger.info(f"Created output directory: {output_dir}")

    return output_dir, logger


def save_results_summary(
    output_dir: str,
    train_metrics: Tuple[float, float],
    test_metrics: Tuple[float, float],
    model_summary: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamp: str,
    logger: logging.Logger,
) -> None:
    """평가 결과 요약을 저장합니다."""
    y_true_class = y_true.argmax(axis=1)
    y_pred_class = y_pred.argmax(axis=1)

    with open(os.path.join(output_dir, f"results_summary_{timestamp}.txt"), "w") as f:
        f.write("=== Model Evaluation Results ===\n\n")

        # 기본 메트릭
        f.write("Basic Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write("Training:\n")
        f.write(f"Loss: {train_metrics[0]:.4f}\n")
        f.write(f"Accuracy: {train_metrics[1]:.4f}\n\n")

        f.write("Testing:\n")
        f.write(f"Loss: {test_metrics[0]:.4f}\n")
        f.write(f"Accuracy: {test_metrics[1]:.4f}\n\n")

        # 추가 평가 지표
        f.write("Detailed Metrics:\n")
        f.write("-" * 20 + "\n")

        # Precision, Recall, F1-Score
        f.write("\nPer-Class Metrics:\n")
        for idx, class_name in enumerate(["Negative", "Positive"]):
            precision = precision_score(
                y_true_class, y_pred_class, labels=[idx], average="binary"
            )
            recall = recall_score(
                y_true_class, y_pred_class, labels=[idx], average="binary"
            )
            f1 = f1_score(y_true_class, y_pred_class, labels=[idx], average="binary")

            f.write(f"\n{class_name} Class:\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")

        # Overall Metrics
        f.write("\nOverall Metrics:\n")
        f.write(
            f"Macro F1-Score: {f1_score(y_true_class, y_pred_class, average='macro'):.4f}\n"
        )
        f.write(
            f"Weighted F1-Score: {f1_score(y_true_class, y_pred_class, average='weighted'):.4f}\n"
        )

        # Classification Report
        f.write("\nClassification Report:\n")
        f.write("-" * 20 + "\n")
        report = classification_report(
            y_true_class, y_pred_class, target_names=["Negative", "Positive"]
        )
        f.write(report)

        # Model Architecture
        f.write("\n=== Model Architecture ===\n")
        f.write("-" * 20 + "\n")
        f.write(model_summary)

    logger.info(f"Saved detailed results summary to {output_dir}")


def evaluate_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    batch_size: int = 32,
    output_dir: str = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """모델을 평가하고 결과를 저장합니다."""
    timestamp = datetime.now().strftime("%m%d_%H%M")

    if output_dir is None:
        return model.evaluate(X_train, Y_train), model.evaluate(X_test, Y_test)

    logger = setup_logger(output_dir, timestamp)

    # 모델 요약 정보 가져오기
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)

    # 예측 및 평가
    logger.info("Starting model evaluation...")
    y_train_pred = model.predict(X_train, batch_size=batch_size)
    y_test_pred = model.predict(X_test, batch_size=batch_size)

    train_metrics = model.evaluate(X_train, Y_train, batch_size=batch_size, verbose=1)
    test_metrics = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)

    # 결과 저장
    plot_evaluation_metrics(Y_test, y_test_pred, output_dir, timestamp, logger)
    save_results_summary(
        output_dir,
        train_metrics,
        test_metrics,
        model_summary,
        Y_test,
        y_test_pred,
        timestamp,
        logger,
    )

    logger.info("Evaluation completed successfully")
    return (train_metrics[0], train_metrics[1]), (test_metrics[0], test_metrics[1])
