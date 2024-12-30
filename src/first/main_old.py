# path: ~/Develop/dnn_test/src/first/main.py
# path: ~/Develop/dnn_test/src/first/main.py

# 1. 표준 라이브러리
import os
import sys
import glob
import datetime

# 2. 써드파티 라이브러리
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 3. 로컬 어플리케이션/라이브러리
from config import Config
from data_preprocessor import DataPreprocessor
from model_builder import ModelBuilder
from evaluation_utils import (
    evaluate_model,
    create_output_directory,
)
from visualization_utils import plot_training_history


def setup_environment(config: Config):
    """환경 설정"""
    # 재귀 제한 증가
    sys.setrecursionlimit(10000)

    # 시드 설정
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # CUDA 설정
    if tf.config.list_physical_devices("GPU"):
        tf.config.experimental.enable_op_determinism()


def load_data(config: Config) -> tuple:
    """데이터 로드"""
    train_sample = pd.read_csv(config.train_file_path, header=0, encoding="utf-8")
    experiment_files = glob.glob(
        os.path.join(config.virtual_data_path, "experiment_*.csv")
    )
    experiment_data = [
        pd.read_csv(f, index_col=None, header=0) for f in experiment_files
    ]

    return train_sample, experiment_data


def main(config):
    """메인 실행 함수"""
    # 설정 로드
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

    # 환경 설정
    setup_environment(config)

    # 출력 디렉토리 생성
    output_dir, logger = create_output_directory(config.run_data_path)
    logger.info("Starting the training process...")

    try:
        # 데이터 로드
        logger.info("Loading data...")
        train_sample, experiment_data = load_data(config)

        # 데이터 전처리
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(logger)
        train_info, stats = preprocessor.preprocess_train_sample(train_sample)

        # 통계 출력
        logger.info(f"Data statistics: {stats}")

        # 실험 데이터 전처리
        processed_data = preprocessor.preprocess_experiment_data(
            experiment_data, train_info
        )

        # 학습/테스트 데이터 준비
        logger.info("Preparing train/test data...")
        X_train, X_test, Y_train, Y_test = preprocessor.prepare_train_test_data(
            processed_data
        )

        # 데이터 형태 로깅
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        # 레이블 원-핫 인코딩
        Y_train = to_categorical(Y_train, config.model_params["nb_classes"])
        Y_test = to_categorical(Y_test, config.model_params["nb_classes"])

        # 모델 생성 및 학습
        logger.info("Building and training model...")
        model_builder = ModelBuilder(config.get_model_params(), logger)
        model = model_builder.build_model()
        callbacks = model_builder.create_callbacks(output_dir, timestamp)

        # 모델 구조 출력
        model.summary()

        # 모델 학습
        history = model_builder.train_model(model, X_train, Y_train, callbacks)

        # 학습 히스토리 시각화
        logger.info("Plotting training history...")
        plot_training_history(history, output_dir, timestamp, logger)

        # 모델 평가
        logger.info("Evaluating model...")
        train_metrics, test_metrics = evaluate_model(
            model,
            X_train,
            Y_train,
            X_test,
            Y_test,
            batch_size=config.model_params["batch_size"],
            output_dir=output_dir,
        )

        # 평가 결과 로깅
        logger.info("Final evaluation results:")
        logger.info(
            f"Training - Loss: {train_metrics[0]:.4f}, Accuracy: {train_metrics[1]:.4f}"
        )
        logger.info(
            f"Testing - Loss: {test_metrics[0]:.4f}, Accuracy: {test_metrics[1]:.4f}"
        )

        logger.info("Training process completed successfully")
        return 0

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1


# main.py 수정부분
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="basic",
        help="Configuration file to use from config directory",
    )
    args = parser.parse_args()

    config = Config(args.config)  # Config 객체 생성
    sys.exit(main(config))  # 수정: config.get_model_params() 대신 config 객체 전달
