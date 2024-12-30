# path: ~/Develop/dnn_test/src/first/data_preprocessor.py
# 표준 라이브러리
import logging
from typing import List, Tuple, Dict

# 써드파티 라이브러리
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, logger: logging.Logger = None):
        self.scaler = MinMaxScaler()
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def tool_condition(input_data: np.ndarray) -> np.ndarray:
        """공구 상태를 수치화"""
        data = input_data.copy()
        data[:, 4] = np.where(data[:, 4] == "unworn", 0, 1)
        return data

    @staticmethod
    def item_inspection(input_data: np.ndarray) -> np.ndarray:
        """검사 결과를 수치화"""
        data = input_data.copy()
        conditions = [
            (data[:, 5] == "no"),
            (data[:, 5] == "yes") & (data[:, 6] == "no"),
            (data[:, 5] == "yes") & (data[:, 6] == "yes"),
        ]
        values = [2, 1, 0]
        data[:, 6] = np.select(conditions, values)
        return data

    @staticmethod
    def machining_process(input_data: np.ndarray) -> np.ndarray:
        """가공 공정을 수치화"""
        process_map = {
            "Prep": 0,
            "Layer 1 Up": 1,
            "Layer 1 Down": 2,
            "Layer 2 Up": 3,
            "Layer 2 Down": 4,
            "Layer 3 Up": 5,
            "Layer 3 Down": 6,
            "Repositioning": 7,
            "End": 8,
            "end": 8,
            "Starting": 9,
        }
        data = input_data.copy()
        data[:, 47] = np.array([process_map.get(x, x) for x in data[:, 47]])
        return data

    def preprocess_train_sample(
        self, train_sample: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict]:
        """학습 샘플 전처리"""
        train_sample_np = train_sample.to_numpy()

        # 기본 통계 수집
        stats = self._collect_statistics(train_sample_np)

        # 데이터 전처리
        train_info = self.tool_condition(train_sample_np)
        train_info = self.item_inspection(train_info)

        # 불필요한 열 제거
        train_info = np.delete(train_info, [0, 1, 5], axis=1)

        self.logger.info("Train sample preprocessing completed")
        return train_info, stats

    def preprocess_experiment_data(
        self, data_frames: List[pd.DataFrame], labels: np.ndarray
    ) -> Tuple[List[np.ndarray], Dict]:
        """실험 데이터 전처리"""
        # 라벨에 따라 데이터 분류
        categorized_data = {0: [], 1: [], 2: []}

        for df, label in zip(data_frames, labels[:, 3]):
            categorized_data[label].append(df)

        processed_data = {}
        for label, dfs in categorized_data.items():
            if dfs:
                combined = pd.concat(dfs, axis=0, ignore_index=True)
                processed_data[label] = self.machining_process(combined.to_numpy())

        self.logger.info("Experiment data preprocessing completed")
        return processed_data

    def prepare_train_test_data(
        self, processed_data: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """학습 및 테스트 데이터 준비"""
        # 데이터 결합
        data01 = processed_data[0][0 : 3228 + 6175, :]
        data02 = processed_data[1][0:6175, :]
        data03 = processed_data[2][0:3228, :]

        train_data = np.concatenate((data01, data02, data03), axis=0)
        test_data = processed_data[0][3228 + 6175 : 22645, :]

        # 스케일링
        X_train = self.scaler.fit_transform(train_data)
        X_test = self.scaler.transform(test_data)

        # 레이블 생성
        Y_train = np.zeros((len(X_train), 1), dtype="int")
        Y_test = np.zeros((len(X_test), 1), dtype="int")

        # 레이블 설정
        half_train = len(Y_train) // 2
        Y_train[half_train:] = 1

        half_test = len(Y_test) // 2
        Y_test[half_test:] = 1

        return X_train, X_test, Y_train, Y_test

    def _collect_statistics(self, data: np.ndarray) -> Dict:
        """데이터 통계 수집"""
        stats = {
            "nb_pass": np.sum((data[:, 5] == "yes") & (data[:, 6] == "yes")),
            "nb_pass_half": np.sum((data[:, 5] == "yes") & (data[:, 6] == "no")),
            "nb_defective": np.sum(data[:, 5] == "no"),
        }
        stats["total"] = sum(stats.values())
        return stats
