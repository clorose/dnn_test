import json
import yaml
import numpy as np
from config import ExperimentConfig
from sklearn.preprocessing import MinMaxScaler


class NumpyEncoder(json.JSONEncoder):
    """Numpy 타입과 ExperimentConfig를 JSON으로 직렬화하기 위한 인코더"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, ExperimentConfig):
            return obj.to_dict()
        return super(NumpyEncoder, self).default(obj)


def save_yaml(config_dict, filepath):
    """최적 파라미터를 yaml 파일로 저장"""
    with open(filepath, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def prepare_data(data_path, config):
    """데이터 전처리 함수"""
    from data_processing import load_data  # 데이터 로딩 관련 함수들을 별도 모듈로 분리

    train_sample_info, data_pass, data_pass_half, data_fail = load_data(data_path)

    # 이하 prepare_data 함수의 나머지 코드는 main.py에서 그대로 가져옴
    if config.classification_type == "binary":
        # PDF 방식의 이진분류 데이터셋 구성
        data01 = data_pass[0 : 3228 + 6175, :]
        data02 = data_pass_half[0:6175, :]
        data03 = data_fail[0:3228, :]

        data = np.concatenate((data01, data02), axis=0)
        data = np.concatenate((data, data03), axis=0)
        data_all = data_pass[3228 + 6175 : 22645, :]

        # 정규화
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(data)
        X_test = scaler.fit_transform(data_all)

        # 이진 라벨링
        y_train = np.zeros((len(X_train),), dtype="int")
        l = int(y_train.shape[0] / 2)
        y_train[l : l * 2] = 1
        y_test = np.zeros((len(X_test),), dtype="int")

    else:
        # 3클래스 분류용 데이터셋 구성
        min_samples = min(len(data_pass), len(data_pass_half), len(data_fail))

        data01 = data_pass[:min_samples, :]
        data02 = data_pass_half[:min_samples, :]
        data03 = data_fail[:min_samples, :]

        data = np.concatenate((data01, data02, data03), axis=0)
        remaining_data = data_pass[min_samples:, :]

        # 정규화
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(data)
        X_test = scaler.transform(remaining_data)

        # 3클래스 라벨링
        y_train = np.zeros((len(X_train),), dtype="int")
        n_per_class = min_samples
        y_train[n_per_class : 2 * n_per_class] = 1
        y_train[2 * n_per_class :] = 2
        y_test = np.zeros((len(X_test),), dtype="int")

    return X_train, X_test, y_train, y_test
