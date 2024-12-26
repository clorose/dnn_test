import json
import yaml
import numpy as np
from config import ExperimentConfig
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
    from data_processing import load_data

    train_sample_info, data_pass, data_pass_half, data_fail = load_data(data_path)

    # 데이터 타입 변환
    data_pass = data_pass.astype(np.float32)
    data_pass_half = data_pass_half.astype(np.float32)
    data_fail = data_fail.astype(np.float32)

    if config.classification_type == "binary":
        # 양품/불량품 데이터 구성
        good_data = data_pass
        bad_data = np.concatenate([data_pass_half, data_fail], axis=0)

        # 데이터 분할 (80% 훈련, 20% 테스트)
        good_train, good_test = train_test_split(
            good_data, test_size=0.2, random_state=42
        )
        bad_train, bad_test = train_test_split(bad_data, test_size=0.2, random_state=42)

        # 훈련/테스트 데이터 구성
        X_train = np.concatenate([good_train, bad_train], axis=0)
        X_test = np.concatenate([good_test, bad_test], axis=0)

        # 라벨링
        y_train = np.concatenate(
            [
                np.zeros(len(good_train), dtype=np.float32),
                np.ones(len(bad_train), dtype=np.float32),
            ]
        )
        y_test = np.concatenate(
            [
                np.zeros(len(good_test), dtype=np.float32),
                np.ones(len(bad_test), dtype=np.float32),
            ]
        )

    else:
        # 3클래스 분류용 데이터셋 구성
        X_list = []
        y_list = []
        X_test_list = []
        y_test_list = []

        for i, data in enumerate([data_pass, data_pass_half, data_fail]):
            X_train_split, X_test_split = train_test_split(
                data, test_size=0.2, random_state=42
            )
            X_list.append(X_train_split)
            X_test_list.append(X_test_split)
            y_list.append(np.full(len(X_train_split), i, dtype=np.int32))
            y_test_list.append(np.full(len(X_test_split), i, dtype=np.int32))

        X_train = np.concatenate(X_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_list)
        y_test = np.concatenate(y_test_list)

    # 데이터 정규화
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 데이터 섞기
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx]
    y_test = y_test[shuffle_idx]

    # 데이터 분포 출력
    print("\n데이터 분포:")
    print("훈련 데이터:")
    print(f"클래스 0 (양품): {np.sum(y_train == 0)}")
    if config.classification_type == "binary":
        print(f"클래스 1 (불량): {np.sum(y_train == 1)}")
    else:
        print(f"클래스 1 (불합격): {np.sum(y_train == 1)}")
        print(f"클래스 2 (불량): {np.sum(y_train == 2)}")

    print("\n테스트 데이터:")
    print(f"클래스 0 (양품): {np.sum(y_test == 0)}")
    if config.classification_type == "binary":
        print(f"클래스 1 (불량): {np.sum(y_test == 1)}")
    else:
        print(f"클래스 1 (불합격): {np.sum(y_test == 1)}")
        print(f"클래스 2 (불량): {np.sum(y_test == 2)}")

    # 최종 확인
    print("\n최종 데이터 타입 확인:")
    print(f"X_train dtype: {X_train.dtype}")
    print(f"X_test dtype: {X_test.dtype}")
    print(f"y_train dtype: {y_train.dtype}")
    print(f"y_test dtype: {y_test.dtype}")

    return X_train, X_test, y_train, y_test
