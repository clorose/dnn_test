# path: D:\DNN_test\src\prepare_dataset.py
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

# prepare_dataset.py
from data_processor import tool_condition, item_inspection, machining_process


def prepare_training_data(data_path, virtual_data_path):
    """학습용 데이터셋 준비"""
    # Load train.csv
    train_sample = pd.read_csv(
        os.path.join(data_path, "train.csv"), header=0, encoding="utf-8"
    )
    train_sample_np = np.array(train_sample.copy())

    # Get All experiment files
    all_files = glob.glob(os.path.join(virtual_data_path, "experiment_*.csv"))
    li_df = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li_df.append(df)

    # Process training data
    train_sample_info = process_train_sample(train_sample_np)

    # Split data by type
    data_split = split_data_by_type(all_files, train_sample_info)
    data_pass, data_pass_half, data_fail = process_split_data(data_split)

    # Prepare final datasets
    X_train, X_test, Y_train, Y_test = create_final_datasets(
        data_pass, data_pass_half, data_fail
    )

    return X_train, X_test, Y_train, Y_test


def process_train_sample(train_sample_np):
    """Train sample 데이터 처리"""
    train_sample_info = train_sample_np.copy()

    # Apply process functions (이 함수들은 기존 data_processor.py에서 가져와야 함)
    train_sample_info = tool_condition(train_sample_info)
    train_sample_info = item_inspection(train_sample_info)

    # Remove unnecessary columns
    train_sample_info = np.delete(train_sample_info, 5, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)

    return train_sample_info


def split_data_by_type(all_files, train_sample_info):
    """데이터 타입별 분리"""
    li_pass = []
    li_pass_half = []
    li_fail = []

    for k, filename in enumerate(all_files):
        df = pd.read_csv(filename, index_col=None, header=0)
        label = train_sample_info[k, 3]

        if label == 0:
            li_pass.append(df)
        elif label == 1:
            li_pass_half.append(df)
        else:
            li_fail.append(df)

    return {"pass": li_pass, "pass_half": li_pass_half, "fail": li_fail}


def process_split_data(data_split):
    """분리된 데이터 처리"""
    frame01 = pd.concat(data_split["pass"], axis=0, ignore_index=True)
    frame02 = pd.concat(data_split["pass_half"], axis=0, ignore_index=True)
    frame03 = pd.concat(data_split["fail"], axis=0, ignore_index=True)

    data_pass = machining_process(np.array(frame01.copy()))
    data_pass_half = machining_process(np.array(frame02.copy()))
    data_fail = machining_process(np.array(frame03.copy()))

    return data_pass, data_pass_half, data_fail


def create_final_datasets(data_pass, data_pass_half, data_fail):
    """최종 데이터셋 생성"""
    # Combine data
    data01 = data_pass[0 : 3228 + 6175, :]
    data02 = data_pass_half[0:6175, :]
    data03 = data_fail[0:3228, :]

    data = np.concatenate((data01, data02), axis=0)
    data = np.concatenate((data, data03), axis=0)
    data_all = data_pass[3228 + 6175 : 22645, :]

    # Scale data
    sc = MinMaxScaler()
    X_train = sc.fit_transform(data)
    X_test = sc.transform(data_all)

    # Prepare labels
    Y_train = np.zeros((len(X_train), 1), dtype="int")
    Y_test = np.zeros((len(X_test), 1), dtype="int")

    # Set labels
    half = int(Y_train.shape[0] / 2)
    Y_train[0:half, :] = 0
    Y_train[half : half * 2, :] = 1

    # Convert to categorical
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    Y_train = to_categorical(Y_train, 2)  # NB_CLASSES = 2
    Y_test = to_categorical(Y_test, 2)

    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    print("Dataset preparation module loaded successfully")
