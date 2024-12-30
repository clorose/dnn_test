# path: ~/Develop/dnn_test/src/data_processing.py
import os
import glob
import numpy as np
import pandas as pd


def tool_condition(input_data):
    """공구 상태 전처리"""
    for i in range(len(input_data)):
        if input_data[i, 4] == "unworn":
            input_data[i, 4] = 0
        else:
            input_data[i, 4] = 1
    return input_data


def item_inspection(input_data):
    """검사 결과 전처리"""
    for i in range(len(input_data)):
        if input_data[i, 5] == "no":
            input_data[i, 6] = 2
        elif input_data[i, 5] == "yes" and input_data[i, 6] == "no":
            input_data[i, 6] = 1
        elif input_data[i, 5] == "yes" and input_data[i, 6] == "yes":
            input_data[i, 6] = 0
    return input_data


def load_data(data_path):
    """데이터 로드"""
    # 모든 열의 dtype을 미리 정의
    dtype_dict = {
        col: np.float64 for col in range(48)
    }  # 기본적으로 모든 열을 float64로
    dtype_dict[47] = str  # Machining_Process 열만 문자열로

    # train.csv 파일 로드
    train_sample = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_sample_np = np.array(train_sample.copy())

    # 라벨 데이터 전처리
    train_sample_info = tool_condition(train_sample_np)
    train_sample_info = item_inspection(train_sample_info)
    train_sample_info = np.delete(train_sample_info, 5, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)
    train_sample_info = train_sample_info.astype(np.float64)  # 명시적 타입 변환

    # experiment 파일들 로드
    all_files = glob.glob(
        os.path.join(data_path, "CNC Virtual Data set _v2", "experiment_*.csv")
    )

    li_pass = []
    li_pass_half = []
    li_fail = []

    for k, filename in enumerate(all_files):
        df = pd.read_csv(filename, dtype=dtype_dict)
        if train_sample_info[k, 3] == 0:
            li_pass.append(df)
        elif train_sample_info[k, 3] == 1:
            li_pass_half.append(df)
        else:
            li_fail.append(df)

    # 데이터 병합
    frame01 = pd.concat(li_pass, axis=0, ignore_index=True)
    frame02 = pd.concat(li_pass_half, axis=0, ignore_index=True)
    frame03 = pd.concat(li_fail, axis=0, ignore_index=True)

    print("Data types before conversion:")
    print(frame01.dtypes)

    # numpy 변환 전 데이터 타입 확인
    for col in frame01.columns:
        if col != "Machining_Process" and frame01[col].dtype != np.float64:
            print(f"Converting {col} to float64")
            frame01[col] = frame01[col].astype(np.float64)
            frame02[col] = frame02[col].astype(np.float64)
            frame03[col] = frame03[col].astype(np.float64)

    # numpy 변환
    data_pass = np.array(frame01.copy())
    data_pass_half = np.array(frame02.copy())
    data_fail = np.array(frame03.copy())

    # machining process 전처리
    data_pass = machining_process(data_pass)
    data_pass_half = machining_process(data_pass_half)
    data_fail = machining_process(data_fail)

    # 최종 데이터가 모두 float64인지 확인
    data_pass = data_pass.astype(np.float64)
    data_pass_half = data_pass_half.astype(np.float64)
    data_fail = data_fail.astype(np.float64)

    print("\n데이터 크기:")
    print(f"양품 데이터: {data_pass.shape}")
    print(f"불합격 데이터: {data_pass_half.shape}")
    print(f"불량품 데이터: {data_fail.shape}")

    print(f"\n데이터 타입:")
    print(f"양품 데이터: {data_pass.dtype}")
    print(f"불합격 데이터: {data_pass_half.dtype}")
    print(f"불량품 데이터: {data_fail.dtype}")

    return train_sample_info, data_pass, data_pass_half, data_fail


def machining_process(input_data):
    """가공 공정 전처리"""
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

    # 47번 열(Machining_Process)만 따로 저장
    process_column = input_data[:, 47].copy()

    # 나머지 열들을 float으로 변환
    numeric_data = input_data[:, :47].astype(np.float64)

    # Machining_Process 열 변환
    process_values = np.array(
        [float(process_map.get(str(val), 0)) for val in process_column],
        dtype=np.float64,
    )

    # 모든 열 합치기
    final_data = np.column_stack((numeric_data, process_values))

    # 최종 확인
    if not np.issubdtype(final_data.dtype, np.floating):
        final_data = final_data.astype(np.float64)

    return final_data
