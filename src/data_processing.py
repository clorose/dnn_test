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
    numeric_data = input_data[:, :47].astype(float)

    # Machining_Process 열 변환
    process_values = np.array(
        [float(process_map.get(str(val), 0)) for val in process_column]
    )

    # 모든 열 합치기
    final_data = np.column_stack((numeric_data, process_values))

    return final_data


def load_data(data_path):
    """데이터 로드"""
    # train.csv 파일 로드 시 numeric_only=True 옵션 추가
    train_sample = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_sample_np = np.array(train_sample.copy())

    # 라벨 데이터 전처리
    train_sample_info = tool_condition(train_sample_np)
    train_sample_info = item_inspection(train_sample_info)
    train_sample_info = np.delete(train_sample_info, 5, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)

    # experiment 파일들 로드
    all_files = glob.glob(
        os.path.join(data_path, "CNC Virtual Data set _v2", "experiment_*.csv")
    )

    li_pass = []
    li_pass_half = []
    li_fail = []

    for k, filename in enumerate(all_files):
        # 숫자 데이터만 로드하도록 수정
        df = pd.read_csv(
            filename, dtype={47: str}
        )  # 47번째 열만 문자열로 읽고 나머지는 자동으로 숫자로
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

    # numpy 변환 전에 데이터 타입 출력하여 확인
    print("Data types before conversion:")
    print(frame01.dtypes)

    # numpy 변환
    data_pass = np.array(frame01.copy())
    data_pass_half = np.array(frame02.copy())
    data_fail = np.array(frame03.copy())

    # machining process 전처리 (이 과정에서 모든 데이터가 float로 변환됨)
    data_pass = machining_process(data_pass)
    data_pass_half = machining_process(data_pass_half)
    data_fail = machining_process(data_fail)

    print("\n데이터 크기:")
    print(f"양품 데이터: {data_pass.shape}")
    print(f"불합격 데이터: {data_pass_half.shape}")
    print(f"불량품 데이터: {data_fail.shape}")

    # 데이터 타입 확인
    print(f"\n데이터 타입:")
    print(f"양품 데이터: {data_pass.dtype}")
    print(f"불합격 데이터: {data_pass_half.dtype}")
    print(f"불량품 데이터: {data_fail.dtype}")

    return train_sample_info, data_pass, data_pass_half, data_fail
