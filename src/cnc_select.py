import pandas as pd
import numpy as np
import glob
import os
from config import *


def analyze_train_csv(csv_path):
    """학습 데이터의 기본 통계를 분석합니다."""
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)

    # 기본 카테고리 카운트
    nb_pass = len(
        df[
            (df["machining_finalized"] == "yes")
            & (df["passed_visual_inspection"] == "yes")
        ]
    )
    nb_pass_half = len(
        df[
            (df["machining_finalized"] == "yes")
            & (df["passed_visual_inspection"] == "no")
        ]
    )
    nb_fail = len(df[df["machining_finalized"] == "no"])

    print("\n=== 기본 데이터 분석 ===")
    print(f"Pass 샘플 수 (공정완료 + 육안검사 통과): {nb_pass}")
    print(f"Pass-half 샘플 수 (공정완료 + 육안검사 실패): {nb_pass_half}")
    print(f"Fail 샘플 수 (공정 미완료): {nb_fail}")
    print(f"전체 샘플 수: {nb_pass + nb_pass_half + nb_fail}")

    return df, nb_pass, nb_pass_half, nb_fail


def analyze_experiment_files(virtual_data_path, df_train):
    """실험 데이터 파일들의 통계를 분석합니다."""
    # 모든 실험 파일 가져오기
    all_files = glob.glob(os.path.join(virtual_data_path, "experiment_*.csv"))

    print("\n=== 실험 파일 분석 ===")
    print(f"총 실험 파일 수: {len(all_files)}")

    # 각 실험 파일의 데이터 수 확인
    data_counts = []
    for filename in all_files:
        df = pd.read_csv(filename)
        data_counts.append(len(df))

    print(f"파일당 평균 데이터 수: {np.mean(data_counts):.2f}")
    print(f"파일당 최소 데이터 수: {np.min(data_counts)}")
    print(f"파일당 최대 데이터 수: {np.max(data_counts)}")
    print(f"전체 데이터 포인트 수: {sum(data_counts)}")

    return all_files, data_counts


def analyze_data_distribution(df_train, all_files):
    """데이터의 분포를 분석합니다."""
    print("\n=== 데이터 분포 분석 ===")

    # 주요 특성들의 통계 출력
    numeric_columns = df_train.select_dtypes(include=[np.number]).columns
    print("\n주요 수치형 특성 통계:")
    print(df_train[numeric_columns].describe())

    # 범주형 데이터 분포
    categorical_columns = df_train.select_dtypes(include=["object"]).columns
    print("\n범주형 데이터 분포:")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df_train[col].value_counts())


if __name__ == "__main__":
    # 데이터 경로 설정
    virtual_data_path = os.path.join(data_path, "CNC_SMART_MICHIGAN")
    train_csv_path = os.path.join(virtual_data_path, "train.csv")

    # 분석 실행
    df_train, nb_pass, nb_pass_half, nb_fail = analyze_train_csv(train_csv_path)
    all_files, data_counts = analyze_experiment_files(virtual_data_path, df_train)
    analyze_data_distribution(df_train, all_files)
