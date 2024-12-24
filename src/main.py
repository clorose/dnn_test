# path: ~/Develop/dnn_test/src/main.py

import argparse
import os
from pathlib import Path
import json
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler

from config import ExperimentConfig
from experiment import ModelExperiment
from optimizers import ModelOptimizer
from bayesian_opt import BayesianOptimizer
from utils import NumpyEncoder, prepare_data


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

    for i in range(len(input_data)):
        input_data[i, 47] = process_map.get(input_data[i, 47], input_data[i, 47])

    return input_data


def load_data(data_path):
    """데이터 로드"""
    # train.csv 파일 로드
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

    # 각 실험 데이터 분류
    li_pass = []
    li_pass_half = []
    li_fail = []

    for k, filename in enumerate(all_files):
        df = pd.read_csv(filename)
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

    # numpy 변환
    data_pass = np.array(frame01.copy())
    data_pass_half = np.array(frame02.copy())
    data_fail = np.array(frame03.copy())

    # machining process 전처리
    data_pass = machining_process(data_pass)
    data_pass_half = machining_process(data_pass_half)
    data_fail = machining_process(data_fail)

    print(f"\n데이터 크기:")
    print(f"양품 데이터: {data_pass.shape}")
    print(f"불합격 데이터: {data_pass_half.shape}")
    print(f"불량품 데이터: {data_fail.shape}")

    return train_sample_info, data_pass, data_pass_half, data_fail


def run_normal_experiment(experiment, config):
    """일반 실험 실행"""
    optimizer = ModelOptimizer()

    for key, value in config.to_dict().items():
        if key in optimizer.current_config:
            optimizer.current_config[key] = value

    print(f"\n{config.tag} 실험 실행 중...")
    experiment.run_experiment(config.tag, optimizer)

    with open(os.path.join(experiment.result_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    print("\n실험이 완료되었습니다.")
    print(f"결과는 {experiment.result_dir} 디렉토리에 저장되었습니다.")


def run_bayesian_optimization(experiment, config, n_trials):
    """베이지안 최적화 실행"""
    print("\nBayesian Optimization 시작...")
    optimizer = BayesianOptimizer(experiment=experiment, base_config=config)
    results = optimizer.optimize(n_calls=n_trials)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimal_dir = os.path.join("../experiments/optimal")
    os.makedirs(optimal_dir, exist_ok=True)

    # 최적 파라미터를 yaml 파일로 저장
    optimal_yaml = os.path.join(optimal_dir, f"optimal_params_{timestamp}.yaml")
    with open(optimal_yaml, "w") as f:
        yaml.dump(results["optimal_config"].to_dict(), f, default_flow_style=False)

    # 전체 최적화 결과 저장
    results_json = os.path.join(optimal_dir, f"optimization_results_{timestamp}.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(f"\n최적화가 완료되었습니다.")
    print(f"최적 파라미터가 저장된 경로: {optimal_yaml}")
    print(f"전체 결과가 저장된 경로: {results_json}")

    return results["optimal_config"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DNN experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="../experiments/base.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Bayesian Optimization",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=40,
        help="Number of optimization trials",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    print("데이터 로드 및 전처리 중...")
    X_train, X_test, y_train, y_test = prepare_data(args.data_path, config)

    print("\n실험 객체 생성 중...")
    experiment = ModelExperiment(X_train, X_test, y_train, y_test)

    if args.optimize:
        config = run_bayesian_optimization(experiment, config, args.n_trials)

    run_normal_experiment(experiment, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류 발생: {str(e)}")
