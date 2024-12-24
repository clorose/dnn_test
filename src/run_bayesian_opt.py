# path: ~/Develop/dnn_test/src/run_bayesian_opt.py
import argparse
import os
import json
from datetime import datetime
import yaml
import numpy as np  # numpy 추가

from config import ExperimentConfig
from experiment import ModelExperiment
from bayesian_opt import BayesianOptimizer
from utils import prepare_data, NumpyEncoder, save_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization")
    parser.add_argument(
        "--config",
        type=str,
        default="../experiments/base.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--data_path", type=str, default="../data", help="Path to data directory"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of optimization trials"
    )
    return parser.parse_args()


def convert_numpy_types(obj):
    """NumPy 타입을 기본 Python 타입으로 변환"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, ExperimentConfig):
        return convert_numpy_types(obj.to_dict())
    return obj


def save_yaml(config_dict, filepath):
    """최적 파라미터를 yaml 파일로 저장"""
    # NumPy 타입을 기본 Python 타입으로 변환
    converted_dict = convert_numpy_types(config_dict)

    with open(filepath, "w") as f:
        yaml.dump(converted_dict, f, default_flow_style=False)


# run_bayesian_opt.py
class NumpyEncoder(json.JSONEncoder):
    """Numpy 타입과 ExperimentConfig를 JSON으로 직렬화하기 위한 인코더"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, ExperimentConfig):  # ExperimentConfig 객체 처리 추가
            return obj.to_dict()
        return super(NumpyEncoder, self).default(obj)


def main():
    args = parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    X_train, X_test, y_train, y_test = prepare_data(args.data_path, config)
    experiment = ModelExperiment(X_train, X_test, y_train, y_test)

    # Bayesian Optimization 실행
    optimizer = BayesianOptimizer(experiment=experiment, base_config=config)
    results = optimizer.optimize(n_calls=args.n_trials)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "../experiments/optimal"
    os.makedirs(save_dir, exist_ok=True)

    # 최적 파라미터를 yaml 파일로 저장
    optimal_yaml = os.path.join(save_dir, f"optimal_params_{timestamp}.yaml")
    save_yaml(results["optimal_config"].to_dict(), optimal_yaml)

    # 전체 최적화 결과 저장
    results_json = os.path.join(save_dir, f"optimization_results_{timestamp}.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(f"\n최적화가 완료되었습니다.")
    print(f"최적 파라미터가 저장된 경로: {optimal_yaml}")
    print(f"전체 결과가 저장된 경로: {results_json}")
    print("\n최적 파라미터로 실험을 실행하려면:")
    print(f"python main.py --config {optimal_yaml}")


if __name__ == "__main__":
    main()
