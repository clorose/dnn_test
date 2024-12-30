# path: ~/Develop/dnn_test/src/first/config.py
import os
import yaml
from pathlib import Path


class Config:
    def __init__(self, config_name: str):
        # 데이터 경로
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        self.root_path = os.getenv("ROOT_PATH", "/app")
        self.virtual_data_path = os.path.join(
            self.data_path, "CNC Virtual Data set _v2"
        )
        self.run_data_path = os.path.join(self.root_path, "runs")

        # YAML 설정 로드
        self.config_dir = Path(__file__).parent / "config"
        self.config_path = self.config_dir / f"{config_name}.yaml"
        self.config = self._load_config()

        # 설정값 로드 및 변환
        self.seed = self.config.get("training", {}).get("seed", 42)
        self.model_params = self.config.get("model_params", {})
        self.training = self.config.get("training", {})
        self._convert_numeric_values()

    def _convert_numeric_values(self):
        """숫자 값들을 적절한 타입으로 변환"""
        # model_params의 숫자 변환
        numeric_keys = ["learning_rate"]
        for key in numeric_keys:
            if key in self.model_params:
                self.model_params[key] = float(self.model_params[key])

        # training의 숫자 변환
        training_numeric_keys = ["dropout_rate", "lr_factor", "min_lr"]
        for key in training_numeric_keys:
            if key in self.training:
                self.training[key] = float(self.training[key])

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def train_file_path(self) -> str:
        return os.path.join(self.data_path, "train.csv")

    def get_model_params(self) -> dict:
        # 기존 인터페이스 유지하면서 YAML 설정 반환
        return {**self.model_params, **self.training}
