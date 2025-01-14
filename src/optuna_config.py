# path: D:\DNN_test\src\optuna_config.py
import os
import sys
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

# 시스템 설정
sys.setrecursionlimit(10000)

# GPU 설정
if tf.config.list_physical_devices("GPU"):
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

# 재현성을 위한 시드 설정
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

# 경로 설정
IS_DOCKER = os.path.exists("/.dockerenv")
BASE_PATH = "/app" if IS_DOCKER else ".."
DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "optimization_results")

# 결과 저장 디렉토리 생성
os.makedirs(RESULTS_PATH, exist_ok=True)


class ModelConfig:
    """모델 구조 관련 설정"""

    INPUT_DIM = 48
    OUTPUT_DIM = 2
    ACTIVATION = "relu"
    OUTPUT_ACTIVATION = "softmax"

    # 레이어 유닛 옵션
    INITIAL_UNITS = [64, 128, 256]
    MIDDLE_UNITS = [256, 512]


class OptunaConfig:
    """Optuna 최적화 관련 설정"""

    # 기본 최적화 설정
    N_TRIALS = 100
    N_EPOCHS = 150
    VALIDATION_SPLIT = 0.3

    # 하이퍼파라미터 범위
    BATCH_SIZES = [128, 256, 512]
    LR_RANGE = (1e-5, 1e-3)
    LAYER_RANGE = (1, 3)
    DROPOUT_RANGE = (0.3, 0.6)
    L2_LAMBDA_RANGE = (1e-4, 5e-2)

    # Early Stopping 설정
    ES_PATIENCE_RANGE = (5, 15)
    ES_MONITORS = ["val_loss", "val_accuracy"]

    # Learning Rate Decay 설정
    DECAY_STEPS_RANGE = (20, 80)
    LR_DECAY_RATE_RANGE = (0.1, 0.5)


def get_timestamp():
    """현재 시간 기반 타임스탬프 반환"""
    return datetime.datetime.now().strftime("%m%d_%H%M")


def create_experiment_dir():
    """실험 결과 저장을 위한 디렉토리 생성"""
    timestamp = get_timestamp()
    experiment_dir = os.path.join(RESULTS_PATH, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_training_results(study, output_dir, study_name):
    """학습 결과 저장"""
    results_file = os.path.join(output_dir, f"optuna_results_{study_name}.txt")
    trial = study.best_trial

    # 결과 파일 작성
    with open(results_file, "w") as f:
        f.write("Best trial results:\n\n")
        f.write(f"Value: {trial.value}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest trial metrics:\n")
        f.write(f"Best validation accuracy: {trial.user_attrs['val_accuracy']:.4f}\n")
        f.write(f"Final training accuracy: {trial.user_attrs['train_accuracy']:.4f}\n")
        f.write(f"Test accuracy: {trial.user_attrs['test_accuracy']:.4f}\n")

    # 학습 진행 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot([t.number for t in study.trials], [t.value for t in study.trials])
    plt.xlabel("Trial number")
    plt.ylabel("Accuracy")
    plt.title("Optimization Progress")
    plt.savefig(os.path.join(output_dir, "optimization_progress.png"))
    plt.close()


def log_experiment_config(output_dir):
    """실험 설정 로깅"""
    config_file = os.path.join(output_dir, "experiment_config.txt")

    with open(config_file, "w") as f:
        f.write("실험 설정\n\n")

        f.write("모델 구조 설정:\n")
        for attr in dir(ModelConfig):
            if not attr.startswith("__"):
                f.write(f"{attr}: {getattr(ModelConfig, attr)}\n")

        f.write("\n최적화 설정:\n")
        for attr in dir(OptunaConfig):
            if not attr.startswith("__"):
                f.write(f"{attr}: {getattr(OptunaConfig, attr)}\n")


def handle_error(error, output_dir):
    """에러 처리 및 로깅"""
    error_log_path = os.path.join(output_dir, "error_log.txt")
    with open(error_log_path, "w") as f:
        f.write(f"Error occurred at {datetime.datetime.now()}: {str(error)}\n")
    print(f"\nError occurred: {str(error)}")
    raise error
