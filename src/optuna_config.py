# path: D:\DNN_test\src\optuna_config.py
import os
import sys
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna

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

    INPUT_DIM = 48  # Michigan 데이터 입력 차원
    OUTPUT_DIM = 2  # 이진 분류

    # 네트워크 구조 옵션
    INITIAL_UNITS_RANGE = (64, 256)  # 첫 레이어 유닛 수 범위
    MIDDLE_UNITS_RANGE = (256, 1024)  # 중간 레이어 유닛 수 범위
    LAYER_RANGE = (2, 4)  # 레이어 수 범위

    # 활성화 함수 옵션
    ACTIVATIONS = ["relu", "elu"]
    OUTPUT_ACTIVATION = "softmax"

    # 초기화 옵션
    INITIALIZERS = ["he_normal", "glorot_uniform"]


class OptunaConfig:
    """Optuna 최적화 관련 설정"""

    # 기본 최적화 설정
    N_TRIALS = 100  # 최적화 시도 횟수
    N_EPOCHS = 150  # 각 시도별 학습 epoch
    VALIDATION_SPLIT = 0.2  # 검증 세트 비율

    # 학습 관련 범위
    BATCH_SIZES = [256, 512, 1024]
    LR_RANGE = (1e-5, 1e-3)

    # 정규화 관련 범위
    DROPOUT_RANGE = (0.2, 0.5)
    L2_LAMBDA_RANGE = (1e-5, 1e-2)

    # 옵티마이저 설정
    OPTIMIZERS = ["adam", "rmsprop"]
    ADAM_BETA1_RANGE = (0.9, 0.999)

    # Early Stopping 설정
    ES_PATIENCE_RANGE = (10, 30)
    ES_MONITORS = ["val_loss", "val_accuracy"]

    # Learning Rate Scheduler 설정
    SCHEDULER_TYPES = ["step", "exponential"]
    LR_DECAY_RATE_RANGE = (0.1, 0.5)
    STEP_DECAY_STEPS_RANGE = (20, 100)
    EXP_DECAY_RATE_RANGE = (0.9, 0.99)


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
    # 1. 기존 Best Trial 결과 저장
    results_file = os.path.join(output_dir, f"optuna_results_{study_name}.txt")
    best_trial = study.best_trial

    with open(results_file, "w") as f:
        f.write("Best trial results:\n\n")
        f.write(f"Value: {best_trial.value}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest trial metrics:\n")
        f.write(
            f"Best validation accuracy: {best_trial.user_attrs['val_accuracy']:.4f}\n"
        )
        f.write(
            f"Final training accuracy: {best_trial.user_attrs['train_accuracy']:.4f}\n"
        )
        f.write(f"Test accuracy: {best_trial.user_attrs['test_accuracy']:.4f}\n")

    # 2. 추가 요약 정보 저장
    summary_file = os.path.join(output_dir, f"optimization_summary_{study_name}.txt")
    with open(summary_file, "w") as f:
        # 기본 정보
        f.write("=== Optimization Summary ===\n")
        start_time = study.trials[0].datetime_start
        end_time = study.trials[-1].datetime_complete
        duration = end_time - start_time

        f.write("[Basic Info]\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {duration}\n")

        # Trial 통계
        completed_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        failed_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        )
        pruned_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        )

        f.write("\n[Trial Statistics]\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Completed: {completed_trials}\n")
        f.write(f"Failed: {failed_trials}\n")
        f.write(f"Pruned: {pruned_trials}\n")
        f.write(f"Average trial time: {duration / len(study.trials)}\n")

        # Top 5 trials
        f.write("\n[Top 5 Trials]\n")
        sorted_trials = sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else float("-inf"),
            reverse=True,
        )
        for i, trial in enumerate(sorted_trials[:5], 1):
            f.write(f"{i}. Trial {trial.number}: {trial.value:.4f} (test_acc)\n")
            f.write(f"   Val Acc: {trial.user_attrs.get('val_accuracy', 'N/A'):.4f}\n")
            f.write(
                f"   Train Acc: {trial.user_attrs.get('train_accuracy', 'N/A'):.4f}\n"
            )

        # 파라미터 중요도 (가능한 경우)
        try:
            f.write("\n[Parameter Importance]\n")
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                f.write(f"{param}: {importance:.4f}\n")
        except:
            f.write("Parameter importance analysis not available\n")

    # 3. 시각화 (기존 코드)
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
        f.write("=== 실험 설정 ===\n\n")

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
