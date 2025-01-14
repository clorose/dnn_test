# path: ~/Develop/dnn_test/src/config.py
# === Imports ===
import os
import sys
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === 시스템 설정 ===
# 재귀 제한 증가
sys.setrecursionlimit(10000)

# GPU 설정
if tf.config.list_physical_devices("GPU"):
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)

# 재현성을 위한 시드 설정
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

# === 경로 설정 ===
# 환경 감지 및 경로 설정
IS_DOCKER = os.path.exists("/.dockerenv")
data_path = "/app/data" if IS_DOCKER else "../data"
root_path = "/app/" if IS_DOCKER else "../"

# 가상 데이터 경로
virtual_data_path = os.path.join(data_path, "CNC Virtual Data set _v2")

# 실행 결과 저장 경로
run_dir = os.path.join(root_path, "runs")
os.makedirs(run_dir, exist_ok=True)

# 타임스탬프 및 출력 디렉토리
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
output_dir = os.path.join(run_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

# === 모델 설정 ===
# 기본 정규화 기법
USE_BATCH_NORM = True  # 활성화 (과적합 방지와 학습 안정화를 위해)
USE_DROPOUT = True  # 유지 (현재도 좋은 결과를 보임)
USE_L2_REG = True  # 활성화 (추가적인 정규화 효과를 위해)

# 학습 최적화 기법
USE_EARLY_STOPPING = True  # 유지 (현재도 잘 작동)
USE_LR_SCHEDULER = True  # 활성화 (학습률 동적 조정을 위해)
USE_HE_INIT = True  # 활성화 (ReLU 활성화 함수와 잘 맞음)

# 추가 기법
USE_LABEL_SMOOTHING = True  # 활성화 (과적합 방지에 도움)
USE_CROSS_VAL = False  # 비활성화 유지 (현재 검증 세트로 충분)

# 모델 학습 관련 하이퍼파라미터
NB_CLASSES = 2
BATCH_SIZE = 1024
EPOCHS = 300
LEARNING_RATE = 1e-5
OPTIMIZER = Adam(LEARNING_RATE)
VALIDATION_SPLIT = 0.1
EVAL_BATCH_SIZE = 32

# 모델 구조 관련 설정
DROPOUT_RATES = [0.1, 0.2, 0.3] if USE_DROPOUT else [0, 0, 0]
HIDDEN_UNITS = [128, 256, 512]
KERNEL_INITIALIZER = "he_normal" if USE_HE_INIT else "glorot_uniform"
ACTIVATION = "relu"

# 정규화 관련 설정
L2_LAMBDA = 0.01
LABEL_SMOOTHING = 0.1

# 모델 체크포인트 설정
MODEL_CHECKPOINT_PATH = os.path.join(output_dir, "weight_CNC_binary.mat")
MODEL_CHECKPOINT_MONITOR = "val_accuracy"
MODEL_CHECKPOINT_SAVE_BEST = True

# Early Stopping 설정
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MONITOR = "val_accuracy"
EARLY_STOPPING_MODE = "max"
EARLY_STOPPING_RESTORE_BEST = True
EARLY_STOPPING_MIN_DELTA = 0.001  # 0.1% 미만의 개선은 개선으로 보지 않음

# Learning Rate 설정
LR_DECAY_STEPS = [100, 150]  # epoch 기준
LR_DECAY_RATES = [0.5, 0.1]  # 각 step에서의 감소율

# Cross Validation 설정
N_SPLITS = 5

# === 데이터 전처리 설정 ===
# 데이터 레이블 매핑
TOOL_CONDITION_MAP = {"unworn": 0, "worn": 1}

ITEM_INSPECTION_MAP = {("no", None): 2, ("yes", "no"): 1, ("yes", "yes"): 0}

MACHINING_PROCESS_MAP = {
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


# === 유틸리티 함수 ===
def save_training_results(model, history, output_dir):
    # 그래프 저장
    plt.figure()
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["accuracy"])
    plt.title("Accuracy During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Validation Accuracy", "Training Accuracy"])
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(history.history["val_loss"])
    plt.plot(history.history["loss"])
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Validation Loss", "Training Loss"])
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # 모델 저장
    saved_model_path = os.path.join(output_dir, "saved_model")
    model.save(saved_model_path, save_format="tf")

    print(f"All results saved in: {output_dir}")
    print(f"Saved model in: {saved_model_path}")


def log_experiment_config(output_dir):
    config_file = os.path.join(output_dir, "experiment_config.txt")
    with open(config_file, "w") as f:
        f.write("실험 설정:\n")
        # 기본 플래그 상태 기록
        f.write(f"USE_EARLY_STOPPING: {USE_EARLY_STOPPING}\n")
        if USE_EARLY_STOPPING:
            f.write(f"  - patience: {EARLY_STOPPING_PATIENCE}\n")
            f.write(f"  - monitor: {EARLY_STOPPING_MONITOR}\n")
            f.write(f"  - mode: {EARLY_STOPPING_MODE}\n")
            f.write(f"  - restore_best_weights: {EARLY_STOPPING_RESTORE_BEST}\n")
            f.write(f"  - min_delta: {EARLY_STOPPING_MIN_DELTA}\n")

        f.write(f"USE_BATCH_NORM: {USE_BATCH_NORM}\n")

        f.write(f"USE_L2_REG: {USE_L2_REG}\n")
        if USE_L2_REG:
            f.write(f"  - L2_LAMBDA: {L2_LAMBDA}\n")

        f.write(f"USE_LR_SCHEDULER: {USE_LR_SCHEDULER}\n")
        if USE_LR_SCHEDULER:
            f.write(f"  - decay_steps: {LR_DECAY_STEPS}\n")
            f.write(f"  - decay_rates: {LR_DECAY_RATES}\n")

        f.write(f"USE_LABEL_SMOOTHING: {USE_LABEL_SMOOTHING}\n")
        if USE_LABEL_SMOOTHING:
            f.write(f"  - smoothing_factor: {LABEL_SMOOTHING}\n")

        f.write(f"USE_DROPOUT: {USE_DROPOUT}\n")
        if USE_DROPOUT:
            f.write(f"  - dropout_rates: {DROPOUT_RATES}\n")

        f.write(f"USE_HE_INIT: {USE_HE_INIT}\n")

        # 공통 모델 설정 기록
        f.write("\n모델 기본 설정:\n")
        f.write(f"HIDDEN_UNITS: {HIDDEN_UNITS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"ACTIVATION: {ACTIVATION}\n")
        f.write(f"OPTIMIZER: {type(OPTIMIZER).__name__}\n")
        f.write(f"VALIDATION_SPLIT: {VALIDATION_SPLIT}\n")
