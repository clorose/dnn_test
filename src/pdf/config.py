import os
import sys
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# 재귀 제한 증가(전체 프로그램에서 사용)
sys.setrecursionlimit(10000)

# 경로 설정
# Docker 환경
data_path = "/app/data"
root_path = "/app/"

# Local 환경 (주석 처리)
# data_path = "../../data"
# root_path = "../../"

# 가상 데이터 경로
virtual_data_path = os.path.join(data_path, "CNC Virtual Data set _v2")

# 실행 결과 저장 경로
run_dir = os.path.join(root_path, "runs")
os.makedirs(run_dir, exist_ok=True)

# 타임스탬프 및 출력 디렉토리
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
output_dir = os.path.join(run_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

# 모델 학습 관련 하이퍼파라미터
NB_CLASSES = 2
BATCH_SIZE = 1024
EPOCHS = 300
LEARNING_RATE = 1e-4
OPTIMIZER = Adam(LEARNING_RATE)

# 모델 체크포인트 설정
MODEL_CHECKPOINT_PATH = "weight_CNC_binary.mat"

# 데이터 분할 관련 설정
VALIDATION_SPLIT = 0.1
