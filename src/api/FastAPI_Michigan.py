# path: ~/Develop/dnn_test/src/api/FastAPI_Michigan.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import os
import numpy as np
import joblib

from src.config import root_path, output_dir

# FastAPI 애플리케이션 생성
app = FastAPI()

# CORS 문제 대응
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 시 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Docker 환경의 경로를 사용하여 AI 모델 로드
# MODEL_PATH = os.path.join(root_path, "runs", "0115_1034", "saved_model")
# SCALER_PATH = os.path.join(root_path, "runs", "0115_1034", "minmax_scaler.joblib")
# /mnt/d/DNN_test/src/api/saved_model
MODEL_PATH = os.path.join(root_path, "src", "api", "saved_model")
SCALER_PATH = os.path.join(root_path, "src", "api", "minmax_scaler.joblib") 

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# 입력 데이터 형식을 위한 Pydantic 클래스 정의
class InputData(BaseModel):
    features: list


# 예측을 위한 엔드포인트
@app.post("/predict")
def predict(input_data: InputData):
    try:
        print("-------- Debugging --------")
        print(f"Type of input_data: {type(input_data)}")
        print(f"Raw input_data: {input_data}")
        print(f"Features: {input_data.features}")
        print("--------------------------")

        # 입력 데이터를 numpy 배열로 변환
        input_array = np.array([input_data.features])
        print(f"Input array shape: {input_array.shape}")

        # 예측 수행
        ns_data = scaler.transform(input_array)
        print(f"Normalized data shape: {ns_data.shape}")

        prediction = model.predict(ns_data).tolist()
        print(f"Prediction: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=490, detail=f"Error: {str(e)}")


# 루트 엔드포인트 - 모델 경로 확인용
@app.get("/")
def read_root():
    return {"model_path": MODEL_PATH, "scaler_path": SCALER_PATH}


# FastAPI 서버 실행 명령어
# uvicorn FastAPI_Michigan:app --host 0.0.0.0 --reload
