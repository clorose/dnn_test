# path: ~/Develop/dnn_test/src/api/FastAPI_CNC_DNN.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib

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


# AI 모델 로드
load_model = tf.keras.models.load_model
model = load_model("CNC_DLL.h5")
scaler = joblib.load("scaler.pkl")


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


@app.get("/")
def read_root():
    return {"Hello": "World"}


# FastAPI 서버 실행
# uvicorn FastAPI_CNC_DNN:app --host 0.0.0.0 --reload
