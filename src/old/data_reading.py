# path: ~/Develop/dnn_test/src/old/data_reading.py
from __future__ import print_function

import os, sys, math, copy
import glob
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.setrecursionlimit(10000)
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


from src.function_dataPreprocess import (
    tool_condition,
    item_inspection,
    machining_process,
)

# 결과 저장 디렉토리 생성
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = "result_" + current_time  # 현재 디렉토리 아래에 바로 생성

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# train data
train_sample = pd.read_csv("../data/train.csv", header=0, encoding="utf-8")
path = r"../data/CNC Virtual Data set _v2"
all_files = glob.glob(path + "/*.csv")

# change data type
train_sample_np = np.array(train_sample.copy())

"""
# 결과 데이터 개수 확인
nb_pass = 0
nb_pass_half = 0
nb_defective = 0

for i in range(len(train_sample_np)):
    if train_sample_np[i,5] == 'no':
        nb_defective += 1
    if train_sample_np[i,5] == 'yes' and train_sample_np[i,6] == 'yes':
        nb_pass += 1
    if train_sample_np[i,5] == 'yes' and train_sample_np[i,6] == 'no':
        nb_pass_half += 1

print('양품 : ', nb_pass)
print('공정완료, 육안검사 불합격 : ', nb_pass_half)
print('공정중지 : ', nb_defective)
print('전체 : ', nb_pass+nb_pass_half+nb_defective)
"""

# load csv file
li_df = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li_df.append(df)

# 학습을 위한 결과라벨 데이터 전처리(train.csv)
train_sample_info = tool_condition(train_sample_np)
train_sample_info = item_inspection(train_sample_info)
train_sample_info = np.delete(train_sample_info, 5, 1)
train_sample_info = np.delete(train_sample_info, 0, 1)
train_sample_info = np.delete(train_sample_info, 0, 1)

# 학습을 위한 학습 데이터 전처리1(experiment_01~25.csv)
k = 0
li_pass = []
li_pass_half = []
li_fail = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)

    if train_sample_info[k, 3] == 0:
        li_pass.append(df)
    elif train_sample_info[k, 3] == 1:
        li_pass_half.append(df)
    else:
        li_fail.append(df)

    k += 1

# 각각 있는 index를 무시하고 병합
frame01 = pd.concat(li_pass, axis=0, ignore_index=True)
frame02 = pd.concat(li_pass_half, axis=0, ignore_index=True)
frame03 = pd.concat(li_fail, axis=0, ignore_index=True)

data_pass = np.array(frame01.copy())
data_pass_half = np.array(frame02.copy())
data_fail = np.array(frame03.copy())

data_pass = machining_process(data_pass)
data_pass_half = machining_process(data_pass_half)
data_fail = machining_process(data_fail)

# Dataset 구성
data01 = data_pass[
    0 : 3228 + 6175, :
]  # 미공정+불합격=불량품9403 개수 만큼 양품 개수 설정
data02 = data_pass_half[0:6175, :]
data03 = data_fail[0:3228, :]

data = np.concatenate((data01, data02), axis=0)
data = np.concatenate(
    (data, data03), axis=0
)  # X_train 18806 < 학습데이터 - 양품9403+불량품9403
data_all = data_pass[3228 + 6175 : 22645, :]  # X_test 13242 < 평가데이터 - 모두 양품

# 학습을 위한 학습 데이터 전처리2
sc = MinMaxScaler()
X_train = sc.fit_transform(data)  # normalization
X_train = np.array(X_train)
X_test = sc.transform(data_all)  # test data는 fit_transform이 아닌 transform 사용
X_test = np.array(X_test)

# 데이터 라벨링
Y_train = np.zeros((len(X_train), 1), dtype="int")  # (18806, 1)
Y_test = np.zeros((len(X_test), 1), dtype="int")  # (13242, 1)
l = int(Y_train.shape[0] / 2)
Y_train[0:l, :] = 0  # 양품
Y_train[l : l * 2, :] = 1  # 불량품

# AI모델 파라미터 설정
nb_classes = 2  # 라벨 종류의 개수 < 현재 데이터에서 '양품', '불량품'으로 2개
batch_size = 1024
epochs = 300
lr = 5e-5

# AI 데이터셋 준비
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = tf.keras.utils.to_categorical(Y_train, nb_classes)  # one-hot encoding
Y_test = tf.keras.utils.to_categorical(Y_test, nb_classes)


# AI모델 디자인
model = Sequential()
model.add(Dense(256, activation="relu", input_dim=48, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1024, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(1024, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(nb_classes, activation="sigmoid", kernel_regularizer=l2(0.01)))

# 모델 체크포인트 및 옵티마이저 설정
model_checkpoint = ModelCheckpoint(
    os.path.join(result_dir, "weight_CNC_binary.keras"),
    monitor="val_accuracy",
    save_best_only=True,
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=30,  # 30회 동안 개선이 없으면 중단
    restore_best_weights=True,  # 가장 좋은 성능의 가중치 복원
    verbose=1,
    min_delta=0.01,  # 0.01 이상 개선되지 않으면 patience 적용
)

opt = Adam(learning_rate=lr)
model.summary()
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# 커스텀 콜백 생성
history = History()

# AI모델 훈련
model.fit(
    X_train,
    Y_train,
    verbose=2,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    shuffle=True,
    callbacks=[history, model_checkpoint, early_stopping],
)

# 모델 저장
model.save_weights(os.path.join(result_dir, "weight_CNC_binary.weights.h5"))
model.save(os.path.join(result_dir, "CNC_DLL.keras"))

# 결과 분석 및 해석
loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=32)
print("Train set results:", loss_and_metrics)

loss_and_metrics2 = model.evaluate(X_test, Y_test, batch_size=32)
print("Test set results:", loss_and_metrics2)


# 최소, 최대 epoch 찾기
def get_min_max_epochs(history_dict):
    train_loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    train_min_epoch = train_loss.index(min(train_loss)) + 1
    train_max_epoch = train_loss.index(max(train_loss)) + 1
    val_min_epoch = val_loss.index(min(val_loss)) + 1
    val_max_epoch = val_loss.index(max(val_loss)) + 1

    return {
        "train_min_epoch": train_min_epoch,
        "train_max_epoch": train_max_epoch,
        "val_min_epoch": val_min_epoch,
        "val_max_epoch": val_max_epoch,
    }


# 에포크 정보 얻기
epoch_info = get_min_max_epochs(history.history)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(result_dir, "accuracy_plot.png"))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
# 최소/최대값 포인트 표시
plt.plot(
    epoch_info["train_min_epoch"] - 1,
    min(history.history["loss"]),
    "go",
    label="Train Min",
)
plt.plot(
    epoch_info["train_max_epoch"] - 1,
    max(history.history["loss"]),
    "ro",
    label="Train Max",
)
plt.plot(
    epoch_info["val_min_epoch"] - 1,
    min(history.history["val_loss"]),
    "g^",
    label="Val Min",
)
plt.plot(
    epoch_info["val_max_epoch"] - 1,
    max(history.history["val_loss"]),
    "r^",
    label="Val Max",
)
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(result_dir, "loss_plot.png"))
plt.show()

# 최종 결과 저장
with open(os.path.join(result_dir, "final_results.txt"), "w") as f:
    f.write("=== Training Results ===\n")
    f.write(
        f"Training Loss - Min: {min(history.history['loss']):.4f} (Epoch {epoch_info['train_min_epoch']}), "
        f"Max: {max(history.history['loss']):.4f} (Epoch {epoch_info['train_max_epoch']})\n"
    )
    f.write(
        f"Validation Loss - Min: {min(history.history['val_loss']):.4f} (Epoch {epoch_info['val_min_epoch']}), "
        f"Max: {max(history.history['val_loss']):.4f} (Epoch {epoch_info['val_max_epoch']})\n"
    )
    f.write(f"Final Training Loss: {loss_and_metrics[0]:.4f}\n")
    f.write(f"Final Training Accuracy: {loss_and_metrics[1]:.4f}\n\n")

    f.write("=== Test Results ===\n")
    f.write(f"Final Test Loss: {loss_and_metrics2[0]:.4f}\n")
    f.write(f"Final Test Accuracy: {loss_and_metrics2[1]:.4f}\n")

# 모델 구조 저장
with open(os.path.join(result_dir, "model_architecture.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# 학습 설정 정보 저장
with open(os.path.join(result_dir, "training_config.txt"), "w") as f:
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Training Data Size: {len(X_train)}\n")
    f.write(f"Test Data Size: {len(X_test)}\n")
