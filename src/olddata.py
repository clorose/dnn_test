# path: D:\DNN_test\src\pdf\olddata.py
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Keras 및 TensorFlow 관련 임포트
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
)
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# Local modules
from config import *
from data_processor import tool_condition, item_inspection, machining_process

virtual_data_path = os.path.join(data_path, "CNC Virtual Data set _v2")
run_data_path = os.path.join(root_path, "runs")

# Load train.csv
train_sample = pd.read_csv(
    os.path.join(data_path, "train.csv"), header=0, encoding="utf-8"
)

# Get All experiment files
all_files = glob.glob(os.path.join(virtual_data_path, "experiment_*.csv"))

# change data type
train_sample_np = np.array(train_sample.copy())

# load csv file
li_df = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li_df.append(df)

# count the number of pass/fail items
nb_pass = 0
nb_pass_half = 0
nb_defective = 0

for i in range(len(train_sample_np)):
    if train_sample_np[i, 5] == "no":
        nb_defective += 1
    if train_sample_np[i, 5] == "yes" and train_sample_np[i, 6] == "yes":
        nb_pass += 1
    if train_sample_np[i, 5] == "yes" and train_sample_np[i, 6] == "no":
        nb_pass_half += 1

print("양품 샘플 개수 :", nb_pass)
print("공정 마쳤으나 육안검사 통과 못한 샘플 개수 :", nb_pass_half)
print("공정 중지된 샘플 개수 :", nb_defective)
print("전체 샘플 개수 :", nb_pass + nb_pass_half + nb_defective)

# Modifying train.csv for training
train_sample_info = np.array(train_sample_np.copy())

# Apply user-defined functions
train_sample_info = tool_condition(train_sample_info)
train_sample_info = item_inspection(train_sample_info)

print(train_sample_info)

train_sample_info = np.delete(train_sample_info, 5, 1)
train_sample_info = np.delete(train_sample_info, 0, 1)
train_sample_info = np.delete(train_sample_info, 0, 1)

print(train_sample_info)
print("train_sample_info shape:", train_sample_info.shape)
print("Number of files:", len(all_files))
print("First few rows of train_sample_info:\n", train_sample_info[:5])

k = 0
li_pass = []
li_pass_half = []
li_fail = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    print(
        f"train_sample_info[k, 3]: {train_sample_info[k, 3]}, type: {type(train_sample_info[k, 3])}"
    )
    label = train_sample_info[k, 3]

    if label == 0:
        li_pass.append(df)
    elif label == 1:
        li_pass_half.append(df)
    else:
        li_fail.append(df)

    k += 1

print(f"li_pass contains {len(li_pass)} items")
print(f"li_pass_half contains {len(li_pass_half)} items")
print(f"li_fail contains {len(li_fail)} items")

frame01 = pd.concat(li_pass, axis=0, ignore_index=True)
frame02 = pd.concat(li_pass_half, axis=0, ignore_index=True)
frame03 = pd.concat(li_fail, axis=0, ignore_index=True)

data_pass = np.array(frame01.copy())
data_pass_half = np.array(frame02.copy())
data_fail = np.array(frame03.copy())

print("공정완료 및 육안검사 합격한 전체 데이터 수 :", len(data_pass))
print("공정완료 및 육안검사 불합격한 전체 데이터 수 :", len(data_pass_half))
print("공정 미완료한 전체 데이터 수 :", len(data_fail))

print(data_pass.shape)
print(data_pass_half.shape)
print(data_fail.shape)

# Modifying experiment data
data_pass = machining_process(data_pass)
data_pass_half = machining_process(data_pass_half)
data_fail = machining_process(data_fail)

# label 0/1 --> data01 / data02+data03
data01 = data_pass[0 : 3228 + 6175, :]
data02 = data_pass_half[0:6175, :]
data03 = data_fail[0:3228, :]

data = np.concatenate((data01, data02), axis=0)
data = np.concatenate((data, data03), axis=0)
data_all = data_pass[3228 + 6175 : 22645, :]

print((data))
print(data.shape)
print(data_all.shape)

sc = MinMaxScaler()
X_train = sc.fit_transform(data)
X_train = np.array(X_train)
X_test = sc.transform(data_all)
X_test = np.array(X_test)

# make label data
Y_train = np.zeros((len(X_train), 1), dtype="int")
Y_test = np.zeros((len(X_test), 1), dtype="int")

# Set the label for the first half as 0 and the second half as 1
half = int(Y_train.shape[0] / 2)
Y_train[0:half, :] = 0
Y_train[half : half * 2, :] = 1

print(Y_train)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
Y_train = to_categorical(Y_train, NB_CLASSES)
Y_test = to_categorical(Y_test, NB_CLASSES)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()

# Expansion layers (차원을 키워가는 층들)
model.add(
    Dense(
        HIDDEN_UNITS[0],
        activation=ACTIVATION,
        input_dim=48,
        # 
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
    )
)
if USE_BATCH_NORM:
    model.add(BatchNormalization())
if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATES[0]))

model.add(
    Dense(
        HIDDEN_UNITS[1],
        activation=ACTIVATION,
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
    )
)
if USE_BATCH_NORM:
    model.add(BatchNormalization())
if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATES[1]))

model.add(
    Dense(
        HIDDEN_UNITS[2],
        activation=ACTIVATION,
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
    )
)
if USE_BATCH_NORM:
    model.add(BatchNormalization())
if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATES[2]))

# Reduction layers (차원을 줄여가는 층들)
model.add(
    Dense(
        HIDDEN_UNITS[2],
        activation=ACTIVATION,
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
    )
)
if USE_BATCH_NORM:
    model.add(BatchNormalization())
if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATES[2]))

model.add(
    Dense(
        HIDDEN_UNITS[1],
        activation=ACTIVATION,
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
    )
)
if USE_BATCH_NORM:
    model.add(BatchNormalization())
if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATES[1]))

model.add(
    Dense(
        HIDDEN_UNITS[0],
        activation=ACTIVATION,
        kernel_initializer=KERNEL_INITIALIZER,
        kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
    )
)
if USE_BATCH_NORM:
    model.add(BatchNormalization())
if USE_DROPOUT:
    model.add(Dropout(DROPOUT_RATES[0]))

model.add(Dense(NB_CLASSES, activation="softmax"))

# 콜백 설정
callbacks = []

if USE_EARLY_STOPPING:
    callbacks.append(
        EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            mode=EARLY_STOPPING_MODE,
            restore_best_weights=EARLY_STOPPING_RESTORE_BEST,
            min_delta = EARLY_STOPPING_MIN_DELTA
        )
    )

if USE_LR_SCHEDULER:

    def lr_schedule(epoch):
        if epoch < LR_DECAY_STEPS[0]:
            return LEARNING_RATE
        elif epoch < LR_DECAY_STEPS[1]:
            return LEARNING_RATE * LR_DECAY_RATES[0]
        return LEARNING_RATE * LR_DECAY_RATES[1]

    callbacks.append(LearningRateScheduler(lr_schedule))

model_checkpoint = ModelCheckpoint(
    MODEL_CHECKPOINT_PATH,
    monitor=MODEL_CHECKPOINT_MONITOR,
    save_best_only=MODEL_CHECKPOINT_SAVE_BEST,
)
callbacks.append(model_checkpoint)

model.summary()
model.compile(
    optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=["accuracy"]
)

print("............model is defined............")

# 모델 학습 직전에 추가
log_experiment_config(output_dir)

# 모델 학습
history = model.fit(
    X_train,
    Y_train,
    verbose=2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    callbacks=callbacks,
)

# 모델 평가
loss_and_metrics = model.evaluate(X_train, Y_train, EVAL_BATCH_SIZE)
print("Training Data Evaluation: ", loss_and_metrics)

loss_and_metrics2 = model.evaluate(X_test, Y_test, EVAL_BATCH_SIZE)
print("Test Data Evaluation: ", loss_and_metrics2)

# 학습 로그 저장
log_file = os.path.join(output_dir, "training_log.txt")
with open(log_file, "w") as f:
    for key, values in history.history.items():
        f.write(f"{key}: {values}\n")
    f.write("\n")
    f.write(f"Training Evaluation Metrics: {loss_and_metrics}\n")
    f.write(f"Test Evaluation Metrics: {loss_and_metrics2}\n")

# 학습 결과 저장
save_training_results(model, history, output_dir)
