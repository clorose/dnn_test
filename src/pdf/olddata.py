# path: ~/Develop/dnn_test/src/pdf/olddata.py
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Keras 및 TensorFlow 관련 임포트 (3.11 기준 최신 버전에 맞게 변경)
import datetime
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers, regularizers, constraints, backend as K

timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
# 재귀 제한 증가
sys.setrecursionlimit(10000)

# For plotting
import matplotlib.pyplot as plt

# Set the path to the data(Docker)
data_path = "/app/data"
root_path = "/app/"

# Set the path to the data(Local)
# data_path = "../../data"
# root_path = "../../"

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


def tool_condition(input):
    for i in range(len(input)):
        if input[i, 4] == "unworn":
            input[i, 4] = 0
        else:
            input[i, 4] = 1
    return input


def item_inspection(input):
    for i in range(len(input)):
        if input[i, 5] == "no":
            input[i, 6] = 2
        elif input[i, 5] == "yes" and input[i, 6] == "no":
            input[i, 6] = 1
        elif input[i, 5] == "yes" and input[i, 6] == "yes":
            input[i, 6] = 0
    return input


def machining_process(input):
    for i in range(len(input)):
        if input[i, 47] == "Prep":
            input[i, 47] = 0
        elif input[i, 47] == "Layer 1 Up":
            input[i, 47] = 1
        elif input[i, 47] == "Layer 1 Down":
            input[i, 47] = 2
        elif input[i, 47] == "Layer 2 Up":
            input[i, 47] = 3
        elif input[i, 47] == "Layer 2 Down":
            input[i, 47] = 4
        elif input[i, 47] == "Layer 3 Up":
            input[i, 47] = 5
        elif input[i, 47] == "Layer 3 Down":
            input[i, 47] = 6
        elif input[i, 47] == "Repositioning":
            input[i, 47] = 7
        elif input[i, 47] == "End" or input[i, 47] == "end":
            input[i, 47] = 8
        elif input[i, 47] == "Starting":
            input[i, 47] = 9
    return input


# Modifying train.csv for training
# - [tool_condition] : unworn/worn -> 0 / 1
# - [item_inspection] : machining_finalized & passed -> yes & yes / yes & no / no -> 0 / 1 / 2
# - delete 'material' column and 'No' column

# Assuming train_sample_np is already defined
train_sample_info = np.array(train_sample_np.copy())

# Apply user-defined functions
train_sample_info = tool_condition(train_sample_info)
train_sample_info = item_inspection(train_sample_info)

# Print the resulting data
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
    # 디버깅을 위한 출력 추가
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

# Debugging: Check if the lists have data before attempting to concatenate
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
# - machining_process : From "Prep" to "End" -> 0~9
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

# Print the label data
print(Y_train)

nb_classes = 2
batch_size = 1024
epochs = 300
lr = 1e-4

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
model.add(Dense(128, activation="relu", input_dim=48))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(nb_classes, activation="softmax"))
model_checkpoint = ModelCheckpoint(
    "weight_CNC_binary.mat", monitor="val_acc", save_best_only=True
)
opt = Adam(lr)
model.summary()
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

history = History()

print("............model is defined............")

model.fit(
    X_train,
    Y_train,
    verbose=2,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    shuffle=True,
    callbacks=[history],
)

loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=32)
print(loss_and_metrics)

loss_and_metrics2 = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics2)

plt.figure()
plt.plot(history.history["val_accuracy"])
plt.plot(history.history["accuracy"])
plt.title("Accuracy During Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Validation Accuracy", "Training Accuracy"])
plt.savefig(f"accuracy_{timestamp}.png")
plt.close()

plt.figure()
plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])
plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Validation Loss", "Training Loss"])
plt.savefig(f"loss_{timestamp}.png")
plt.close()
