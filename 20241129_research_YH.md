다음은 두 연구 노트를 합쳐서 요약한 내용입니다. 코드 부분은 그대로 유지하면서 내용을 간략히 정리했습니다.

---

## CNC 가공 데이터 기반 머신러닝 모델 구축

### 1. 연구 개요

**연구 목적**: CNC 가공 데이터를 기반으로 머신러닝 모델을 구축하여 제품 품질을 예측하는 시스템 개발  
**연구 구성**:  
- 데이터 전처리  
- 모델 구축 및 훈련  
- 평가 및 결과 분석

---

### 2. 데이터 분석 및 전처리

#### 2.1 데이터셋 구성
- **훈련 데이터**: 18,806개  
  - 양품: 9,403개  
  - 불량품: 9,403개 (공정미완료 3,228개, 검사불합격 6,175개)
- **평가 데이터**: 13,242개 (양품)

#### 2.2 데이터 수집 및 전처리
```python
train_sample = pd.read_csv("../data/train.csv", header=0, encoding='utf-8')
path = r'../data/CNC Virtual Data set _v2'
all_files = glob.glob(path + "/*.csv")
```

#### 2.3 데이터 라벨링 및 병합
```python
train_sample_np = np.array(train_sample.copy())
train_sample_info = tool_condition(train_sample_np)
train_sample_info = item_inspection(train_sample_info)

# 데이터 병합
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
```

#### 2.4 데이터 정규화 및 분할
```python
sc = MinMaxScaler()
X_train = sc.fit_transform(data)
X_train = np.array(X_train)
X_test = sc.fit_transform(data_all)
X_test = np.array(X_test)

Y_train = np.zeros((len(X_train), 1), dtype='int')
Y_test = np.zeros((len(X_test), 1), dtype='int')
```

---

### 3. 딥러닝 모델 설계

#### 3.1 모델 아키텍처
```python
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=48))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes, activation='sigmoid'))
```

#### 3.2 모델 컴파일
```python
model_checkpoint = ModelCheckpoint('weight_CNC_binary.mat', monitor='val_acc', save_best_only=True)
opt = Adam(lr=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
```

---

### 4. 실험 결과

#### 4.1 모델 훈련
```python
history = model.fit(X_train, Y_train, verbose=2, batch_size=1024, epochs=300, validation_split=0.1, shuffle=True, callbacks=[history])
model.save_weights('weight_CNC_binary.mat')
model.save("CNC_DLL.h5")
```

#### 4.2 모델 평가
```python
loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=32)
print(loss_and_metrics)

loss_and_metrics2 = model.evaluate(X_test, Y_test, batch_size=32)
print(loss_and_metrics2)
```

#### 4.3 시각화
```python
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
```