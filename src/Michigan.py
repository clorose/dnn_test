# path: D:\DNN_test\src\Michigan.py
import pandas as pd
import numpy as np
import glob
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from config import *
from data_processor import tool_condition, item_inspection, machining_process

class CNCDataProcessor:
    def __init__(self, data_path):
        """데이터 프로세서 초기화"""
        self.data_path = data_path
        self.virtual_data_path = os.path.join(data_path, "CNC_SMART_MICHIGAN")

    def load_and_preprocess(self):
        """데이터 로드 및 전처리"""
        # train.csv 로드 및 전처리
        train_sample = pd.read_csv(
            os.path.join(self.virtual_data_path, "train.csv"), encoding="utf-8"
        )
        train_sample_np = np.array(train_sample.copy())

        # 18개의 실험 파일 로드
        all_files = sorted(
            glob.glob(os.path.join(self.virtual_data_path, "experiment_*.csv"))
        )
        if len(all_files) != 18:
            raise ValueError(
                f"Expected 18 experiment files, but found {len(all_files)}"
            )

        print(f"\n=== 파일 정보 ===")
        print(f"실험 파일 수: {len(all_files)}")

        # train.csv 전처리
        train_sample_info = np.array(train_sample_np.copy())
        train_sample_info = tool_condition(train_sample_info)
        train_sample_info = item_inspection(train_sample_info)
        train_sample_info = np.delete(train_sample_info, 5, 1)
        train_sample_info = np.delete(train_sample_info, 0, 1)
        train_sample_info = np.delete(train_sample_info, 0, 1)

        # 데이터 분류
        li_pass, li_pass_half, li_fail = [], [], []

        for k, filename in enumerate(all_files):
            if k >= len(train_sample_info):
                print(f"Warning: 더 이상의 파일은 처리하지 않습니다. ({k}번째 파일)")
                break

            df = pd.read_csv(filename, index_col=None, header=0)
            print(
                f"Processing file {k+1}/{len(train_sample_info)}: {os.path.basename(filename)}"
            )
            label = train_sample_info[k, 3]

            if label == 0:
                li_pass.append(df)
            elif label == 1:
                li_pass_half.append(df)
            else:
                li_fail.append(df)

        # 데이터 병합
        frame01 = pd.concat(li_pass, axis=0, ignore_index=True)
        frame02 = pd.concat(li_pass_half, axis=0, ignore_index=True)
        frame03 = pd.concat(li_fail, axis=0, ignore_index=True)

        # numpy 배열로 변환 및 전처리
        data_pass = machining_process(np.array(frame01.copy()))
        data_pass_half = machining_process(np.array(frame02.copy()))
        data_fail = machining_process(np.array(frame03.copy()))

        return data_pass, data_pass_half, data_fail

    def prepare_train_test_data(self, data_pass, data_pass_half, data_fail):
        """학습/테스트 데이터 준비"""
        # 불량품(불합격 + 미완료) 데이터 실제 크기 사용
        defect_size = len(data_pass_half) + len(data_fail)  # 6,103개

        # 양품은 불량품과 같은 크기만큼만 훈련에 사용
        data01 = data_pass[0:defect_size, :]  # 양품 6,103개, 실제 값 19,183개
        data02 = data_pass_half  # 불합격 3,942개
        data03 = data_fail  # 미완료 2,161개

        print("\n=== 데이터 분포 변경 ===")
        print(f"정상 데이터: {len(data01)}")
        print(f"육안검사 불합격: {len(data02)}")
        print(f"공정 미완료: {len(data03)}")

        # 데이터 병합 (총 12,206개: 양품 6,103 + 불량품 6,103)
        data = np.concatenate((data01, data02), axis=0)
        data = np.concatenate((data, data03), axis=0)

        # 나머지 양품은 테스트 세트로 사용 (13,080개: 19,183 - 6,103)
        data_all = data_pass[defect_size:]

        # 스케일링
        sc = MinMaxScaler()
        X_train = sc.fit_transform(data)
        X_test = sc.transform(data_all)

        scaler_path = os.path.join(output_dir, 'minmax_scaler.joblib')
        joblib.dump(sc, scaler_path)

        # 레이블 생성
        Y_train = np.zeros((len(X_train), 1), dtype="int")
        Y_test = np.zeros((len(X_test), 1), dtype="int")

        # 레이블링
        half = int(Y_train.shape[0] / 2)
        Y_train[0:half, :] = 0
        Y_train[half : half * 2, :] = 1

        # 데이터 타입 변환
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        Y_train = to_categorical(Y_train, NB_CLASSES)
        Y_test = to_categorical(Y_test, NB_CLASSES)

        return X_train, X_test, Y_train, Y_test

    def create_model(self):
        """모델 생성"""
        model = Sequential()

        # Expansion layers
        model.add(
            Dense(
                HIDDEN_UNITS[0],
                activation=ACTIVATION,
                input_dim=48,
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

        # model.add(
        #     Dense(
        #         HIDDEN_UNITS[2],
        #         activation=ACTIVATION,
        #         kernel_initializer=KERNEL_INITIALIZER,
        #         kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
        #     )
        # )
        # if USE_BATCH_NORM:
        #     model.add(BatchNormalization())
        # if USE_DROPOUT:
        #     model.add(Dropout(DROPOUT_RATES[2]))

        # # Reduction layers
        # model.add(
        #     Dense(
        #         HIDDEN_UNITS[2],
        #         activation=ACTIVATION,
        #         kernel_initializer=KERNEL_INITIALIZER,
        #         kernel_regularizer=l2(L2_LAMBDA) if USE_L2_REG else None,
        #     )
        # )
        # if USE_BATCH_NORM:
        #     model.add(BatchNormalization())
        # if USE_DROPOUT:
        #     model.add(Dropout(DROPOUT_RATES[2]))

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

        return model

    def train_and_evaluate(self, model, X_train, Y_train, X_test, Y_test, output_dir):
        """모델 학습 및 평가"""
        # 콜백 설정
        callbacks = []

        # Learning Rate Scheduler
        if USE_LR_SCHEDULER:

            def lr_schedule(epoch):
                if epoch < LR_DECAY_STEPS[0]:
                    return LEARNING_RATE
                elif epoch < LR_DECAY_STEPS[1]:
                    return LEARNING_RATE * LR_DECAY_RATES[0]
                return LEARNING_RATE * LR_DECAY_RATES[1]

            callbacks.append(LearningRateScheduler(lr_schedule))

        # 모델 체크포인트
        callbacks.append(
            ModelCheckpoint(
                MODEL_CHECKPOINT_PATH,
                monitor=MODEL_CHECKPOINT_MONITOR,
                save_best_only=MODEL_CHECKPOINT_SAVE_BEST,
            )
        )

        # 모델 컴파일
        model.compile(
            optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # 실험 설정 로깅
        log_experiment_config(output_dir)

        # 모델 학습
        history = model.fit(
            X_train,
            Y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            shuffle=True,
            callbacks=callbacks,
            verbose=2,
        )

        # 모델 평가
        train_results = model.evaluate(X_train, Y_train, batch_size=EVAL_BATCH_SIZE)
        test_results = model.evaluate(X_test, Y_test, batch_size=EVAL_BATCH_SIZE)

        # 결과 저장
        save_training_results(model, history, output_dir)

        return history, train_results, test_results


if __name__ == "__main__":
    # 데이터 처리
    processor = CNCDataProcessor(data_path)
    data_pass, data_pass_half, data_fail = processor.load_and_preprocess()
    X_train, X_test, Y_train, Y_test = processor.prepare_train_test_data(
        data_pass, data_pass_half, data_fail
    )

    # 모델 생성
    model = processor.create_model()
    model.summary()

    # 모델 학습 및 평가
    history, train_results, test_results = processor.train_and_evaluate(
        model, X_train, Y_train, X_test, Y_test, output_dir
    )
