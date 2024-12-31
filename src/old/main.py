# path: ~/Develop/dnn_test/src/main.py
import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 최적화 모듈 import
from optimizers import ModelOptimizer
from model_factory import ModelFactory
from experiment import ModelExperiment
from function_dataPreprocess import tool_condition, item_inspection, machining_process

def load_data(data_path):
    """데이터 로드"""
    # train.csv 파일 로드
    train_sample = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_sample_np = np.array(train_sample.copy())
    
    # 라벨 데이터 전처리
    train_sample_info = tool_condition(train_sample_np)
    train_sample_info = item_inspection(train_sample_info)
    train_sample_info = np.delete(train_sample_info, 5, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)
    train_sample_info = np.delete(train_sample_info, 0, 1)
    
    # experiment 파일들 로드
    all_files = glob.glob(os.path.join(data_path, 'CNC Virtual Data set _v2', 'experiment_*.csv'))
    
    # 각 실험 데이터 분류
    li_pass = []
    li_pass_half = []
    li_fail = []
    
    for k, filename in enumerate(all_files):
        df = pd.read_csv(filename)
        if train_sample_info[k, 3] == 0:
            li_pass.append(df)
        elif train_sample_info[k, 3] == 1:
            li_pass_half.append(df)
        else:
            li_fail.append(df)
    
    # 데이터 병합
    frame01 = pd.concat(li_pass, axis=0, ignore_index=True)
    frame02 = pd.concat(li_pass_half, axis=0, ignore_index=True)
    frame03 = pd.concat(li_fail, axis=0, ignore_index=True)
    
    # numpy 변환
    data_pass = np.array(frame01.copy())
    data_pass_half = np.array(frame02.copy())
    data_fail = np.array(frame03.copy())
    
    # machining process 전처리
    data_pass = machining_process(data_pass)
    data_pass_half = machining_process(data_pass_half)
    data_fail = machining_process(data_fail)
    
    print(f"\n데이터 크기:")
    print(f"양품 데이터: {data_pass.shape}")
    print(f"불합격 데이터: {data_pass_half.shape}")
    print(f"불량품 데이터: {data_fail.shape}")
    
    return train_sample_info, data_pass, data_pass_half, data_fail

def prepare_data(train_sample_info, data_pass, data_pass_half, data_fail):
    """데이터 전처리"""
    # 각 클래스별 데이터 수를 동일하게 맞춤
    min_samples = min(len(data_pass), len(data_pass_half), len(data_fail))
    
    # 데이터셋 구성 (균형잡힌 샘플링)
    data01 = data_pass[:min_samples, :]    # 양품
    data02 = data_pass_half[:min_samples, :]  # 불합격
    data03 = data_fail[:min_samples, :]    # 불량품
    
    # 데이터 병합
    data = np.concatenate((data01, data02, data03), axis=0)
    
    # 검증 데이터는 남은 데이터에서 추출
    remaining_data = data_pass[min_samples:, :]
    
    # Machining_Process 컬럼 제외
    data = data[:, :-1]
    remaining_data = remaining_data[:, :-1]
    
    # 정규화
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(data)
    X_test = scaler.transform(remaining_data)
    
    # 라벨링 (3클래스 분류)
    Y_train = np.zeros((len(X_train),), dtype='int')
    Y_test = np.zeros((len(X_test),), dtype='int')
    
    # 훈련 데이터 라벨링
    n_per_class = min_samples
    Y_train[:n_per_class] = 0          # 양품
    Y_train[n_per_class:2*n_per_class] = 1  # 불합격
    Y_train[2*n_per_class:] = 2        # 불량품
    
    print("\n데이터 특성:")
    print(f"학습 데이터 형태: {X_train.shape}")
    print(f"테스트 데이터 형태: {X_test.shape}")
    print(f"레이블 분포:\n{pd.Series(Y_train).value_counts()}")
    
    return X_train, X_test, Y_train, Y_test

def display_results(experiment_results):
    """실험 결과 출력"""
    print("\n=== 실험 결과 요약 ===")
    for name, result in experiment_results.histories.items():
        print(f"\n{name} 실험:")
        print(f"테스트 정확도: {result['test_score'][1]:.4f}")
        print("설정:")
        for key, value in result['config'].items():
            print(f"  {key}: {value}")

def main():
    # 데이터 경로 설정
    data_path = '/app/data'
    
    print("데이터 로드 중...")
    train_sample_info, data_pass, data_pass_half, data_fail = load_data(data_path)
    
    print("데이터 전처리 중...")
    X_train, X_test, y_train, y_test = prepare_data(train_sample_info, data_pass, data_pass_half, data_fail)
    
    # 실험 객체 생성
    experiment = ModelExperiment(X_train, X_test, y_train, y_test)
    
    # 1. 기본 모델 실험
    print("\n기본 모델 학습 중...")
    base_optimizer = ModelOptimizer()
    experiment.run_experiment('base', base_optimizer)
    
    # 2. BatchNorm 적용 실험
    print("\nBatchNorm 모델 학습 중...")
    batchnorm_optimizer = ModelOptimizer().enable_batch_norm()
    experiment.run_experiment('batchnorm', batchnorm_optimizer)
    
    # 3. Dropout 조정 실험
    print("\nDropout 조정 모델 학습 중...")
    dropout_optimizer = ModelOptimizer().set_dropout(0.5)
    experiment.run_experiment('dropout_0.5', dropout_optimizer)
    
    # 4. 복합 최적화 실험
    print("\n복합 최적화 모델 학습 중...")
    complex_optimizer = ModelOptimizer()\
        .set_hidden_size(256)\
        .enable_batch_norm()\
        .set_dropout(0.3)\
        .enable_label_smoothing()\
        .enable_cosine_scheduler()
    experiment.run_experiment('complex', complex_optimizer)
    
    # 결과 시각화 및 출력
    print("\n결과 시각화 중...")
    experiment.plot_results()
    display_results(experiment)
    
    print("\n모든 실험이 완료되었습니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류 발생: {str(e)}")