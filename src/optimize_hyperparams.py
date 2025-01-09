# path: D:\DNN_test\src\optimize_hyperparams.py
# 최적화 설정값
N_TRIALS = 10          # 트라이얼 횟수
N_EPOCHS = 100         # 에포크 수
BATCH_SIZES = [256, 512, 1024]  # 배치 사이즈 옵션
LR_RANGE = (1e-5, 1e-3)  # 학습률 범위
LAYER_RANGE = (1, 3)    # 레이어 수 범위
DECAY_STEPS_RANGE = (20, 40)  # lr decay steps 범위 (에포크 수에 맞춰 조정)

import os
import sys
import datetime
import optuna
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Local imports
from olddata import X_train, X_test, Y_train, Y_test
from config import save_training_results, log_experiment_config

def create_model(trial):
    """Auto-encoder 스타일의 모델 생성"""
    model = Sequential()
    
    # 하이퍼파라미터 설정
    use_batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    use_he_init = trial.suggest_categorical('he_init', [True, False])
    use_l2_reg = trial.suggest_categorical('l2_reg', [True, False])
    l2_lambda = trial.suggest_float('l2_lambda', 1e-5, 1e-2, log=True) if use_l2_reg else None
    
    # 초기화 설정
    kernel_init = "he_normal" if use_he_init else "glorot_uniform"
    
    # 레이어 개수 최적화 (인코더 부분)
    n_encoder_layers = trial.suggest_int('n_encoder_layers', LAYER_RANGE[0], LAYER_RANGE[1])
    
    # 레이어 유닛 수 최적화
    first_units = trial.suggest_categorical('first_units', [64, 128, 256])
    middle_units = trial.suggest_categorical('middle_units', [256, 512, 1024])  # 이게 빠졌었네요
    
    # 첫 번째 레이어 (입력층)
    model.add(Dense(first_units, 
                   activation='relu',
                   input_dim=48,
                   kernel_initializer=kernel_init,
                   kernel_regularizer=l2(l2_lambda) if use_l2_reg else None))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(trial.suggest_float('dropout_rate1', 0.1, 0.5)))

    # 추가 인코더 레이어들
    current_units = first_units
    for i in range(n_encoder_layers - 1):  # 첫 번째 레이어는 이미 추가했으므로 -1
        # 점진적으로 middle_units까지 증가
        if i == 0:
            next_units = middle_units
        else:
            next_units = middle_units * 2
        model.add(Dense(next_units,
                       activation='relu',
                       kernel_initializer=kernel_init,
                       kernel_regularizer=l2(l2_lambda) if use_l2_reg else None))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(trial.suggest_float(f'dropout_rate_enc_{i+2}', 0.1, 0.5)))
        current_units = next_units

    # 디코더 레이어들 (인코더와 대칭되게)
    for i in range(n_encoder_layers - 1):
        next_units = current_units // 2
        model.add(Dense(next_units,
                       activation='relu',
                       kernel_initializer=kernel_init,
                       kernel_regularizer=l2(l2_lambda) if use_l2_reg else None))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(trial.suggest_float(f'dropout_rate_dec_{i+1}', 0.1, 0.5)))
        current_units = next_units

    # 출력층
    model.add(Dense(2, activation='softmax'))
    
    return model

def objective(trial):
    """Optuna 최적화를 위한 목적 함수"""
    # 하이퍼파라미터
    batch_size = trial.suggest_categorical('batch_size', BATCH_SIZES)
    learning_rate = trial.suggest_float('learning_rate', LR_RANGE[0], LR_RANGE[1], log=True)
    use_lr_scheduler = trial.suggest_categorical('lr_scheduler', [True, False])
    
    # 모델 생성
    model = create_model(trial)
    
    # 옵티마이저 및 컴파일
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # 콜백 초기화 (Early Stopping 제거)
    callbacks = []
    
    # Learning Rate Scheduler
    if use_lr_scheduler:
        def lr_schedule(epoch):
            initial_lr = learning_rate
            decay_rate = trial.suggest_float('lr_decay_rate', 0.1, 0.5)
            decay_steps = trial.suggest_int('lr_decay_steps', DECAY_STEPS_RANGE[0], DECAY_STEPS_RANGE[1])
            if epoch < decay_steps:
                return initial_lr
            return initial_lr * (decay_rate ** (epoch // decay_steps))
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule))
    
    try:
        # 학습
        history = model.fit(X_train, Y_train,
                          batch_size=batch_size,
                          epochs=N_EPOCHS,
                          validation_split=0.2,
                          callbacks=callbacks,
                          verbose=1)
        
        # 평가
        test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
        
        # 결과 저장
        trial.set_user_attr('val_accuracy', max(history.history['val_accuracy']))
        trial.set_user_attr('train_accuracy', history.history['accuracy'][-1])
        trial.set_user_attr('test_accuracy', test_accuracy)
        
        return test_accuracy
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def main():
    try:
        # 결과 저장 디렉토리 생성
        output_dir = "optuna_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 연구 이름에 타임스탬프 추가
        study_name = f"cnc_optimization_{datetime.datetime.now().strftime('%m%d_%H%M')}"
        
        # 명시적 경로 지정으로 DB 파일 저장
        db_path = os.path.join(output_dir, f"{study_name}.db")
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f"sqlite:///{db_path}"
        )
        
        # 최적화 실행
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
        
        # 최적의 하이퍼파라미터 출력
        print("\nBest trial:")
        trial = study.best_trial
        
        print("  Value: ", trial.value)
        print("\nBest hyperparameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        print("\nBest trial metrics:")
        print(f"  Best validation accuracy: {trial.user_attrs['val_accuracy']:.4f}")
        print(f"  Final training accuracy: {trial.user_attrs['train_accuracy']:.4f}")
        print(f"  Test accuracy: {trial.user_attrs['test_accuracy']:.4f}")
        
        # 결과를 파일로 저장
        results_file = os.path.join(output_dir, f"optuna_results_{study_name}.txt")
        with open(results_file, "w") as f:
            f.write("Best trial results:\n\n")
            f.write(f"Value: {trial.value}\n\n")
            f.write("Best hyperparameters:\n")
            for key, value in trial.params.items():
                f.write(f"{key}: {value}\n")
            f.write("\nBest trial metrics:\n")
            f.write(f"Best validation accuracy: {trial.user_attrs['val_accuracy']:.4f}\n")
            f.write(f"Final training accuracy: {trial.user_attrs['train_accuracy']:.4f}\n")
            f.write(f"Test accuracy: {trial.user_attrs['test_accuracy']:.4f}\n")
        
        print(f"\nResults saved to {results_file}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        # 에러 로그 저장
        error_log_path = os.path.join(output_dir, "error_log.txt")
        with open(error_log_path, "w") as f:
            f.write(f"Error occurred at {datetime.datetime.now()}: {str(e)}\n")
        raise

if __name__ == "__main__":
    main()