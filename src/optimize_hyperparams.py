# path: D:\DNN_test\src\optimize_hyperparams.py
import os
import optuna
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Local imports
from prepare_dataset import prepare_training_data
from optuna_config import (
    ModelConfig,
    OptunaConfig,
    save_training_results,
    log_experiment_config,
    create_experiment_dir,
    handle_error,
    get_timestamp,
    DATA_PATH,
)

# 데이터 로드
virtual_data_path = os.path.join(DATA_PATH, "CNC_SMART_MICHIGAN")
X_train, X_test, Y_train, Y_test = prepare_training_data(DATA_PATH, virtual_data_path)


def create_model(trial):
    """Auto-encoder 스타일의 모델 생성"""
    model = Sequential()

    # 하이퍼파라미터 설정
    use_batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    use_he_init = trial.suggest_categorical("he_init", [True, False])
    l2_lambda = trial.suggest_float(
        "l2_lambda", *OptunaConfig.L2_LAMBDA_RANGE, log=True
    )
    activation = trial.suggest_categorical("activation", ModelConfig.ACTIVATIONS)

    # 초기화 설정
    kernel_init = (
        ModelConfig.INITIALIZERS[0] if use_he_init else ModelConfig.INITIALIZERS[1]
    )

    # 레이어 개수 최적화
    n_encoder_layers = trial.suggest_int("n_encoder_layers", *ModelConfig.LAYER_RANGE)

    # 레이어 유닛 수 최적화
    first_units = trial.suggest_int("first_units", *ModelConfig.INITIAL_UNITS_RANGE)
    middle_units = trial.suggest_int("middle_units", *ModelConfig.MIDDLE_UNITS_RANGE)

    # 첫 번째 레이어 (입력층)
    model.add(
        Dense(
            first_units,
            activation=activation,
            input_dim=ModelConfig.INPUT_DIM,
            kernel_initializer=kernel_init,
            kernel_regularizer=l2(l2_lambda),
        )
    )
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(
        Dropout(trial.suggest_float("dropout_rate1", *OptunaConfig.DROPOUT_RANGE))
    )

    # 인코더 레이어들
    current_units = first_units
    for i in range(n_encoder_layers - 1):
        next_units = middle_units if i == 0 else middle_units * 2
        model.add(
            Dense(
                next_units,
                activation=activation,
                kernel_initializer=kernel_init,
                kernel_regularizer=l2(l2_lambda),
            )
        )
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(
            Dropout(
                trial.suggest_float(
                    f"dropout_rate_enc_{i+2}", *OptunaConfig.DROPOUT_RANGE
                )
            )
        )
        current_units = next_units

    # 디코더 레이어들
    for i in range(n_encoder_layers - 1):
        next_units = current_units // 2
        model.add(
            Dense(
                next_units,
                activation=activation,
                kernel_initializer=kernel_init,
                kernel_regularizer=l2(l2_lambda),
            )
        )
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(
            Dropout(
                trial.suggest_float(
                    f"dropout_rate_dec_{i+1}", *OptunaConfig.DROPOUT_RANGE
                )
            )
        )
        current_units = next_units

    # 출력층
    model.add(Dense(ModelConfig.OUTPUT_DIM, activation=ModelConfig.OUTPUT_ACTIVATION))

    return model


def create_optimizer(trial):
    """옵티마이저 생성 함수"""
    optimizer_name = trial.suggest_categorical("optimizer", OptunaConfig.OPTIMIZERS)
    learning_rate = trial.suggest_float(
        "learning_rate", *OptunaConfig.LR_RANGE, log=True
    )

    if optimizer_name == "adam":
        beta1 = trial.suggest_float("adam_beta1", *OptunaConfig.ADAM_BETA1_RANGE)
        return Adam(learning_rate=learning_rate, beta_1=beta1)
    else:
        return RMSprop(learning_rate=learning_rate)


def create_callbacks(trial):
    """콜백 생성 함수"""
    callbacks = []

    # Early Stopping 설정
    use_early_stopping = trial.suggest_categorical("use_early_stopping", [True, False])
    if use_early_stopping:
        patience = trial.suggest_int(
            "early_stopping_patience", *OptunaConfig.ES_PATIENCE_RANGE
        )
        monitor = trial.suggest_categorical(
            "early_stopping_monitor", OptunaConfig.ES_MONITORS
        )
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    # Learning Rate Scheduler
    use_lr_scheduler = trial.suggest_categorical("lr_scheduler", [True, False])
    if use_lr_scheduler:
        scheduler_type = trial.suggest_categorical(
            "scheduler_type", OptunaConfig.SCHEDULER_TYPES
        )

        if scheduler_type == "step":
            decay_steps = trial.suggest_int(
                "lr_decay_steps", *OptunaConfig.STEP_DECAY_STEPS_RANGE
            )
            decay_rate = trial.suggest_float(
                "lr_decay_rate", *OptunaConfig.LR_DECAY_RATE_RANGE
            )

            def step_decay(epoch):
                initial_lr = trial.params["learning_rate"]
                drop = decay_rate
                epochs_drop = decay_steps
                lr = initial_lr * tf.math.pow(drop, tf.math.floor(epoch / epochs_drop))
                return float(lr)

            callbacks.append(tf.keras.callbacks.LearningRateScheduler(step_decay))
        else:
            decay_rate = trial.suggest_float(
                "exp_decay_rate", *OptunaConfig.EXP_DECAY_RATE_RANGE
            )

            def exp_decay(epoch):
                initial_lr = trial.params["learning_rate"]
                return float(initial_lr * decay_rate**epoch)

            callbacks.append(tf.keras.callbacks.LearningRateScheduler(exp_decay))

    return callbacks


def objective(trial):
    """Optuna 최적화를 위한 목적 함수"""
    # 하이퍼파라미터
    batch_size = trial.suggest_categorical("batch_size", OptunaConfig.BATCH_SIZES)

    # 모델 생성
    model = create_model(trial)

    # 옵티마이저 생성
    optimizer = create_optimizer(trial)

    # 모델 컴파일
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # 콜백 설정
    callbacks = create_callbacks(trial)

    try:
        # 학습
        history = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=OptunaConfig.N_EPOCHS,
            validation_split=OptunaConfig.VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1,
        )

        # 평가
        val_accuracy = max(history.history["val_accuracy"])
        train_accuracy = history.history["accuracy"][-1]
        test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        # 결과 저장
        trial.set_user_attr("val_accuracy", val_accuracy)
        trial.set_user_attr("train_accuracy", train_accuracy)
        trial.set_user_attr("test_accuracy", test_accuracy)

        return test_accuracy
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        raise optuna.exceptions.TrialPruned()


def main():
    try:
        # 실험 디렉토리 생성
        output_dir = create_experiment_dir()

        # 연구 이름에 타임스탬프 추가
        study_name = f"michigan_optimization_{get_timestamp()}"
        study = optuna.create_study(direction="maximize")

        # 최적화 실행
        study.optimize(
            objective,
            n_trials=OptunaConfig.N_TRIALS,
            show_progress_bar=True,
        )

        # 결과 저장 및 설정 로깅
        save_training_results(study, output_dir, study_name)
        log_experiment_config(output_dir)

    except Exception as e:
        handle_error(e, output_dir)


if __name__ == "__main__":
    main()
