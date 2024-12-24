# path: ~/Develop/dnn_test/src/bayesian_opt.py

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


class BayesianOptimizer:
    def __init__(self, experiment, base_config):
        self.experiment = experiment
        self.base_config = base_config

        # 하이퍼파라미터 탐색 공간 정의
        self.dimensions = [
            # 네트워크 구조
            Integer(128, 1024, name="layer1_size"),
            Integer(256, 2048, name="layer2_size"),
            Integer(512, 2048, name="layer3_size"),
            # 학습 파라미터
            Real(1e-4, 1e-2, prior="log-uniform", name="learning_rate"),
            Integer(256, 2048, name="batch_size"),
            Real(0.1, 0.5, name="dropout_rate"),
            # 정규화
            Real(1e-4, 1e-1, prior="log-uniform", name="l2_lambda"),
            Real(0.0, 0.2, name="label_smoothing"),
            # 배치 정규화 & 옵티마이저
            Categorical([True, False], name="use_batch_norm"),
            Categorical(["adam", "adamw", "rmsprop", "sgd"], name="optimizer"),
            # 옵티마이저 설정
            Real(0.8, 0.999, name="beta_1"),
            Real(0.9, 0.9999, name="beta_2"),
            Real(0.001, 0.1, name="momentum"),
            Real(0.0001, 0.01, name="weight_decay"),
            # 학습률 스케줄링
            Categorical([True, False], name="use_reduce_lr"),
            Categorical([True, False], name="use_cosine_scheduler"),
            Real(1e-6, 1e-4, prior="log-uniform", name="min_lr"),
            Integer(5, 20, name="reduce_lr_patience"),
            Real(0.1, 0.5, name="reduce_lr_factor"),
            # 조기 종료
            Categorical([True, False], name="use_early_stopping"),
            Integer(10, 30, name="early_stopping_patience"),
        ]

        self.objective = use_named_args(dimensions=self.dimensions)(self.objective)

    def objective(
        self,
        layer1_size,
        layer2_size,
        layer3_size,
        learning_rate,
        batch_size,
        dropout_rate,
        l2_lambda,
        label_smoothing,
        use_batch_norm,
        optimizer,
        beta_1,
        beta_2,
        momentum,
        weight_decay,
        use_reduce_lr,
        use_cosine_scheduler,
        min_lr,
        reduce_lr_patience,
        reduce_lr_factor,
        use_early_stopping,
        early_stopping_patience,
    ):

        from optimizers import ModelOptimizer

        model_optimizer = ModelOptimizer()

        # 네트워크 구조 설정
        model_optimizer.current_config["hidden_sizes"] = [
            layer1_size,
            layer2_size,
            layer3_size,
            layer3_size,
            layer2_size,
            layer1_size,
        ]
        model_optimizer.current_config["dropout_rates"] = [dropout_rate] * 6
        model_optimizer.current_config["use_batch_norm"] = use_batch_norm

        # 학습 파라미터 설정
        model_optimizer.current_config["learning_rate"] = learning_rate
        model_optimizer.current_config["batch_size"] = batch_size
        model_optimizer.current_config["l2_lambda"] = l2_lambda
        model_optimizer.current_config["label_smoothing"] = label_smoothing

        # 옵티마이저 설정
        model_optimizer.current_config["optimizer"] = optimizer
        model_optimizer.current_config["optimizer_config"].update(
            {
                "beta_1": beta_1,
                "beta_2": beta_2,
                "epsilon": 1e-07,
            }
        )

        # 옵티마이저별 특별 설정
        if optimizer == "adamw":
            model_optimizer.current_config["optimizer_config"][
                "weight_decay"
            ] = weight_decay
        elif optimizer == "sgd":
            model_optimizer.current_config["optimizer_config"]["momentum"] = momentum

        # 학습률 스케줄링 설정
        model_optimizer.current_config["use_reduce_lr"] = use_reduce_lr
        model_optimizer.current_config["use_cosine_scheduler"] = use_cosine_scheduler
        model_optimizer.current_config["min_lr"] = min_lr
        model_optimizer.current_config["reduce_lr_patience"] = reduce_lr_patience
        model_optimizer.current_config["reduce_lr_factor"] = reduce_lr_factor

        # 조기 종료 설정
        model_optimizer.current_config["use_early_stopping"] = use_early_stopping
        model_optimizer.current_config["early_stopping_patience"] = (
            early_stopping_patience
        )

        # 기본 설정 유지
        model_optimizer.current_config["epochs"] = self.base_config.epochs
        model_optimizer.current_config["classification_type"] = (
            self.base_config.classification_type
        )

        # 실험 실행 및 결과 반환
        result = self.experiment.run_optimization(model_optimizer)
        return result["best_val_loss"]

    def optimize(self, n_calls=40):
        """Bayesian Optimization 실행"""
        n_random_starts = max(5, n_calls // 4)

        result = gp_minimize(
            func=self.objective,
            dimensions=self.dimensions,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            noise=0.1,
            random_state=42,
        )

        return self.parse_results(result)

    def parse_results(self, result):
        """최적화 결과 파싱"""
        best_params = {
            "layer1_size": result.x[0],
            "layer2_size": result.x[1],
            "layer3_size": result.x[2],
            "learning_rate": result.x[3],
            "batch_size": result.x[4],
            "dropout_rate": result.x[5],
            "l2_lambda": result.x[6],
            "label_smoothing": result.x[7],
            "use_batch_norm": result.x[8],
            "optimizer": result.x[9],
            "beta_1": result.x[10],
            "beta_2": result.x[11],
            "momentum": result.x[12],
            "weight_decay": result.x[13],
            "use_reduce_lr": result.x[14],
            "use_cosine_scheduler": result.x[15],
            "min_lr": result.x[16],
            "reduce_lr_patience": result.x[17],
            "reduce_lr_factor": result.x[18],
            "use_early_stopping": result.x[19],
            "early_stopping_patience": result.x[20],
        }

        # 최적 설정 생성
        optimal_config = self.base_config.copy()
        optimal_config.hidden_sizes = [
            best_params["layer1_size"],
            best_params["layer2_size"],
            best_params["layer3_size"],
            best_params["layer3_size"],
            best_params["layer2_size"],
            best_params["layer1_size"],
        ]
        optimal_config.dropout_rates = [best_params["dropout_rate"]] * 6
        optimal_config.learning_rate = best_params["learning_rate"]
        optimal_config.batch_size = best_params["batch_size"]
        optimal_config.l2_lambda = best_params["l2_lambda"]
        optimal_config.label_smoothing = best_params["label_smoothing"]
        optimal_config.use_batch_norm = best_params["use_batch_norm"]
        optimal_config.optimizer = best_params["optimizer"]

        # 옵티마이저 설정
        optimal_config.optimizer_config = {
            "beta_1": best_params["beta_1"],
            "beta_2": best_params["beta_2"],
            "epsilon": 1e-07,
        }

        if best_params["optimizer"] == "adamw":
            optimal_config.optimizer_config["weight_decay"] = best_params[
                "weight_decay"
            ]
        elif best_params["optimizer"] == "sgd":
            optimal_config.optimizer_config["momentum"] = best_params["momentum"]

        # 학습률 스케줄링 설정
        optimal_config.use_reduce_lr = best_params["use_reduce_lr"]
        optimal_config.use_cosine_scheduler = best_params["use_cosine_scheduler"]
        optimal_config.min_lr = best_params["min_lr"]
        optimal_config.reduce_lr_patience = best_params["reduce_lr_patience"]
        optimal_config.reduce_lr_factor = best_params["reduce_lr_factor"]

        # 조기 종료 설정
        optimal_config.use_early_stopping = best_params["use_early_stopping"]
        optimal_config.early_stopping_patience = best_params["early_stopping_patience"]

        return {
            "best_params": best_params,
            "optimal_config": optimal_config,
            "best_score": result.fun,
            "all_scores": result.func_vals,
            "n_iterations": len(result.func_vals),
        }
