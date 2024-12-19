# path: ~/Develop/dnn_test/src/experiment.py
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Dense

class ModelExperiment:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.histories = {}
        
        # 결과 저장을 위한 디렉토리 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src 디렉토리
        project_root = os.path.dirname(current_dir)  # dnn_test 디렉토리
        self.result_dir = os.path.join(project_root, 'runs', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.result_dir, exist_ok=True)
    
    def run_experiment(self, name, optimizer_config):
        """실험 실행 및 결과 저장"""
        from model_factory import ModelFactory
        model = ModelFactory.create_model(optimizer_config, self.X_train.shape[1])
        
        history = model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.1,
            epochs=300,  # 필요하다면 epochs 조정
            batch_size=1024,
            callbacks=optimizer_config.get_callbacks(300),
            verbose=1
        )
        
        test_score = model.evaluate(self.X_test, self.y_test, verbose=0)

        self.histories[name] = {
            'history': history.history,
            'config': optimizer_config.current_config.copy(),
            'test_score': test_score
        }
        
        # 결과 저장
        self.save_results(name, history, model)
        
        return history

    def save_results(self, name, history, model):
        """실험 결과 저장"""
        experiment_dir = os.path.join(self.result_dir, name)
        os.makedirs(experiment_dir, exist_ok=True)

        # 학습 히스토리 저장
        history_dict = {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'max_accuracy': float(max(history.history['accuracy'])),
            'min_loss': float(min(history.history['loss'])),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_epoch': int(history.history['val_accuracy'].index(max(history.history['val_accuracy'])))
        }

        with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as f:
            json.dump(history_dict, f, indent=4)

        # 학습 곡선 저장
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Model Accuracy
        ax1.plot(history.history['accuracy'], label='Training')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title(f'{name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Model Loss
        ax2.plot(history.history['loss'], label='Training')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title(f'{name} - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'learning_curves.png'))
        plt.close()

        # 모델 저장
        model.save(os.path.join(experiment_dir, 'model.keras'))

    def plot_results(self):
        """모든 실험 결과 시각화"""
        plt.figure(figsize=(15, 5))

        # 정확도 그래프
        plt.subplot(1, 2, 1)
        for name, result in self.histories.items():
            plt.plot(result['history']['accuracy'], label=f'{name}_train_acc')
            plt.plot(result['history']['val_accuracy'], label=f'{name}_val_acc')
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # 손실 그래프
        plt.subplot(1, 2, 2)
        for name, result in self.histories.items():
            plt.plot(result['history']['loss'], label=f'{name}_train_loss')
            plt.plot(result['history']['val_loss'], label=f'{name}_val_loss')
        plt.title('Model Loss Comparison')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()