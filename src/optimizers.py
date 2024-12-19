# path: ~/Develop/dnn_test/src/optimizers.py
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import LearningRateScheduler
import math

class ModelOptimizer:
    def __init__(self):
        self.current_config = {
            'hidden_size': 128,
            'dropout_rate': 0.3,
            'use_batch_norm': False,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'label_smoothing': 0.0,
            'use_cosine_scheduler': False,
            'gradient_clip': None
        }
    
    def set_hidden_size(self, size):
        """Hidden Layer 크기 설정"""
        self.current_config['hidden_size'] = size
        return self
    
    def set_dropout(self, rate):
        """Dropout 비율 설정"""
        self.current_config['dropout_rate'] = rate
        return self
    
    def enable_batch_norm(self, enable=True):
        """Batch Normalization 활성화/비활성화"""
        self.current_config['use_batch_norm'] = enable
        return self
    
    def set_optimizer(self, optimizer_name, learning_rate=0.001):
        """옵티마이저 설정"""
        self.current_config['optimizer'] = optimizer_name
        self.current_config['learning_rate'] = learning_rate
        return self
    
    def enable_label_smoothing(self, smoothing=0.1):
        """Label Smoothing 설정"""
        self.current_config['label_smoothing'] = smoothing
        return self
    
    def enable_cosine_scheduler(self, enable=True):
        """Cosine Scheduler 활성화/비활성화"""
        self.current_config['use_cosine_scheduler'] = enable
        return self
    
    def set_gradient_clip(self, clip_value):
        """Gradient Clipping 값 설정"""
        self.current_config['gradient_clip'] = clip_value
        return self
    
    def create_dense_block(self, units, input_shape=None):
        """Dense 블록 생성 (BatchNorm + Dense + Dropout)"""
        layers = []
        
        if input_shape is not None:
            dense = Dense(units, activation='relu', input_shape=input_shape)
        else:
            dense = Dense(units, activation='relu')
        
        if self.current_config['use_batch_norm']:
            layers.append(BatchNormalization())
        
        layers.append(dense)
        
        if self.current_config['dropout_rate'] > 0:
            layers.append(Dropout(self.current_config['dropout_rate']))
        
        return layers
    
    def get_optimizer(self):
        """설정된 옵티마이저 반환"""
        lr = self.current_config['learning_rate']
        
        if self.current_config['optimizer'] == 'adamw':
            return AdamW(learning_rate=lr)
        else:
            return Adam(learning_rate=lr)
    
    def get_callbacks(self, epochs):
        """콜백 함수들 반환"""
        callbacks = []
        
        if self.current_config['use_cosine_scheduler']:
            def cosine_decay(epoch):
                return 0.5 * (1 + math.cos(math.pi * epoch / epochs)) * self.current_config['learning_rate']
            
            callbacks.append(LearningRateScheduler(cosine_decay))
        
        return callbacks
    
    def get_compile_kwargs(self):
        """모델 컴파일을 위한 인자들 반환"""
        kwargs = {
            'optimizer': self.get_optimizer(),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']
        }
        
        if self.current_config['label_smoothing'] > 0:
            kwargs['loss'] = tf.keras.losses.BinaryCrossentropy(
                label_smoothing=self.current_config['label_smoothing']
            )
        
        return kwargs