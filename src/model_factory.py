# path: ~/Develop/dnn_test/src/model_factory.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

class ModelFactory:
    @staticmethod
    def create_model(optimizer_config, input_dim):
        """최적화 설정을 적용한 모델 생성"""
        model = Sequential([
            # 입력층
            Dense(128, activation='relu', input_shape=(input_dim,),
                kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            
            # 은닉층 1
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            
            # 은닉층 2
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            
            # 출력층 (3클래스)
            Dense(3, activation='softmax')
        ])
        
        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model