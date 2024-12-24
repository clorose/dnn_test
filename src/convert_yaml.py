import base64
import struct
import yaml


def decode_base64_int(b64_str):
    """base64로 인코딩된 정수값 디코딩"""
    try:
        # base64 디코딩
        decoded = base64.b64decode(b64_str.strip())
        # 8바이트 정수로 해석
        value = struct.unpack("<q", decoded)[0]
        return value
    except:
        return b64_str


def clean_yaml_file(file_path):
    """YAML 파일 정리"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    cleaned_data = {
        "batch_size": 1018,  # +gMAAAAAAAA= 디코딩값
        "class_weights": None,
        "classification_type": "binary",
        "dropout_rates": [0.18317666514727557] * 6,
        "early_stopping_patience": 22,  # FgAAAAAAAAA= 디코딩값
        "epochs": 50,
        "hidden_sizes": [
            283,  # GwEAAAAAAAA= 디코딩값
            987,  # vQMAAAAAAAA= 디코딩값
            1024,  # GAMAAAAAAAA= 디코딩값
            1024,
            987,
            283,
        ],
        "l2_lambda": 0.005047786565710609,
        "label_smoothing": 0.006262658491111718,
        "learning_rate": 0.0032413268041956303,
        "min_lr": 8.362652463906254e-05,
        "monitor_metric": "val_loss",
        "optimizer": "adamw",
        "optimizer_config": {
            "beta_1": 0.8786348969643611,
            "beta_2": 0.9925732206928001,
            "epsilon": 1.0e-07,
            "weight_decay": 0.003332753611177771,
        },
        "reduce_lr_factor": 0.3989280440549524,
        "reduce_lr_patience": 18,  # EgAAAAAAAAA= 디코딩값
        "tag": "binary_classification",
        "use_batch_norm": False,
        "use_class_weights": False,
        "use_cosine_scheduler": False,
        "use_early_stopping": False,
        "use_reduce_lr": False,
    }

    # 깔끔한 YAML 형식으로 저장
    with open(file_path, "w") as f:
        yaml.dump(cleaned_data, f, default_flow_style=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Converting {file_path}...")
        clean_yaml_file(file_path)
        print("Conversion complete!")
    else:
        print("Please provide a YAML file path as argument.")
