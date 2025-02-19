import argparse
import yaml
from utils.train import Trainer

def parse_config(config_path):
    """YAML 파일을 로드해 Python dict로 반환"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train a language model using a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # 1) YAML 파싱
    config = parse_config(args.config)

    # 2) 필요한 설정값 가져오기
    model_name = config["model_name"]
    tokenizer_name = config.get("tokenizer_name", model_name)  # tokenizer_name이 없으면 model_name을 사용
    txt_file = config.get("txt_file", "/home/jovyan/sumin_data/cp4llm/utils/dataloader/law_datasets/new-legal-dataset.txt")
    output_dir = config.get("output_dir", "/home/jovyan/sumin_data/saved_model/non-label/")
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("batch_size", 8)
    max_length = config.get("max_length", 1024)
    mixed_precision = config.get("mixed_precision", "fp16")

    # 3) Trainer 인스턴스 생성
    trainer = Trainer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        txt_file=txt_file,
        output_dir=output_dir,
        mixed_precision=mixed_precision,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # 4) 학습 실행
    trainer.train()

if __name__ == "__main__":
    main()
