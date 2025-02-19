import os
import gc
import random
import torch
import numpy as np
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader


# Seed 설정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Collate 함수 정의
def my_custom_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }


class Trainer:
    def __init__(
        self,
        model_name="/path/to/model",
        tokenizer_name="/path/to/model",
        dataset_list=None,
        output_dir="/home/jovyan/sumin_data/saved_model/naive/",
        batch_size=4,
        seed=777,
        num_epochs=1,
        learning_rate=1e-5,
        max_length=512,
        chunk_size=64,
    ):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.dataset_list = dataset_list or []
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.chunk_size = chunk_size

        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.tokenizer = None
        self.model = None

        if self.accelerator.is_main_process:
            wandb.init(project="cross-entropy-plasticity", config={"learning_rate": learning_rate})
        else:
            wandb.init(mode="disabled")

    def calculate_plasticity(self, loss_before: float, loss_after: float, eps=1e-8) -> float:
        delta_loss = loss_before - loss_after
        relative_improvement = delta_loss / (loss_before + eps)
        return max(0.0, min(1.0, relative_improvement))
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 (Teacher와 Student 동일한 초기화)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def prepare_data(self, txt_file):
        """단일 데이터셋 준비"""
        from utils.dataset_loader import TextDatasetwchunk  # 데이터셋 로더 임포트

        dataset = TextDatasetwchunk(
            txt_file=txt_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chunk_size=self.chunk_size,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=my_custom_collate_fn,
        )
        
        return dataloader

    def train_on_dataset(self, dataloader, dataset_name):
        """단일 데이터셋 학습"""
        
        print(f"Training on dataset: {dataset_name}")
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

        # Accelerator로 준비 (모델, 옵티마이저, 데이터로더)
        self.model, optimizer, dataloader = \
            self.accelerator.prepare(
                self.model,
                optimizer,
                dataloader,
            )

        global_step = 0

        for epoch in range(self.num_epochs):
            epoch_loss_sum = 0.0
            epoch_plasticity_sum = 0.0
            step_count = 0

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process,
                leave=False,
            )

            for step, batch in enumerate(progress_bar):
                global_step += 1

                # Forward pass 및 손실 계산 (Cross-Entropy Loss 사용)
                with torch.no_grad():
                    outputs_before_update = self.model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
                        labels=batch["labels"].to(self.device),
                    )
                    loss_before_val = outputs_before_update.loss.item()

                # Forward pass (학습 중)
                outputs_after_update = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                
                loss_after_val = outputs_after_update.loss.item()

                # 역전파 및 가중치 업데이트
                outputs_after_update.loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Plasticity 계산
                plasticity_val = self.calculate_plasticity(loss_before_val, loss_after_val)

                epoch_loss_sum += loss_after_val
                epoch_plasticity_sum += plasticity_val
                step_count += 1

                # WandB 로깅 (각 스텝별 정보 기록)
                if self.accelerator.is_main_process:
                    wandb.log({
                        "step": global_step,
                        "train/loss_before": loss_before_val,
                        "train/loss_after": loss_after_val,
                        "train/plasticity_step": plasticity_val,
                    })

                # tqdm 진행바 업데이트
                progress_bar.set_postfix({
                    "Step": f"{step+1}/{len(dataloader)}",
                    "GlobalStep": global_step,
                    "Loss Before": f"{loss_before_val:.4f}",
                    "Loss After": f"{loss_after_val:.4f}",
                    "Plasticity": f"{plasticity_val:.4f}",
                })

            avg_epoch_loss = epoch_loss_sum / step_count
            avg_epoch_plasticity = epoch_plasticity_sum / step_count
            
            print(f"Dataset: {dataset_name}, Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Avg Plasticity: {avg_epoch_plasticity:.4f}")
            
            scheduler.step(avg_epoch_loss)

            # 모델 저장 (데이터셋 이름에 맞게)
            model_save_path = os.path.join(self.output_dir, f"{dataset_name}_epoch_{epoch+1}")
            self.model.save_pretrained(model_save_path)
            print(f"Model saved at: {model_save_path}")

    def train_across_datasets(self):
        """모든 데이터셋에 대해 순차적으로 학습"""
        
        for dataset_file in self.dataset_list:
            dataloader = self.prepare_data(dataset_file)  # 각 데이터셋 준비
            
            dataset_name = os.path.basename(dataset_file).split(".")[0]  # 파일명에서 이름 추출
            
            # 단일 데이터셋 학습 실행
            self.train_on_dataset(dataloader, dataset_name)

if __name__ == "__main__":
    trainer = Trainer(
        model_name="EleutherAI/gpt-neo-125m",
        tokenizer_name="EleutherAI/gpt-neo-125m",
        
         # 순차적으로 학습할 데이터셋 경로 리스트 추가 
         dataset_list=[
             "/home/jovyan/sumin_data/cp4llm/utils/dataloader/med_kor_dt/new-medical-kor-dataset.txt",
             "/home/jovyan/sumin_data/cp4llm/utils/dataloader/med_eng_dt/guidline_medical.txt",
             "/home/jovyan/sumin_data/cp4llm/utils/dataloader/legal_kor_dt/new-legal-kor-dataset.txt",
             "/home/jovyan/sumin_data/cp4llm/utils/dataloader/law_datasets/eng-new-legal-dataset.txt"
         ],
         
         output_dir="/home/jovyan/sumin_data/saved_model/naive/",
         batch_size=16,
         num_epochs=3,
         learning_rate=1e-5,
    )
    
    trainer.load_model_and_tokenizer()
    trainer.train_across_datasets()
