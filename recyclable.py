import os
import gc
import random
import torch
import numpy as np
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from eval_stability import Eval  # eval.py의 Eval 클래스 import
from utils.metrics import ContinualLearningMetrics

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
        dataset_list=None,
        output_dir="/path/to/save",
        batch_size=2,
        seed=777,
        num_epochs=1,
        learning_rate=2e-5,
        max_length=512,
        chunk_size=128,
    ):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer_name = model_name
        self.dataset_list = dataset_list or []
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.wandb_initialized = False
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.domain_order = ['kor_medical', 'eng_medical', 'kor_legal', 'eng_legal']  # 도메인 순서 정의
        self.shot_types = ['zero-shot', '1shot', '3shot', '5shot']  # 평가할 shot 타입들
        self.cl_metrics = ContinualLearningMetrics(num_tasks=len(self.domain_order))

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
        from utils.data.dataset_loader import TextDatasetwchunk  # 데이터셋 로더 임포트

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
        
        # Initialize wandb for this dataset if not already done
        if not self.wandb_initialized and self.accelerator.is_main_process:
            # Process dataset name
            if 'new-medical-kor-dataset' in dataset_name:
                wandb_dataset_name = 'kor-medical'
            elif 'guidline_medical' in dataset_name:
                wandb_dataset_name = 'eng-medical'
            elif 'new-legal-kor-dataset' in dataset_name:
                wandb_dataset_name = 'kor-legal'
            elif 'eng-new-legal-dataset' in dataset_name:
                wandb_dataset_name = 'eng-legal'
            else: 
                wandb_dataset_name = dataset_name
            
            wandb.init(
                project="kd-recyclable", 
                config={
                    "learning_rate": self.learning_rate,
                    "model": self.model_name,
                    "dataset_name": wandb_dataset_name
                },
                name=f"{wandb_dataset_name}_{self.model_name}",
                group="training"
            )
            self.wandb_initialized = True
        elif not self.wandb_initialized:
            wandb.init(mode="disabled")
            self.wandb_initialized = True
        print(f"Training on dataset: {dataset_name}")
        
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
      
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
                    "Global Step": global_step,
                    "Loss Before": f"{loss_before_val:.4f}",
                    "Loss After": f"{loss_after_val:.4f}",
                    "Plasticity": f"{plasticity_val:.4f}",
                })

            avg_epoch_loss = epoch_loss_sum / step_count
            avg_epoch_plasticity = epoch_plasticity_sum / step_count
            
            print(f"Dataset: {dataset_name}, Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Avg Plasticity: {avg_epoch_plasticity:.4f}")
            gc.collect()
            torch.cuda.empty_cache()
            # scheduler.step(avg_epoch_loss)

            # 모델 저장 (데이터셋 이름에 맞게)
            model_save_path = os.path.join(self.output_dir, f"{dataset_name}_epoch_{epoch+1}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.model.save_pretrained(model_save_path)
            self.model.save_pretrained(model_save_path)

            self.tokenizer.save_pretrained(model_save_path)
            print(f"Model saved at: {model_save_path}")

        if self.accelerator.is_main_process:
            wandb.finish()
        
    def evaluate_and_update_metrics(self, evaluator, task_id_to_eval, shot_type, is_init=False):
        """단일 태스크 평가 및 CL 메트릭 업데이트 헬퍼 함수"""
        domain_to_eval = self.domain_order[task_id_to_eval]
        metrics = evaluator.evaluate(domain=domain_to_eval, shot_type=shot_type)
        accuracy = metrics["cosine_similarity"]
            
        self.cl_metrics_tracker.update_accuracy(
            task_id=task_id_to_eval,
            accuracy=accuracy,
            is_init=is_init
        )
        return metrics

    def train_across_datasets(self):
        """모든 데이터셋에 대해 순차적으로 학습하고 각 단계별로 평가 수행"""
        from utils.metrics import calculate_metrics
    
        evaluator = Eval(
            model_name=self.model_name,
            accelerator=self.accelerator,
            batch_size=1,
            max_length=self.max_length,
            num_epochs=1
        )
        
        criterion = torch.nn.CrossEntropyLoss()  # Fisher Information 계산용
        
        for task_id, dataset_file in enumerate(self.dataset_list):
            # 1. 현재 데이터셋으로 학습 수행
            dataloader = self.prepare_data(dataset_file)
            dataset_name = os.path.basename(dataset_file).split(".")[0]
            current_domain = self.domain_order[task_id]
            
            print(f"\n=== Training on {current_domain} (Task {task_id+1}) ===")
            self.train_on_dataset(dataloader, dataset_name)

            # 2. 평가용 wandb 초기화
            if self.accelerator.is_main_process:
                wandb.init(
                    project="kd-recyclable-eval",
                    name=f"task_{task_id}_{current_domain}",
                    group="evaluation",
                    reinit=True,
                    config={
                        "trained_on": current_domain,
                        "task_id": task_id,
                        "model": self.model_name,
                    }
                )
            
            # 3. 현재까지 학습된 모든 도메인에 대해 다양한 shot으로 평가
            print(f"\n=== Evaluating after {current_domain} training ===")
            
            initial_metrics = self.evaluate_and_update_metrics(evaluator, task_id, 'zero-shot', is_init=True)
            
            for domain in self.domain_order[:task_id+1]:
                domain_metrics = {}

                for shot_type in self.shot_types:
                    print(f"Evaluating {domain} with {shot_type}")
                    metrics = evaluator.evaluate(domain=domain, shot_type=shot_type)
                    # 메트릭 로깅
                    wandb.log({
                        "metrics/loss": metrics["loss"],
                        "metrics/rouge1": metrics["rouge1"],
                        "metrics/rouge2": metrics["rouge2"],
                        "metrics/rougeL": metrics["rougeL"],
                        "metrics/bleu": metrics["bleu"],
                        "metrics/meteor": metrics["meteor"],
                        "metrics/f1": metrics["f1"],
                        "metrics/r2": metrics["r2"],
                        "metrics/exact_match": metrics["exact_match"],
                        "metrics/groundedness": metrics["groundedness"],
                        "metrics/cosine_similarity": metrics["cosine_similarity"],
                        "metrics/jaccard_similarity": metrics["jaccard_similarity"]
                    })

                    if shot_type == 'zero-shot':
                        accuracy = metrics["cosine_similarity"]
                        self.cl_metrics.update_accuracy(
                            task_id=self.domain_order.index(domain),
                            accuracy=accuracy,
                            is_init=(task_id == self.domain_order.index(domain))
                        )
            
            # 4. CL 메트릭, Fisher Information, EMA Drift 계산
            all_metrics = self.cl_metrics.compute_all_metrics(
                model=self.model,
                current_task=task_id,
                dataloader=dataloader,  # Fisher Information 계산용
                criterion=criterion
            )
            
            avgf = all_metrics.get('AvgF', 0.0)
            bwt = all_metrics.get('BWT', 0.0)
            delta_fisher = all_metrics.get('DeltaFisher', 0.0)
            ema_drift = all_metrics.get('EMA_Drift', 0.0)
            
            # 5. 메트릭 로깅
            if self.accelerator.is_main_process:
                wandb.log({
                    'current_task': task_id,
                    'domain': current_domain,
                    'avgf': avgf,
                    'bwt': bwt,
                    'delta_fisher': delta_fisher,
                    'ema_drift': ema_drift,
                    'fisher_info_history': len(self.cl_metrics.fisher_history)
                })
                
                # 메트릭 출력
                print(f"\n=== Metrics for {current_domain} ===")
                print(f"AvgF: {avgf:.4f}")
                print(f"BWT: {bwt:.4f}")
                print(f"Delta Fisher: {delta_fisher:.4f}")
                print(f"EMA Drift: {ema_drift:.4f}")
                
                wandb.finish()
            
            # 6. 중간 체크포인트 및 메트릭 저장
            checkpoint_path = os.path.join(
                self.output_dir,
                f"checkpoint_after_{current_domain}"
            )
            self.model.save_pretrained(checkpoint_path)
            
            # 메트릭 저장
            metrics_path = os.path.join(checkpoint_path, "metrics.json")
            metrics_data = {
                "task_id": task_id,
                "domain": current_domain,
                "avgf": avgf,
                "bwt": bwt,
                "delta_fisher": delta_fisher,
                "ema_drift": ema_drift,
                "fisher_info_size": len(self.cl_metrics.fisher_history)
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            print(f"Checkpoint and metrics saved at: {checkpoint_path}")


if __name__ == "__main__":
    trainer = Trainer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        dataset_list=[
            "/home/jovyan/sumin_data/cp4llm/utils/dataloader/med_kor_dt/new-medical-kor-dataset.txt",
            "/home/jovyan/sumin_data/cp4llm/utils/dataloader/med_eng_dt/guidline_medical.txt",
            "/home/jovyan/sumin_data/cp4llm/utils/dataloader/legal_kor_dt/new-legal-kor-dataset.txt",
            "/home/jovyan/sumin_data/cp4llm/utils/dataloader/law_datasets/eng-new-legal-dataset.txt"
        ],
        output_dir="/home/jovyan/sumin_data/saved_model/recyclable/",
        batch_size=8,
        num_epochs=1,
        learning_rate=2e-5,
    )
    
    trainer.load_model_and_tokenizer()
    trainer.train_across_datasets()
