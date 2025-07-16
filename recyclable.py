import os, json
import gc
import random
import torch
import numpy as np
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from eval_stability import Eval
from utils.metrics import ContinualLearningMetrics, calculate_metrics
from utils.domain_map import domain_info

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def my_custom_collate_fn(batch):
    def safe_tensor_conversion(item, dtype=torch.long):
        if isinstance(item, torch.Tensor):
            return item.clone().detach().to(dtype=dtype)
        elif isinstance(item, (list, np.ndarray)):
            return torch.tensor(item, dtype=dtype)
        else:
            return torch.tensor([item], dtype=dtype)
    
    input_ids_list = [safe_tensor_conversion(item["input_ids"]) for item in batch]
    attention_mask_list = [safe_tensor_conversion(item["attention_mask"]) for item in batch]
    labels_list = [safe_tensor_conversion(item["labels"]) for item in batch]

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

        self.domain_order = ['kor_medical', 'eng_medical', 'kor_legal', 'eng_legal']
        self.shot_types = ['zero-shot', '1shot', '3shot', '5shot']
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

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def prepare_data(self, txt_file):
        """단일 데이터셋 준비"""
        from utils.dataset_loader import TextDatasetwchunk

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

    def train_on_dataset(self, dataloader, dataset_name, task_id):
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
                name=f"{wandb_dataset_name}_{self.model_name.replace('/', '_')}",
                group="training"
            )
            self.wandb_initialized = True
        elif not self.wandb_initialized:
            wandb.init(mode="disabled")
            self.wandb_initialized = True
            
        print(f"Training on dataset: {dataset_name}")
        if self.accelerator.is_main_process:
            wandb.log({"current_task/id": task_id, "current_task/name": dataset_name})
            
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
      
        # Accelerator로 준비
        self.model, optimizer, dataloader = self.accelerator.prepare(self.model, optimizer, dataloader)
        
        global_step = 0
        for epoch in range(self.num_epochs):
            self.model.train()
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

                with torch.no_grad():
                    outputs_before_update = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss_before_val = outputs_before_update.loss.item()

                # Forward pass (학습 중)
                outputs_after_update = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss_after_val = outputs_after_update.loss.item()

                self.accelerator.backward(outputs_after_update.loss)
                optimizer.step()
                optimizer.zero_grad()

                # Plasticity 계산
                plasticity_val = self.calculate_plasticity(loss_before_val, loss_after_val)

                epoch_loss_sum += loss_after_val
                epoch_plasticity_sum += plasticity_val
                step_count += 1

                if self.accelerator.is_main_process:
                    wandb.log({
                        "step": global_step,
                        f"train_task_{task_id}/loss": loss_after_val,  # .item() 제거
                        f"train_task_{task_id}/plasticity": plasticity_val,
                    })

                progress_bar.set_postfix({
                    "Step": f"{step+1}/{len(dataloader)}",
                    "Global Step": global_step,
                    "Loss Before": f"{loss_before_val:.4f}",
                    "Loss After": f"{loss_after_val:.4f}",
                    "Plasticity": f"{plasticity_val:.4f}",
                })

            avg_epoch_loss = epoch_loss_sum / step_count if step_count > 0 else 0.0
            avg_epoch_plasticity = epoch_plasticity_sum / step_count if step_count > 0 else 0.0
            
            print(f"Dataset: {dataset_name}, Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Avg Plasticity: {avg_epoch_plasticity:.4f}")
            gc.collect()
            torch.cuda.empty_cache()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                model_save_path = os.path.join(self.output_dir, f"{dataset_name}_epoch_{epoch+1}")
                os.makedirs(model_save_path, exist_ok=True)
                
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)
                print(f"Model saved at: {model_save_path}")

        if self.accelerator.is_main_process:
            wandb.finish()
            self.wandb_initialized = False  # 재설정
        
    def evaluate_and_update_metrics(self, evaluator, task_id_to_eval, shot_type, is_init=False):
        """단일 태스크 평가 및 CL 메트릭 업데이트 헬퍼 함수"""
        
        domain_to_eval = self.domain_order[task_id_to_eval]
        metrics = evaluator.evaluate(domain=domain_to_eval, shot_type=shot_type)
        accuracy = metrics["cosine_similarity"]
        
        self.cl_metrics.update_accuracy(
            task_id=task_id_to_eval,
            accuracy=accuracy,
            is_init=is_init
        )
        return metrics

    def train_across_datasets(self):
        """모든 데이터셋에 대해 순차적으로 학습하고 각 단계별로 평가 수행"""
        
        criterion = torch.nn.CrossEntropyLoss()
        
        for task_id, dataset_file in enumerate(self.dataset_list):
            # 1. 현재 데이터셋으로 학습 수행
            dataloader = self.prepare_data(dataset_file)
            dataset_name = os.path.basename(dataset_file).split(".")[0]
            current_domain = self.domain_order[task_id]
            
            print(f"\n=== Training on {current_domain} (Task {task_id+1}) ===")
            self.train_on_dataset(dataloader, dataset_name, task_id)

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
            
            # 3. CL 메트릭 계산 및 로깅
            if self.accelerator.is_main_process:
                try:
                    all_metrics = self.cl_metrics.compute_all_metrics(
                        model=self.accelerator.unwrap_model(self.model),
                        current_task=task_id + 1,
                        dataloader=dataloader,
                        criterion=criterion
                    )
                    
                    avgf = all_metrics.get('AvgF', 0.0)
                    bwt = all_metrics.get('BWT', 0.0)
                    delta_fisher = all_metrics.get('DeltaFisher', 0.0)
                    ema_drift = all_metrics.get('EMA_Drift', 0.0)
                    
                    wandb.log({
                        'current_task': task_id,
                        'domain': current_domain,
                        'avgf': avgf,
                        'bwt': bwt,
                        'delta_fisher': delta_fisher,
                        'ema_drift': ema_drift,
                        'fisher_info_history': len(self.cl_metrics.fisher_history)
                    })
                    
                    print(f"\n=== Metrics for {current_domain} ===")
                    print(f"AvgF: {avgf:.4f}")
                    print(f"BWT: {bwt:.4f}")
                    print(f"Delta Fisher: {delta_fisher:.4f}")
                    print(f"EMA Drift: {ema_drift:.4f}")
                    
                except Exception as e:
                    print(f"Error computing CL metrics: {e}")
                    
                wandb.finish()
            
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_after_{current_domain}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                
                # 메트릭 저장
                try:
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
                except:
                    print("Could not save metrics.json")
                
                print(f"Checkpoint saved at: {checkpoint_path}")

if __name__ == "__main__":
    trainer = Trainer(
        model_name="openai-community/gpt2-large",
        dataset_list=[
            "/home/infonet/sumin/cp4gm/utils/data_storage/guidline_medical.txt"
        ],
        output_dir="/home/infonet/sumin/saved_model/recyclable/",
        batch_size=1,
        num_epochs=5,
        learning_rate=2e-5,
    )
    trainer.load_model_and_tokenizer()
    trainer.train_across_datasets()
