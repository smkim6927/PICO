import os
import gc
import random
import torch
import numpy as np
import wandb
import json
from tqdm import tqdm

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from torch.optim import AdamW,Adam

from optimizer.IVE import HybridPlasticityOptimizer, HybridPlasticityOptimizer64, UPGD_SophiaStyle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

from utils.metrics import ContinualLearningMetrics
from utils.dataset_loader import TextDatasetwchunk
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ 불필요한 캐시 비활성화
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Trainer:
    def __init__(
        self,
        model_name="/path/to/model",
        dataset_list=None,
        output_dir="/home/jovyan/sumin_data/saved_model/naive/",
        batch_size=4,
        seed=777,
        num_epochs=1,
        learning_rate=2e-5,
        max_length=256,
        chunk_size=64,
        debug=True,
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
        self.debug = debug

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=16,
            mixed_precision="bf16",  # fp16보다 안정적
            split_batches=False,
        )
        # ✅ Accelerator가 할당한 device 사용
        self.device = self.accelerator.device
        
        # 멀티 GPU 정보 출력
        if self.debug and self.accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"[INFO] Distributed Training Configuration")
            print(f"{'='*60}")
            print(f"Number of processes: {self.accelerator.num_processes}")
            print(f"Current process index: {self.accelerator.process_index}")
            print(f"Device for this process: {self.device}")
            print(f"Is main process: {self.accelerator.is_main_process}")
            print(f"{'='*60}\n")

        self.wandb_initialized = False
        self._collate_debug_done = False

        self.tokenizer = None
        self.model = None

    def calculate_plasticity(self, loss_before: float, loss_after: float, eps=1e-8) -> float:
        delta_loss = loss_before - loss_after
        relative_improvement = delta_loss / (loss_before + eps)
        return max(0.0, min(1.0, relative_improvement))

    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        if self.accelerator.is_main_process:
            print(f"[INFO] Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # torch_dtype=torch.float32,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        self.model.gradient_checkpointing_enable()
        
        if self.debug and self.accelerator.is_main_process:
            print(f"[DEBUG] Model dtype: {next(self.model.parameters()).dtype}")
            print(f"[DEBUG] Gradient checkpointing: enabled")                

    def prepare_data(self, txt_file):
        """단일 데이터셋 준비"""
        self._collate_debug_done = False

        dataset = TextDatasetwchunk(
            txt_file=txt_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chunk_size=self.chunk_size,
        )

        def _collate_fn(batch):
            from torch.nn.utils.rnn import pad_sequence
            pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)

            input_ids_list = [item["input_ids"] for item in batch]
            attention_mask_list = [item["attention_mask"] for item in batch]

            has_orig_labels = "labels" in batch[0]
            orig_labels_padded = None
            if has_orig_labels:
                labels_list = [item["labels"] for item in batch]
                orig_labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

            input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
            attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

            labels_padded = input_ids_padded.clone()
            labels_padded[attention_mask_padded == 0] = -100

            if self.debug and not self._collate_debug_done and self.accelerator.is_main_process:
                self._collate_debug_done = True
                valid_from_inputs = (labels_padded != -100).sum().item()
                print(f"[debug][collate] pad_id={pad_id}, batch_size={len(batch)}, "
                      f"seq_len={input_ids_padded.size(1)}, valid_labels={valid_from_inputs}")

                mismatch = ((input_ids_padded == pad_id) ^ (attention_mask_padded == 0)).sum().item()
                print(f"[debug][collate] pad-mask mismatch={mismatch}")

                if has_orig_labels:
                    valid_from_orig = (orig_labels_padded != -100).sum().item()
                    print(f"[debug][collate] valid_labels_from_original={valid_from_orig}")

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "labels": labels_padded,
            }

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=0,  # 병렬 데이터 로딩
            pin_memory=True,  # GPU 전송 속도 향상
        )
        return dataloader

    def train_on_dataset(self, dataloader, dataset_name):
        """단일 데이터셋 학습"""
        if not self.wandb_initialized and self.accelerator.is_main_process:
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
                project="UPGD", 
                config={
                    "learning_rate": self.learning_rate,
                    "model": self.model_name,
                    "dataset_name": wandb_dataset_name,
                    "epoch_num":self.num_epochs
                },
                name=f"noise_gd_{self.num_epochs}_{wandb_dataset_name}_{self.model_name}",
                # name=f"UPGD_SophiaStyle{self.num_epochs}_{wandb_dataset_name}_{self.model_name}",
                group="training"
            )

            self.wandb_initialized = True
        elif not self.wandb_initialized:
            wandb.init(mode="disabled")
            self.wandb_initialized = True
        
        print(f"[Process {self.accelerator.process_index}] Training on: {dataset_name}\n")

        # Optimizer 생성
        self.optimizer = HybridPlasticityOptimizer(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            sigma=0.001,
        )
        # self.optimizer = UPGD_SophiaStyle(
        #     params=self.model.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=0.01,
        #     sigma=0.001,
        # )

        self.model, optimizer, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, dataloader
        )

        global_step = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss_sum, epoch_plasticity_sum, step_count = 0.0, 0.0, 0

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs} [GPU {self.accelerator.process_index}]",
                disable=not self.accelerator.is_main_process,
                leave=False,
            )

            for step, batch in enumerate(progress_bar):
                global_step += 1

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # 디버깅 (첫 배치, main process만)
                if self.debug and step == 0 and self.accelerator.is_main_process:
                    valid_cnt = (labels != -100).sum().item()
                    lengths = attention_mask.sum(dim=1)
                    print(f"[debug][train] Process {self.accelerator.process_index}")
                    print(f"  labels shape={tuple(labels.shape)}, valid labels={valid_cnt}")
                    print(f"  mask lengths (min/mean/max)="
                          f"({int(lengths.min().item())}/"
                          f"{float(lengths.float().mean().item()):.1f}/"
                          f"{int(lengths.max().item())})")

                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    mismatch = ((input_ids == pad_id) ^ (attention_mask == 0)).sum().item()
                    print(f"  pad-mask mismatch={mismatch}")

                    try:
                        preview = self.tokenizer.decode(
                            input_ids[0][:80].detach().cpu().tolist(),
                            skip_special_tokens=False
                        )
                        print(f"  decode preview:\n{preview}\n")
                    except Exception as e:
                        print(f"  decode error: {e}")

                # 유효 라벨 없으면 스킵
                valid_cnt = (labels != -100).sum().item()
                if valid_cnt == 0:
                    if self.accelerator.is_main_process and self.debug:
                        print(f"[warn][train] skip batch {step}: no valid labels")
                    continue

                with self.accelerator.accumulate(self.model):
                    outputs_after = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs_after.loss

                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    loss_after_val = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).loss.item()

                loss_before_val = float(loss.item())
                plasticity_val = self.calculate_plasticity(loss_before_val, loss_after_val)

                epoch_loss_sum += loss_before_val
                epoch_plasticity_sum += plasticity_val
                step_count += 1

                if self.accelerator.is_main_process:
                    wandb.log({
                        "step": global_step,
                        "train/loss_before": loss_before_val,
                        "train/loss_after": loss_after_val,
                        "train/plasticity_step": plasticity_val,
                        "epoch": epoch + 1,
                        "train/valid_labels": valid_cnt,
                    })

                progress_bar.set_postfix({
                    "GPU": self.accelerator.process_index,
                    "Step": f"{step+1}/{len(dataloader)}",
                    "Loss": f"{loss_before_val:.4f}",
                    "Plasticity": f"{plasticity_val:.4f}",
                })

            avg_epoch_loss = epoch_loss_sum / max(step_count, 1)
            avg_epoch_plasticity = epoch_plasticity_sum / max(step_count, 1)

            if self.accelerator.is_main_process:
                print(f"\nDataset: {dataset_name}, Epoch {epoch+1}")
                print(f"  Avg Loss: {avg_epoch_loss:.4f}")
                print(f"  Avg Plasticity: {avg_epoch_plasticity:.4f}")
                wandb.log({
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/avg_plasticity": avg_epoch_plasticity,
                    "epoch": epoch + 1,
                })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # ✅ FSDP 전용 저장 (수정된 부분)
            self.accelerator.wait_for_everyone()
            
            model_save_path = os.path.join(self.output_dir, f"{dataset_name}_epoch_{epoch+1}")
            
            # 방법 1: save_model 사용 (권장)
            self.accelerator.save_model(
                self.model, 
                model_save_path,
                safe_serialization=True
            )
            
            # 토크나이저 저장
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(model_save_path)
                print(f"Model saved at: {model_save_path}")
            
            self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            wandb.finish()

    def train_across_datasets(self):
        """모든 데이터셋에 대해 순차적으로 학습"""
        for dataset_file in self.dataset_list:
            dataloader = self.prepare_data(dataset_file)
            dataset_name = os.path.basename(dataset_file).split(".")[0]
            self.train_on_dataset(dataloader, dataset_name)

if __name__ == "__main__":
    trainer = Trainer(
        model_name="/home/jovyan/sumin_data/saved_model/noise_gd/3B/new-medical-kor-dataset_epoch_5",
        dataset_list=[
            "/home/jovyan/sumin_data/cp4llm/utils/data_storage/guidline_medical.txt"
        ],
        output_dir="/home/jovyan/sumin_data/saved_model/noise_gd/3B/",
        batch_size=16,
        num_epochs=5,
        learning_rate=0.0001,
        max_length=128,
        chunk_size=64,
    )
    trainer.load_model_and_tokenizer()
    trainer.train_across_datasets()
