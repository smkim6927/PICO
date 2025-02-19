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

from optimizer.adapter import NoiseInjecter
from utils.dataset_loader import TextDatasetwchunk
# (1) KL + Huber Combined Loss
from loss_function.combined_loss import CombinedDistillationLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        "labels": labels_padded
    }

class Trainer:
    def __init__(
        self,
        model_name="/path/to/model",
        tokenizer_name="/path/to/model",
        txt_file="/home/jovyan/sumin_data/cp4llm/utils/dataloader/med_kor_dt/new-medical-kor-dataset.txt",
        output_dir="./checkpoints",
        mixed_precision="fp16",
        batch_size=4,
        seed=777,
        num_epochs=1,
        learning_rate=1e-5,
        # EMA
        ema_momentum=0.999,
        # NoiseInjecter
        noise_scale=0.001,
        adjust_noise_interval=10,
        # Dataset config
        max_length=512,
        chunk_size=64,
    ):
        set_seed(seed)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.txt_file = txt_file
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.ema_momentum = ema_momentum
        self.noise_scale = noise_scale
        self.adjust_noise_interval = adjust_noise_interval
        self.max_length = max_length
        self.chunk_size = chunk_size

        self.accelerator = Accelerator(mixed_precision="no")
        self.device = self.accelerator.device

        # (2) KL + Huber 사용
        self.continual_loss_fn = CombinedDistillationLoss(
            alpha_kl=1.0, 
            alpha_huber=1.0, 
            huber_delta=1.0, 
            temperature=1.0
        )

        self.tokenizer = None
        self.student_model = None
        self.teacher_model = None
        self.dataloader = None

        if self.accelerator.is_main_process:
            wandb.init(project="kl+huber-distillation-len1024", notes=f"dataset:{self.txt_file} model:{self.model_name} len:{self.max_length}")
        else:
            wandb.init(mode="disabled")

    def calculate_plasticity(self, loss_before: float, loss_after: float, eps=1e-8) -> float:
        delta_loss = loss_before - loss_after
        relative_improvement = delta_loss / (loss_before + eps)
        return max(0.0, min(1.0, relative_improvement))

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_student = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(self.model_name)

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.student_model = NoiseInjecter(
            model=base_student,
            noise_scale=self.noise_scale,
            adjust_noise_interval=self.adjust_noise_interval,
            output_dir=f"/home/jovyan/sumin_data/cp4llm/plots/save"
        )

    def prepare_data(self):
        dataset = TextDatasetwchunk(
            txt_file=self.txt_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chunk_size=self.chunk_size
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=my_custom_collate_fn
        )

    def _ema_update_teacher(self):
        with torch.no_grad():
            new_params = self.student_model.model.parameters()
            post_params = self.teacher_model.parameters()

            # Track EMA parameter differences
            ema_diff_sum = 0.0
            param_count = 0

            for post_param, new_param in zip(post_params, new_params):
                # Calculate parameter difference before update
                param_diff = torch.abs(post_param.data - new_param.data).mean().item()
                ema_diff_sum += param_diff
                param_count += 1

                # Perform EMA update
                post_param.data = self.ema_momentum * post_param.data + (1.0 - self.ema_momentum) * new_param.data

            # Return average parameter difference after EMA update
            return ema_diff_sum / param_count if param_count > 0 else 0.0

    
    def train(self):
        self.load_model_and_tokenizer()
        self.prepare_data()

        print(f'model:{self.model_name}')
        
        optimizer = AdamW(self.student_model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)

        self.student_model, self.teacher_model, optimizer, self.dataloader = \
            self.accelerator.prepare(
                self.student_model,
                self.teacher_model,
                optimizer,
                self.dataloader
            )

        global_step = 0

        for epoch in range(self.num_epochs):
            self.student_model.train()
            self.teacher_model.eval()

            epoch_loss_sum = 0.0
            epoch_plasticity_sum = 0.0
            step_count = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process,
                leave=False
            )

            for step, batch in enumerate(progress_bar):
                global_step += 1

                # teacher forward
                with torch.no_grad():
                    teacher_out = self.teacher_model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device)
                    )
                teacher_logits = teacher_out.logits

                # student logits (before update)
                with torch.no_grad():
                    student_out_before = self.student_model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device)
                    )
                student_logits_before = student_out_before.logits

                loss_before_val = self.continual_loss_fn(
                    student_logits=student_logits_before,
                    teacher_logits=teacher_logits,
                    mask=batch["attention_mask"].to(self.device)
                ).item()

                # student forward (train)
                student_out = self.student_model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                student_logits = student_out.logits

                total_loss = self.continual_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    mask=batch["attention_mask"].to(self.device)
                )

                self.accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

                # student logits (after update)
                with torch.no_grad():
                    student_out_after = self.student_model(
                        input_ids=batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device)
                    )
                student_logits_after = student_out_after.logits

                loss_after_val = self.continual_loss_fn(
                    student_logits=student_logits_after,
                    teacher_logits=teacher_logits,
                    mask=batch["attention_mask"].to(self.device)
                ).item()

                plasticity_val = self.calculate_plasticity(loss_before_val, loss_after_val)

                # NoiseInjecter adjust only when necessary
                if step > 0 and step % self.student_model.adjust_noise_interval == 0:
                    unwrapped_student_model = self.accelerator.unwrap_model(self.student_model)
                    unwrapped_student_model.adjust_noise_scale()
                    unwrapped_student_model.save_tracking_plots()
                
                # teacher EMA update and log EMA difference
                ema_avg_diff = self._ema_update_teacher()

                epoch_loss_sum += total_loss.item()
                epoch_plasticity_sum += plasticity_val
                step_count += 1

                if self.accelerator.is_main_process:
                    wandb.log({
                        "step": global_step,
                        "train/step_loss": total_loss.item(),
                        "train/loss_before": loss_before_val,
                        "train/loss_after": loss_after_val,
                        "train/plasticity_step": plasticity_val,
                        "train/ema_avg_diff": ema_avg_diff,
                        
                    })

                # tqdm 진행바 메시지를 업데이트
                progress_bar.set_postfix({
                    "Step": f"{step+1}/{len(self.dataloader)}",
                    "GlobalStep": global_step,
                    "Loss": f"{total_loss.item():.4f}",
                    "Before": f"{loss_before_val:.4f}",
                    "After": f"{loss_after_val:.4f}",
                    "Plasticity": f"{plasticity_val:.4f}",
                    "EMA_Diff": f"{ema_avg_diff:.4f}"
                })

            avg_epoch_loss = epoch_loss_sum / step_count
            avg_epoch_plasticity = epoch_plasticity_sum / step_count

            scheduler.step(avg_epoch_loss)

            if self.accelerator.is_main_process:
                wandb.log({
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/epoch_avg_loss": avg_epoch_loss,
                    "train/epoch_avg_plasticity": avg_epoch_plasticity,
                })
            gc.collect()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

            # NoiseInjecter 내부의 원래 모델을 가져옴
            unwrapped_student = self.accelerator.unwrap_model(self.student_model)
            
            # NoiseInjecter 객체인지 확인 후 내부 모델 저장
            if isinstance(unwrapped_student, NoiseInjecter):
                # 내부의 실제 모델 저장
                unwrapped_student.model.save_pretrained(self.output_dir)
            else:
                # 일반적인 Hugging Face 모델인 경우
                unwrapped_student.save_pretrained(self.output_dir)
            
            # 토크나이저 저장
            self.tokenizer.save_pretrained(self.output_dir)
            
            # WandB 종료
            wandb.finish()


        gc.collect()
        torch.cuda.empty_cache()

