# utils/trainer.py

import os, time, torch, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import wandb
from typing import Dict, Any, Optional

from utils.dataset_loader import TextDatasetwchunk
from utils.domain_map import get_collate_fn
from loss_function.combined_loss import CombinedselfDLoss
from loss_function.distillation import KLDistillationLoss  
from loss_function.cross_entropy import CrossEntropyLossWithMask
from torch import optim
from utils.metrics import ContinualLearningMetrics
from optimizer.adapter import ImprovedNoiseInjecter
from utils.wandb_metrics_logger import RealTimeWandBLogger
from utils.metrics import calculate_plasticity

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, args):
        """완전한 학습 파이프라인을 위한 Trainer 초기화"""
        set_seed(args.seed)

        args.use_wandb = True
        args.use_noise_injection = True
        args.self_distill = True
        
        print("🚀 FORCED TRAINER CONFIGURATION:")
        print(f"  ✓ WandB: {args.use_wandb}")
        print(f"  ✓ Noise Injection: {args.use_noise_injection}")
        print(f"  ✓ Self-Distillation: {args.self_distill}")

        # ─────── Argument mapping ──────────────────────────────────────
        self.args = args
        self.model_name      = args.model_name
        self.tokenizer_name  = args.tokenizer_name or args.model_name
        self.dataset_files   = args.dataset_files
        self.output_dir      = args.output_dir
        self.mixed_precision = args.mixed_precision
        self.cpu             = args.cpu
        self.batch_size      = args.batch_size
        self.num_epochs      = args.num_epochs
        self.learning_rate   = args.learning_rate
        self.max_length      = args.max_length
        self.chunk_size      = args.chunk_size

        # ─────── Accelerator ───────────────────────────────────────────
        self.accelerator = Accelerator(
            cpu=self.cpu,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1)
        )
        self.device = self.accelerator.device

        # ─────── Place-holders ─────────────────────────────────────────
        self.tokenizer, self.post_model, self.previous_model = None, None, None
        self.dataloader, self.optimizer, self.scheduler = None, None, None
        self.loss_fn = None

        # ─────── Metrics & logging ────────────────────────────────────
        self._init_metrics_and_logging()
        print(f"Trainer initialized on device: {self.device}")

    # Utility Methods
    def _safe_adjust_noise_scale(self, model):
        """DDP 래핑된 모델에서 안전하게 adjust_noise_scale 호출"""
        # DDP로 래핑된 경우 module 속성으로 원래 모델 접근
        if hasattr(model, 'module'):
            real_model = model.module
        else:
            real_model = model
            
        if hasattr(real_model, 'adjust_noise_scale'):
            real_model.adjust_noise_scale()
            return True
        return False

    def _get_unwrapped_model(self, model):
        """DDP 또는 다른 래퍼에서 원래 모델을 추출"""
        if hasattr(model, 'module'):
            return model.module
        return model

    # Logging & Metrics
    def _init_metrics_and_logging(self):
        self.metrics_tracker = ContinualLearningMetrics(
            num_tasks=getattr(self.args, 'num_tasks', 1),
        )

        # 🔥 WANDB 초기화는 main process에서만 수행
        if self.accelerator.is_main_process:
            print("🔥 FORCING WandB Logger activation (MAIN PROCESS ONLY)...")
            self.wandb_logger = RealTimeWandBLogger(
                project_name=getattr(self.args, 'wandb_project', 'cp4llm_forced_training'),
                run_name=getattr(self.args, 'wandb_run_name', f'forced_run_{os.getpid()}'),
                config=vars(self.args),
                log_frequency=getattr(self.args, 'log_frequency', 50),
                enable_async_logging=getattr(self.args, 'async_logging', True)
            )
            self.wandb_logger.create_continual_learning_dashboard()
            print("✅ WandB Logger FORCED activation complete (MAIN PROCESS)!")
        else:
            print("📌 Skipping WandB initialization on worker process")
            self.wandb_logger = None

        self.global_step, self.current_epoch = 0, 0
        self.loss_history, self.best_loss = [], float('inf')

    # Model & Tokenizer
    def _requires_previous_model(self) -> bool:
        """해당 loss type이 previous model을 요구하는지"""
        distil_losses = [
            "kl_distillation", "mse_distillation", "weighted_combined"
        ]
        return getattr(self.args, 'loss_type', 'ce') in distil_losses

    def _estimate_model_size(self, model):
        """모델의 파라미터 수를 추정하여 크기 반환"""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return float(total_params)

    def load_model_and_tokenizer(self):
        print(f"Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading post model: {self.model_name}")
        self.post_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=getattr(self.args, 'load_in_8bit', False),
            device_map=getattr(self.args, 'device_map', None),
            torch_dtype=torch.float16 if getattr(self.args, 'use_fp16', False) else None,
            trust_remote_code=True
        )
        
        print("🔥 Stabilizing model parameters...")
        with torch.no_grad():
            for name, param in self.post_model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ Found NaN/Inf in {name}, reinitializing...")
                    param.data = torch.randn_like(param.data) * 0.02
                # 매우 큰 값들 클리핑
                param.data = torch.clamp(param.data, min=-10, max=10)
        
        print("✅ Model parameters stabilized")
            
        # 모델 크기 추정
        model_size = self._estimate_model_size(self.post_model)
        print(f"Estimated model size: {model_size/1e9:.2f}B parameters")

        self.post_model = ImprovedNoiseInjecter(
            model=self.post_model,
            model_size=model_size,
            ema_decay=getattr(self.args, 'ema_decay', 0.99),
            adjust_noise_interval=getattr(self.args, 'adjust_noise_interval', 100),
            output_dir=os.path.join(self.output_dir, "noise_plots"),
            seed=getattr(self.args, 'noise_seed', 777)
        )
        
        # 메모리 사용량 정보 출력
        memory_info = self.post_model.get_memory_usage_info()
        print("✅ ImprovedNoiseInjecter FORCED activation complete!")
        print(f"  Strategy: {memory_info['strategy']}")
        print(f"  Total params: {memory_info['total_params']}")
        print(f"  EMA coverage: {memory_info['ema_coverage']}")
        print(f"  GPU memory: {memory_info['gpu_memory']}")

        print("Models loaded successfully")

    # Data preprocessing
    def prepare_data(self):
        print(f"Creating dataset type: {getattr(self.args, 'dataset_type', 'text_chunk')}")
        dataset = TextDatasetwchunk(
            txt_file=self.dataset_files,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chunk_size=self.chunk_size,
        )

        collate_fn = get_collate_fn(getattr(self.args, 'collate_type', 'smart'))
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=getattr(self.args, 'num_workers', 0),
            pin_memory=True,
            drop_last=True
        )
        print(f"DataLoader ready — {len(self.dataloader)} batches/epoch")

    # Optimizer & Scheduler
    def setup_optimizer_and_scheduler(self):
        print("Setting up optimizer & scheduler …")
        self.optimizer = optim.Adam(
            self.post_model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
            eps=1e-08
        )

        total_steps = len(self.dataloader) * self.num_epochs
        warmup_steps = int(total_steps * getattr(self.args, 'warmup_ratio', 0.03))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=0
        )
        print(f"Optimizer: {type(self.optimizer).__name__} (Adam for noise injection) | "
              f"Scheduler: {type(self.scheduler).__name__} | "
              f"Steps: {total_steps} (warmup {warmup_steps})")

    # Loss-function handling
    def _prepare_loss_kwargs(self, loss_type: str) -> Dict[str, Any]:
        """Loss 타입별 파라미터 매핑"""
        args, kw = self.args, {}
        if loss_type in ("ce", "CrossEntropyLossWithMask"):
            kw = {"ignore_index": getattr(args, 'ignore_index', -100)}

        elif loss_type == "KLDistillationLoss":
            kw = {"temperature": getattr(args, 'distillation_temperature', 3.0),
                  "alpha": getattr(args, 'distillation_alpha', 0.7)}

        elif loss_type in ("CombinedDistillationLoss", "combined_self_distillation"):
            # 🔥 ImprovedNoiseInjecter에서 자동으로 계산된 가중치 사용
            unwrapped_model = self._get_unwrapped_model(self.post_model)
            if hasattr(unwrapped_model, 'kl_weight') and hasattr(unwrapped_model, 'huber_weight'):
                print(f"🔍 Using model-computed weights: KL={unwrapped_model.kl_weight:.2f}, Huber={unwrapped_model.huber_weight:.2f}")
                kw = {
                    "alpha_kl": unwrapped_model.kl_weight,
                    "alpha_huber": unwrapped_model.huber_weight,
                    "temperature": getattr(args, 'distillation_temperature', 3.0),
                    "huber_delta": getattr(args, 'huber_delta', 1.0),
                    "self_distill": True
                }
            else:
                # 🔥 fallback: 모델 크기 기반 기본값
                model_size = getattr(self.args, 'model_size', 1e9)
                if model_size < 3e9:
                    alpha_kl, alpha_huber = 0.8, 0.2
                elif model_size < 7e9:
                    alpha_kl, alpha_huber = 0.6, 0.4
                else:
                    alpha_kl, alpha_huber = 0.4, 0.6
                
                kw = {
                    "alpha_kl": alpha_kl,
                    "alpha_huber": alpha_huber,
                    "temperature": getattr(args, 'distillation_temperature', 3.0),
                    "huber_delta": getattr(args, 'huber_delta', 1.0),
                    "self_distill": True
                }
                print(f"🔍 Using size-based fallback weights: KL={alpha_kl:.2f}, Huber={alpha_huber:.2f}")
        
        return kw

    def setup_loss_function(self):
        """Loss 함수 설정 - 모듈별 직접 초기화"""
        loss_type = getattr(self.args, 'loss_type', 'ce')
        print(f"Setting up loss function: {loss_type}")

        kw = self._prepare_loss_kwargs(loss_type)
        print(f"🔍 Loss kwargs: {kw}")  # 🔥 kwargs 출력해서 확인
        
        try:
            if loss_type in ("combined_distillation", "combined_self_distillation", "CombinedDistillationLoss"):
                self.loss_fn = CombinedselfDLoss(**kw)
                print("✓ CombinedselfDLoss initialized")
                # 🔥 생성된 loss 함수의 self_distill 값 확인
                print(f"🔍 Created loss function self_distill: {self.loss_fn.self_distill}")
            elif loss_type in ("ce", "cross_entropy", "label_smoothing", "CrossEntropyLossWithMask"):
                self.loss_fn = CrossEntropyLossWithMask(**kw)
                print("✓ CrossEntropyLossWithMask initialized")
            elif loss_type in ("kl_distillation", "KLDistillationLoss"):
                self.loss_fn = KLDistillationLoss(**kw)
                print("✓ KLDistillationLoss initialized")
            else:
                print(f"⚠️ Unsupported loss_type '{loss_type}', falling back to CrossEntropyLossWithMask")
                fallback_kw = {"ignore_index": getattr(self.args, 'ignore_index', -100)}
                self.loss_fn = CrossEntropyLossWithMask(**fallback_kw)
                
            if hasattr(self.loss_fn, 'get_loss_info'):
                info = self.loss_fn.get_loss_info()
                for k, v in info.items():
                    print(f"  {k}: {v}")
                    
        except Exception as e:
            print(f"❌ Failed to create loss function: {e}")
            raise

    # Accelerator Preparation
    def prepare_training(self):
        print("Preparing components via Accelerator …")
        comps = [self.post_model, self.optimizer, self.dataloader]
        if self.previous_model: comps.append(self.previous_model)
        if self.scheduler: comps.append(self.scheduler)

        prepared = self.accelerator.prepare(*comps)
        self.post_model, self.optimizer, self.dataloader = prepared[:3]
        idx = 3
        if self.previous_model:
            self.previous_model = prepared[idx]; idx += 1
        if self.scheduler:
            self.scheduler = prepared[idx]
        print("All components wrapped for distributed training")

    # Training loop
    def training_step(self, batch, step_in_epoch):
        self.optimizer.zero_grad()

        # 최적화 전 손실 계산 (loss_before) - Plasticity 계산용
        with torch.no_grad():
            with self.accelerator.autocast():
                post_out_before = self.post_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True
                )
                print(f"Labels shape: {batch['labels'].shape}")
                print(f"Labels unique values: {torch.unique(batch['labels'])}")
                print(f"Attention mask sum: {batch['attention_mask'].sum()}")

                previous_out_before = None
                if self._requires_previous_model() and self.previous_model is not None:
                    previous_out_before = self.previous_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True
                    )

                # CombinedselfDLoss에 맞는 인터페이스로 호출
                if isinstance(self.loss_fn, CombinedselfDLoss):
                    loss_before = self.loss_fn(
                        post_logits=post_out_before.logits,
                        prev_logits=previous_out_before.logits if previous_out_before else None,
                        mask=batch.get("attention_mask", None),
                        eval_mode=True
                    ).item()
                else:
                    loss_dict_before = self.loss_fn(
                        post_outputs=post_out_before,
                        previous_outputs=previous_out_before,
                        labels=batch["labels"]
                    )
                    loss_before = loss_dict_before["loss"].item()

        # 정상적인 forward 및 backward
        with self.accelerator.autocast():
            post_out = self.post_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_hidden_states=True
            )
            print(f"Logits mean: {post_out.logits.mean()}")
            print(f"Logits std: {post_out.logits.std()}")
            previous_out = None
            if self._requires_previous_model() and self.previous_model is not None:
                with torch.no_grad():
                    previous_out = self.previous_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True
                    )

            # CombinedselfDLoss에 맞는 인터페이스로 호출
            if isinstance(self.loss_fn, CombinedselfDLoss):
                loss = self.loss_fn(
                    post_logits=post_out.logits,
                    prev_logits=previous_out.logits if previous_out else None,
                    mask=batch.get("attention_mask", None),
                    eval_mode=False
                )
                loss_dict = {"loss": loss}
            else:
                loss_dict = self.loss_fn(
                    post_outputs=post_out,
                    previous_outputs=previous_out,
                    labels=batch["labels"]
                )
                loss = loss_dict["loss"]

        self.accelerator.backward(loss)
        if getattr(self.args, 'max_grad_norm', 0) > 0:
            self.accelerator.clip_grad_norm_(self.post_model.parameters(),0.1)  # 🔥 1.0에서 0.1로 강화
        self.optimizer.step()
        if self.scheduler: self.scheduler.step()

        # 최적화 후 손실 계산 (loss_after) - Plasticity 계산용
        with torch.no_grad():
            with self.accelerator.autocast():
                post_out_after = self.post_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True
                )

                previous_out_after = None
                if self._requires_previous_model() and self.previous_model is not None:
                    previous_out_after = self.previous_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True
                    )

                if isinstance(self.loss_fn, CombinedselfDLoss):
                    loss_after = self.loss_fn(
                        post_logits=post_out_after.logits,
                        prev_logits=previous_out_after.logits if previous_out_after else None,
                        mask=batch.get("attention_mask", None),
                        eval_mode=True
                    ).item()
                else:
                    loss_dict_after = self.loss_fn(
                        post_outputs=post_out_after,
                        previous_outputs=previous_out_after,
                        labels=batch["labels"]
                    )
                    loss_after = loss_dict_after["loss"].item()

        # Plasticity 계산
        plasticity = calculate_plasticity(loss_before, loss_after)
        loss_dict["plasticity"] = plasticity

        # 🔥 안전한 노이즈 조정 호출
        if self.accelerator.is_main_process:
            print(f"🔥 Step {self.global_step}: Forcing noise scale adjustment...")
        
        success = self._safe_adjust_noise_scale(self.post_model)
        
        if self.accelerator.is_main_process:
            if success:
                print(f"✅ Noise adjustment forced at step {self.global_step}")
            else:
                print(f"⚠️ Noise adjustment not available at step {self.global_step}")

        # Metrics
        self.global_step += 1
        self.loss_history.append(loss.item())
        return loss_dict

    def train_epoch(self, epoch):
        """에포크별 학습"""
        self.post_model.train()
        epoch_losses = []
        
        if self.accelerator.is_main_process:
            from tqdm import tqdm
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", dynamic_ncols=True, leave=False)
        else:
            progress_bar = self.dataloader

        for step, batch in enumerate(progress_bar):
            # 학습 스텝 실행
            loss_dict = self.training_step(batch, step)
            loss_val = loss_dict["loss"].item()
            epoch_losses.append(loss_val)
            plastic_val = float(loss_dict.get("plasticity", 0.0))

            # tqdm progress bar에 실시간 표시 (main process에서만)
            if self.accelerator.is_main_process:
                progress_bar.set_postfix({
                    "Loss": f"{loss_val:.4f}",
                    "Plasticity": f"{plastic_val:.4f}",
                    "WandB": "✅" if self.wandb_logger else "❌",
                    "Noise": "✅"
                })

            if step % getattr(self.args, 'log_interval', 100) == 0 and self.accelerator.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_loss = np.mean(epoch_losses[-100:])

                # 로깅 정보 구성
                log_info = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": self.global_step,
                    "loss": loss_val,
                    "avg_loss": avg_loss,
                    "plasticity": plastic_val,
                    "learning_rate": current_lr,
                    "batch_size": batch["input_ids"].size(0),
                    "process_rank": self.accelerator.process_index,
                    "num_processes": self.accelerator.num_processes
                }

                for key, value in loss_dict.items():
                    if key not in ("loss", "plasticity"):
                        log_info[key] = value.item() if torch.is_tensor(value) else value

                # 🔥 WANDB 로깅은 main process에서만
                if self.wandb_logger:
                    print(f"🔥 Step {step}: Forcing WandB logging (MAIN PROCESS)...")
                    quick_metrics = self._compute_quick_metrics(loss_val)
                    quick_metrics['plasticity'] = plastic_val
                    quick_metrics['forced_logging'] = True
                    
                    self.wandb_logger.log_step_metrics(
                        metrics=quick_metrics,
                        training_metrics=log_info
                    )
                    print(f"✅ WandB logging forced at step {step}")
                
                # 콘솔 출력
                print(f"🔥 FORCED MODE - Rank {self.accelerator.process_index} - "
                      f"Epoch {epoch+1}, Step {step}, Global Step {self.global_step}, "
                      f"Loss: {loss_val:.4f}, Plasticity: {plastic_val:.4f}, "
                      f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

        return np.mean(epoch_losses)

    def _compute_quick_metrics(self, current_loss):
        """빠른 메트릭 계산"""
        quick_metrics = {}
        
        # Plasticity 계산
        if hasattr(self, '_prev_loss') and self._prev_loss is not None:
            plasticity = self.metrics_tracker.compute_plasticity(self._prev_loss, current_loss)
            quick_metrics['Plasticity'] = plasticity
        
        self._prev_loss = current_loss

        quick_metrics['DeltaFisher'] = 0.0
        quick_metrics['EMA_Drift'] = 0.0
        
        return quick_metrics

    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        if self.accelerator.is_main_process:
            save_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
            if is_best:
                save_dir = os.path.join(self.output_dir, "best-model")
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 모델 저장
            unwrapped_model = self.accelerator.unwrap_model(self.post_model)
            unwrapped_model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            
            # 학습 상태 저장
            state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "loss_history": self.loss_history[-1000:],
                "args": vars(self.args)
            }
            
            torch.save(state, os.path.join(save_dir, "training_state.pt"))
            print(f"Checkpoint saved: {save_dir}")

    def train(self):
        """전체 학습 프로세스"""
        print("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*50}")
            
            epoch_start_time = time.time()
            
            # 에포크 학습
            avg_epoch_loss = self.train_epoch(epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # 에포크 결과 출력
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Average Loss: {avg_epoch_loss:.4f}")
                print(f"  Time: {epoch_time:.2f} seconds")
                print(f"  Global Steps: {self.global_step}")
                
                # 최고 성능 업데이트
                is_best = avg_epoch_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_epoch_loss
                    print(f"New best loss: {self.best_loss:.4f}")
                
                # 체크포인트 저장
                if (epoch + 1) % getattr(self.args, 'save_interval', 1) == 0:
                    self.save_checkpoint(epoch + 1, is_best)
                
                # 에포크 요약 로깅
                if self.wandb_logger:
                    epoch_summary = {
                        "epoch_loss": avg_epoch_loss,
                        "epoch_time": epoch_time,
                        "best_loss": self.best_loss,
                        "is_best": is_best
                    }
                    
                    self.wandb_logger.log_epoch_summary(
                        epoch=epoch,
                        epoch_metrics=epoch_summary
                    )
        
        print("\nTraining completed successfully!")

    def run(self):
        """전체 학습 파이프라인 실행"""
        try:
            print("=" * 60)
            print("STARTING TRAINING PIPELINE")
            print("=" * 60)
            
            # 1. 모델과 토크나이저 로드
            self.load_model_and_tokenizer()            
            # 2. 데이터 준비
            self.prepare_data()            
            # 3. Loss 함수 설정
            self.setup_loss_function()            
            # 4. Optimizer와 Scheduler 설정            
            self.setup_optimizer_and_scheduler()            
            # 5. Accelerator로 준비
            self.prepare_training()            
            # 6. 학습 실행
            self.train()
            
            print("=" * 60)
            print("TRAINING PIPELINE COMPLETED")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        finally:
            # 리소스 정리
            if self.wandb_logger:
                self.wandb_logger.finish()
