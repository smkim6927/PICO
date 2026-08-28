"""
cp4llm/train/methods.py
=======================
PICO and every baseline, expressed only as hooks on BaseCPTTrainer.

Every method inherits the same loop, data pipeline, seeding, padding,
checkpointing, and logging from common.py. What appears below is exactly the
difference between methods and nothing else.

Method summary
--------------
  pico     PICO optimizer. GateU + SpecFlag + group pause + gated noise.
  adam     Adam, constant LR. The plain sequential baseline.
  sgd      plain SGD. Momentum is fixed at 0 and cannot be overridden.
  ewc      Adam plus Elastic Weight Consolidation penalty.
  replay   Adam plus a FIFO token replay buffer, mixed batches.
  rewarm   Adam plus a per-stage warmup and cosine schedule.
  mer      Adam plus replay plus MER-style Reptile interpolation.
  lora     LoRA adapters trained with Adam.

Deliberate changes from the original per-method scripts
-------------------------------------------------------

  1. Global TF32 enabling in the replay script removed. It changed matmul
     precision for that arm only.
  2. Gradient clipping removed from LoRA and MER. PICO uses none.
  3. Batch size, accumulation, epochs, LR, seed, and workers unified.
  4. Checkpoint directories keyed by stage id, not by source filename.
  5. LoRA uses Adam.
  6. MER now uses the shared TextDatasetwchunk pipeline instead of its own
     packing loop. The two produced different token streams.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Optional

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from optimizer.PICO import PICO
from train.common import BaseCPTTrainer, left_pad_sequence, left_pad_to_length


# =============================================================================
# PICO
# =============================================================================

class PICOTrainer(BaseCPTTrainer):
    method_name = "pico"

    def build_optimizer(self, model):
        a = self.args
        return PICO(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=a.weight_decay,
            beta_utility=a.beta_utility,
            sigma=a.sigma,
            spectral_update_freq=a.spectral_update_freq,
            power_iterations=a.power_iterations,
        )


# Plain optimizer baselines

class AdamTrainer(BaseCPTTrainer):
    method_name = "adam"

    def build_optimizer(self, model):
        return Adam(model.parameters(), lr=self.learning_rate,
                    betas=(0.9, 0.999), eps=1e-8)

class SGDTrainer(BaseCPTTrainer):
    """Paper H.2.1: plain SGD, no momentum, no weight decay, no Nesterov.
    momentum is fixed at 0.0 in code and has no CLI flag, so no script or
    override can reintroduce it."""

    method_name = "sgd"

    def build_optimizer(self, model):
        return SGD(model.parameters(), lr=self.learning_rate,
                   momentum=0.0, weight_decay=0.0,
                   nesterov=False)


# EWC 

class _EWCConstraint:
    def __init__(self, fisher_cpu: dict, params_cpu: dict, lambda_ewc: float):
        self.fisher = fisher_cpu
        self.params = params_cpu
        self.lambda_ewc = lambda_ewc

    def penalty(self, model) -> torch.Tensor:
        device = next(model.parameters()).device
        loss = torch.zeros((), device=device)
        for n, p in model.named_parameters():
            if n not in self.fisher or not p.requires_grad:
                continue
            f = self.fisher[n].to(device, non_blocking=True)
            theta_star = self.params[n].to(device, non_blocking=True)
            loss = loss + (f * (p - theta_star).pow(2)).sum()
        return 0.5 * self.lambda_ewc * loss


class EWCTrainer(BaseCPTTrainer):
    method_name = "ewc"

    def __init__(self, args):
        super().__init__(args)
        self.constraints: List[_EWCConstraint] = []

    def build_optimizer(self, model):
        return Adam(model.parameters(), lr=self.learning_rate,
                     betas=(0.9, 0.999), eps=1e-8)

    def extra_loss(self):
        if not self.constraints:
            return None
        total = torch.zeros((), device=self.device)
        for c in self.constraints:
            total = total + c.penalty(self.model)
        return total

    def _estimate_fisher(self, dataloader) -> _EWCConstraint:
        model = self.model
        model.eval()

        fisher_gpu = {n: torch.zeros_like(p.data)
                      for n, p in model.named_parameters() if p.requires_grad}
        snapshot = {n: p.detach().to("cpu", copy=True)
                    for n, p in model.named_parameters() if p.requires_grad}

        n_used = 0
        bar = tqdm(dataloader, desc="Fisher",
                   total=min(self.args.fisher_num_batches, len(dataloader)),
                   disable=not self.accelerator.is_main_process, leave=False)

        for idx, batch in enumerate(bar):
            if idx >= self.args.fisher_num_batches:
                break
            if (batch["labels"] != -100).sum().item() == 0:
                continue
            for p in model.parameters():
                p.grad = None
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            self.accelerator.backward(out.loss)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None and n in fisher_gpu:
                        fisher_gpu[n] += p.grad.detach() ** 2
            n_used += 1

        for p in model.parameters():
            p.grad = None

        fisher_cpu = {n: (f / max(n_used, 1)).detach().to("cpu", copy=True)
                      for n, f in fisher_gpu.items()}
        del fisher_gpu
        model.train()

        if self.accelerator.is_main_process:
            total = sum(v.sum().item() for v in fisher_cpu.values())
            print(f"[EWC] Fisher done. batches={n_used}, sum(F)={total:.3e}")

        return _EWCConstraint(fisher_cpu, snapshot, self.args.lambda_ewc)

    def on_stage_end(self, dataset, stage_id: str, stage_idx: int) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(f"[EWC] Estimating Fisher for '{stage_id}'")

        _, loader = self.prepare_data(dict(self.stages)[stage_id])
        loader = self.accelerator.prepare(loader)
        new_c = self._estimate_fisher(loader)

        if self.args.online_ewc and self.constraints:
            old = self.constraints[-1]
            merged = {}
            for n, f_new in new_c.fisher.items():
                f_old = old.fisher.get(n, torch.zeros_like(f_new))
                merged[n] = self.args.ewc_gamma * f_old + f_new
            self.constraints = [
                _EWCConstraint(merged, new_c.params, self.args.lambda_ewc)
            ]
        else:
            self.constraints.append(new_c)

        if self.accelerator.is_main_process:
            print(f"[EWC] stored constraints: {len(self.constraints)}")
        self.accelerator.wait_for_everyone()


# Replay
class _TokenReplayBuffer:
    """FIFO buffer. Sampling uses a dedicated generator seeded identically on
    every rank, so all ranks draw the same replay rows."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed + 202)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, samples: list) -> None:
        for s in samples:
            self.buffer.append({
                "input_ids": s["input_ids"].cpu(),
                "attention_mask": s["attention_mask"].cpu(),
            })

    def sample(self, n: int) -> list:
        n = min(n, len(self.buffer))
        if n == 0:
            return []
        return self._rng.sample(list(self.buffer), n)


class ReplayTrainer(BaseCPTTrainer):
    method_name = "replay"

    def __init__(self, args):
        super().__init__(args)
        self.buffer = _TokenReplayBuffer(args.replay_capacity, args.seed)

    def build_optimizer(self, model):
        return Adam(model.parameters(), lr=self.learning_rate,
                    betas=(0.9, 0.999), eps=1e-8)

    def pre_forward(self, batch: dict) -> dict:
        ratio = self.args.replay_ratio
        if len(self.buffer) == 0 or ratio <= 0.0:
            return batch

        bsz = batch["input_ids"].size(0)
        n_replay = int(bsz * ratio)
        n_replay = max(1, min(n_replay, bsz - 1)) if bsz >= 2 else 0
        if n_replay == 0:
            return batch

        samples = self.buffer.sample(n_replay)
        if not samples:
            return batch
        n_replay = len(samples)
        n_keep = bsz - n_replay
        pad_id = int(self.tokenizer.pad_token_id or 0)

        kept_ids = batch["input_ids"][:n_keep]
        kept_attn = batch["attention_mask"][:n_keep]
        kept_lab = batch["labels"][:n_keep]

        rep_ids = left_pad_sequence(
            [s["input_ids"] for s in samples], pad_id).to(self.device)
        rep_attn = left_pad_sequence(
            [s["attention_mask"] for s in samples], 0).to(self.device)

        target = max(kept_ids.size(1) if n_keep > 0 else 0, rep_ids.size(1))
        if n_keep > 0 and kept_ids.size(1) < target:
            kept_ids = left_pad_to_length(kept_ids, target, pad_id)
            kept_attn = left_pad_to_length(kept_attn, target, 0)
            kept_lab = left_pad_to_length(kept_lab, target, -100)
        if rep_ids.size(1) < target:
            rep_ids = left_pad_to_length(rep_ids, target, pad_id)
            rep_attn = left_pad_to_length(rep_attn, target, 0)

        rep_lab = rep_ids.clone()
        rep_lab[rep_attn == 0] = -100

        if n_keep > 0:
            return {
                "input_ids": torch.cat([kept_ids, rep_ids], dim=0),
                "attention_mask": torch.cat([kept_attn, rep_attn], dim=0),
                "labels": torch.cat([kept_lab, rep_lab], dim=0),
            }
        return {"input_ids": rep_ids, "attention_mask": rep_attn,
                "labels": rep_lab}

    def on_stage_end(self, dataset, stage_id: str, stage_idx: int) -> None:
        n_total = len(dataset)
        n_sample = min(self.args.replay_sample_per_stage, n_total)
        rng = random.Random(self.seed * 1_000_003 + stage_idx)
        indices = rng.sample(range(n_total), n_sample)
        self.buffer.add([
            {"input_ids": dataset[i]["input_ids"],
             "attention_mask": dataset[i]["attention_mask"]}
            for i in indices
        ])
        if self.accelerator.is_main_process:
            print(f"[Replay] buffer {len(self.buffer)}/{self.args.replay_capacity}")
        self.accelerator.wait_for_everyone()


# Re-warm
class RewarmTrainer(BaseCPTTrainer):
    method_name = "rewarm"

    def __init__(self, args):
        super().__init__(args)
        self.scheduler = None

    def build_optimizer(self, model):
        return Adam(model.parameters(), lr=self.learning_rate,
                    betas=(0.9, 0.999), eps=1e-8)

    def on_stage_start(self, stage_id: str, stage_idx: int, updates: int) -> None:
        total = max(1, int(updates))
        warmup = max(1, int(round(total * self.args.warmup_ratio)))
        floor = float(self.args.min_lr_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cos

        # LambdaLR multiplies against initial_lr, so rebuilding per stage
        # re-warms exactly to peak instead of inheriting the decayed value.
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        if self.accelerator.is_main_process:
            print(f"[rewarm] stage schedule: {total} updates, warmup {warmup}, "
                  f"cosine floor {floor:.0%}")

    def on_optimizer_update(self):
        if self.scheduler is not None:
            self.scheduler.step()


# MER-style Reptile with replay
class MERTrainer(ReplayTrainer):
    method_name = "mer"

    def __init__(self, args):
        super().__init__(args)
        self.anchor: Optional[List[torch.Tensor]] = None
        self.updates_since_anchor = 0
        self.interp_count = 0

    def post_prepare(self) -> None:
        self._reset_anchor()

    @torch.no_grad()
    def _reset_anchor(self) -> None:
        self.anchor = [p.detach().to("cpu", torch.float32, copy=True)
                       for p in self.model.parameters()]
        self.updates_since_anchor = 0

    def on_stage_start(self, stage_id: str, stage_idx: int, updates: int) -> None:
        if self.args.mer_reset_anchor_per_stage:
            self._reset_anchor()

    @torch.no_grad()
    def on_optimizer_update(self):
        self.updates_since_anchor += 1
        if self.updates_since_anchor < self.args.mer_k:
            return

        drift_sq = 0.0
        eps = self.args.mer_eps
        for p, a in zip(self.model.parameters(), self.anchor):
            cur = p.detach().to("cpu", torch.float32)
            delta = cur - a
            drift_sq += float((delta * delta).sum())
            p.data.copy_((a + eps * delta).to(p.dtype).to(p.device))

        self._reset_anchor()
        self.interp_count += 1
        if self.accelerator.is_main_process:
            print(f"[mer] interpolation {self.interp_count}, "
                  f"window drift L2 = {math.sqrt(drift_sq):.4e}")


# LoRA
class LoRATrainer(BaseCPTTrainer):
    method_name = "lora"

    def wrap_model(self, model):
        from peft import LoraConfig, TaskType, get_peft_model

        targets = self.args.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        # alpha is fixed at 16 in code per paper H.2.7 and has no CLI flag.
        config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        model = get_peft_model(model, config, autocast_adapter_dtype=False)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
        return model

    def build_optimizer(self, model):
        # Paper H.2.7: "The optimizer is Adam ... applied only to LoRA
        # parameters", with no decoupled weight decay.
        params = [p for p in model.parameters() if p.requires_grad]
        return Adam(params, lr=self.learning_rate, betas=(0.9, 0.999),
                    eps=1e-8)


# =============================================================================
# Sophia (Liu et al., 2024) — paper H.2.2
# =============================================================================

class SophiaTrainer(BaseCPTTrainer):
    """Paper H.2.2: eta 2e-5, beta1 0.965, beta2 0.99, rho 0.05, Hessian
    update interval k = 10, Gauss-Newton-Bartlett estimator.

    The optimizer is the official SophiaG implementation shipped verbatim in
    cp4llm/optimizer/sophia.py. update_hessian() consumes the gradients of the
    GNB loss as h <- beta2 h + (1 - beta2) g^2; the bs factor enters inside
    step() through the official ratio exp_avg.abs() / (rho * bs * hessian).
    step() is called by accelerate without kwargs, so bs stays at the official
    default of 5120."""

    method_name = "sophia"

    def __init__(self, args):
        super().__init__(args)
        self._cached_batch: Optional[dict] = None
        self._updates = 0

    def build_optimizer(self, model):
        from optimizer.sophia import SophiaG
        return SophiaG(
            model.parameters(),
            lr=self.learning_rate,
            betas=(self.args.sophia_beta1, self.args.sophia_beta2),
            rho=self.args.sophia_rho,
            weight_decay=0.0,
        )

    def pre_forward(self, batch: dict) -> dict:
        self._cached_batch = batch
        return batch

    def on_optimizer_update(self):
        self._updates += 1
        if self._updates % self.args.sophia_k != 0 or self._cached_batch is None:
            return

        batch = self._cached_batch
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"])
        logits = outputs.logits[:, :-1, :]
        mask = (batch["labels"][:, 1:] != -100)
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=-1)
            sampled = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).view(mask.shape)
        gnb_labels = sampled.masked_fill(~mask, -100)
        gnb_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            gnb_labels.reshape(-1),
            ignore_index=-100,
        )
        self.accelerator.backward(gnb_loss)
        opt = self.optimizer
        target = opt.optimizer if hasattr(opt, "optimizer") else opt
        target.update_hessian()
        self.optimizer.zero_grad(set_to_none=True)


# =============================================================================
# Registry
# =============================================================================

METHODS: Dict[str, type] = {
    "pico": PICOTrainer,
    "adam": AdamTrainer,
    "sgd": SGDTrainer,
    "ewc": EWCTrainer,
    "replay": ReplayTrainer,
    "rewarm": RewarmTrainer,
    "sophia": SophiaTrainer,
    "mer": MERTrainer,
    "lora": LoRATrainer,
}
