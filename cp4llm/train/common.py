"""
cp4llm/train/common.py
======================
Shared continual pre-training core for PICO and every baseline.

Design rule: the training loop, data pipeline, seeding, padding, checkpointing,
and logging live here and are IDENTICAL for all methods. A method may only
differ through the hooks declared on BaseCPTTrainer. Anything a method changes
outside those hooks is a confound, not a method.

Unified conditions (taken from the PICO run):
    model                 meta-llama/Llama-3.2-1B-Instruct
    dtype                 bfloat16, accelerate mixed_precision="bf16"
    batch_size            4 per device (paper H.2.8)
    grad_accum            8 (effective 32 per device, paper H.2.8)
    num_epochs            5 per stage
    learning_rate         2e-5, constant
    max_length            256
    chunk_size            64
    padding               left
    gradient clipping     none
    LR schedule / warmup  none
    optimizer state       preserved across stages
    gradient checkpoint   enabled, use_cache=False
    seed                  777, deterministic algorithms on
    parallelism           DDP only, FSDP explicitly rejected

FSDP is rejected on purpose. FSDP flattens parameters, so the `p.dim() == 2`
guard inside PICO's spectral monitor silently matches nothing and SpecFlag
never fires. A run that quietly disables half the method is worse than a run
that refuses to start.
"""

from __future__ import annotations

import gc
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.dataset_loader import TextDatasetwchunk


# =============================================================================
# Unified training conditions
# =============================================================================

TRAIN_DEFAULTS: Dict[str, object] = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "max_length": 256,
    "chunk_size": 64,
    "num_workers": 8,
    "seed": 777,
}

HF_DATASET_REPO = "aigogongburani/PICO_Cross-Lingual_CPT_Corpora"

# Stage order follows the paper: KO-Medical, EN-Medical, KO-Legal, EN-Legal.
# `hf_path` is the path inside the dataset repo. KO-Medical has none: per the
# paper's data appendix it combines material derived from two AI Hub datasets,
# 71487 (Medical and Legal Professional Book Corpus, v1.2) and 110
# (Professional Domain Corpus, v1.1). AI Hub terms permit training use but not
# redistribution, so neither source is mirrored.
STAGE_SPEC: List[dict] = [
    {"id": "kor_medical", "hf_path": None,
     "local_env": "KOR_MEDICAL_PATH", "required": False},
    {"id": "eng_medical", "hf_path": "data/guidline_medical.txt",
     "local_env": "ENG_MEDICAL_PATH", "required": True},
    {"id": "kor_legal", "hf_path": "data/new-legal-kor-dataset.txt",
     "local_env": "KOR_LEGAL_PATH", "required": True},
    {"id": "eng_legal", "hf_path": "data/eng-new-legal-dataset.txt",
     "local_env": "ENG_LEGAL_PATH", "required": True},
]

KOR_MEDICAL_NOTE = (
    "        KO-Medical combines material derived from two AI Hub datasets:\n"
    "        71487 (Medical and Legal Professional Book Corpus, v1.2) and\n"
    "        110 (Professional Domain Corpus, v1.1). AI Hub terms permit\n"
    "        training use but prohibit redistribution, so it is not mirrored\n"
    "        on the Hub. Obtain the sources yourself and set KOR_MEDICAL_PATH."
)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    accelerate_set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Left padding
# =============================================================================

def left_pad_sequence(sequences, padding_value: int = 0) -> torch.Tensor:
    max_len = max(s.size(0) for s in sequences)
    padded = []
    for s in sequences:
        pad_size = max_len - s.size(0)
        if pad_size > 0:
            s = torch.nn.functional.pad(s, (pad_size, 0), value=padding_value)
        padded.append(s)
    return torch.stack(padded, dim=0)


def left_pad_to_length(t: torch.Tensor, target_len: int, value) -> torch.Tensor:
    cur_len = t.size(1)
    if cur_len >= target_len:
        return t
    return torch.nn.functional.pad(t, (target_len - cur_len, 0), value=value)


def assert_left_padded(input_ids, attention_mask) -> bool:
    bsz, _ = attention_mask.shape
    for i in range(bsz):
        zeros = (attention_mask[i] == 0).nonzero(as_tuple=True)[0]
        ones = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(zeros) == 0 or len(ones) == 0:
            continue
        if zeros.max() >= ones.min():
            return False
    return True


# =============================================================================
# Corpus resolution
# =============================================================================

def resolve_stage_files(
    repo_id: str = HF_DATASET_REPO,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """Resolve every stage to a local path. Local override beats the Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. pip install huggingface_hub"
        ) from exc

    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    resolved: List[Tuple[str, str]] = []

    for spec in STAGE_SPEC:
        sid = spec["id"]
        override = os.environ.get(spec["local_env"]) if spec["local_env"] else None

        if override:
            if not os.path.isfile(override):
                raise FileNotFoundError(
                    f"[{sid}] {spec['local_env']} set but missing: {override}"
                )
            resolved.append((sid, override))
            if verbose:
                print(f"[data] {sid:<12} local  {override}")
            continue

        if spec["hf_path"] is None:
            msg = (f"[data] {sid:<12} SKIPPED. Not hosted. "
                   f"Set {spec['local_env']} to a local copy.")
            if spec["required"]:
                raise FileNotFoundError(msg)
            if verbose:
                print(msg)
                if sid == "kor_medical":
                    print(KOR_MEDICAL_NOTE)
            continue

        path = hf_hub_download(
            repo_id=repo_id, filename=spec["hf_path"], repo_type="dataset",
            revision=revision, token=token, cache_dir=cache_dir,
        )
        resolved.append((sid, path))
        if verbose:
            mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[data] {sid:<12} hub    {spec['hf_path']}  ({mb:.0f} MB)")

    if not resolved:
        raise RuntimeError("No corpora resolved. Nothing to train on.")
    return resolved


def filter_stages(
    stages: List[Tuple[str, str]], only: Optional[List[str]]
) -> List[Tuple[str, str]]:
    if not only:
        return stages
    keep = [s for s in stages if s[0] in only]
    missing = [s for s in only if s not in {x[0] for x in stages}]
    if missing:
        raise ValueError(f"requested stages not resolvable: {missing}")
    order = {spec["id"]: i for i, spec in enumerate(STAGE_SPEC)}
    return sorted(keep, key=lambda s: order[s[0]])


# =============================================================================
# Base trainer
# =============================================================================

class BaseCPTTrainer:
    """Continual pre-training loop shared by every method.

    Subclasses override only these hooks:
        method_name         str, used for run naming and logging
        wrap_model(model)   return a possibly wrapped model, before prepare
        build_optimizer(model)  return the raw optimizer, before prepare
        post_prepare()      called once after accelerator.prepare
        on_stage_start(stage_id, stage_idx, updates_this_stage)
        pre_forward(batch)  return a possibly modified batch
        extra_loss()        return a tensor added to the task loss, or None
        on_optimizer_update()  called on true optimizer-update boundaries
        on_stage_end(dataset, stage_id, stage_idx)
        save_stage(stage_id, epoch)  override only if the artifact differs
    """

    method_name = "base"

    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.output_dir = args.output_dir
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.max_length = args.max_length
        self.chunk_size = args.chunk_size
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.debug = args.debug

        set_seed(self.seed, deterministic=True)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
        )
        self.device = self.accelerator.device

        if self.accelerator.distributed_type == DistributedType.FSDP:
            raise RuntimeError(
                "FSDP is not supported. FSDP flattens parameters, so PICO's "
                "p.dim() == 2 spectral guard matches nothing and SpecFlag is "
                "silently disabled. Use DDP, or --device_map balanced for "
                "models that do not fit on one GPU."
            )

        self.device_map = getattr(args, "device_map", None)
        if self.device_map and self.accelerator.num_processes != 1:
            raise RuntimeError(
                "--device_map requires a single process. Launch with plain "
                "python (not accelerate launch with multiple processes). "
                "Layers are spread across the visible GPUs by the map itself, "
                "and parameters keep their 2D shape so SpecFlag stays active."
            )

        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.global_update_step = 0
        self.current_stage_idx = 0
        self._collate_debug_done = False

        # Corpora
        if getattr(args, "stage_paths", None):
            self.stages = args.stage_paths
        else:
            with self.accelerator.main_process_first():
                self.stages = resolve_stage_files(
                    repo_id=args.hf_repo_id,
                    revision=args.hf_revision,
                    cache_dir=args.hf_cache_dir,
                    verbose=self.accelerator.is_main_process,
                )
        self.stages = filter_stages(self.stages, getattr(args, "only_stages", None))

        os.makedirs(self.output_dir, exist_ok=True)
        if self.accelerator.is_main_process:
            self._print_header()

    # ── hooks (defaults are no-ops) ──────────────────────────────────────
    def wrap_model(self, model):
        return model

    def build_optimizer(self, model):
        raise NotImplementedError

    def post_prepare(self) -> None:
        pass

    def on_stage_start(self, stage_id: str, stage_idx: int, updates: int) -> None:
        pass

    def pre_forward(self, batch: dict) -> dict:
        return batch

    def extra_loss(self):
        return None

    def on_optimizer_update(self) -> None:
        pass

    def on_stage_end(self, dataset, stage_id: str, stage_idx: int) -> None:
        pass

    # ── reporting ────────────────────────────────────────────────────────
    def _print_header(self) -> None:
        stream = " -> ".join(sid for sid, _ in self.stages)
        eff = self.batch_size * self.gradient_accumulation_steps
        print(f"\n{'=' * 72}")
        print(f"[INFO] Continual Pre-Training  |  method = {self.method_name}")
        print(f"{'=' * 72}")
        print(f"Model                    : {self.model_name}")
        print(f"# processes              : {self.accelerator.num_processes}")
        print(f"Distributed type         : {self.accelerator.distributed_type}")
        if self.device_map:
            print(f"Device map               : {self.device_map} (single process, model parallel)")
        print(f"# stages                 : {len(self.stages)}")
        print(f"Stream                   : {stream}")
        print(f"Batch size (per device)  : {self.batch_size}")
        print(f"Grad accumulation        : {self.gradient_accumulation_steps}")
        print(f"Effective batch size     : {eff}")
        print(f"Epochs per stage         : {self.num_epochs}")
        print(f"Learning rate            : {self.learning_rate}")
        print(f"max_length / chunk_size  : {self.max_length} / {self.chunk_size}")
        print(f"Seed                     : {self.seed}")
        print("padding_side             : left")
        print("Gradient clipping        : None")
        print("LR schedule / warmup     : None (unless the method defines it)")
        print("Optimizer state          : preserved across stages")
        if len(self.stages) < len(STAGE_SPEC):
            print()
            print("WARNING: reduced stream. This covers only part of the")
            print("         four-stage protocol described in the paper.")
        print(f"{'=' * 72}\n")

    # ── model ────────────────────────────────────────────────────────────
    def load_model_and_tokenizer(self) -> None:
        if self.accelerator.is_main_process:
            print(f"[INFO] Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        load_kwargs = dict(torch_dtype=torch.bfloat16, use_cache=False,
                           trust_remote_code=True)
        if self.device_map:
            load_kwargs["device_map"] = self.device_map
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = self.wrap_model(model)
        model.train()
        self.model = model

        raw_optimizer = self.build_optimizer(self.model)
        if self.device_map:
            # The model is already placed across GPUs by the device map.
            # Wrapping it in DDP would try to move it and break the layout,
            # so only the optimizer goes through prepare.
            self.optimizer = self.accelerator.prepare(raw_optimizer)
        else:
            self.model, self.optimizer = self.accelerator.prepare(
                self.model, raw_optimizer
            )
        self.post_prepare()

        if self.debug and self.accelerator.is_main_process:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)
            n2d = sum(1 for p in self.model.parameters() if p.dim() == 2)
            print(f"[DEBUG] pad_token_id       : {self.tokenizer.pad_token_id}")
            print(f"[DEBUG] Trainable params   : {trainable:,} / {total:,}")
            print(f"[DEBUG] 2D params visible  : {n2d}")
            if n2d == 0:
                print("[WARN ] No 2D parameters visible. A spectral monitor "
                      "would never fire under this setup.")

    # ── data ─────────────────────────────────────────────────────────────
    def _make_collate_fn(self):
        self._collate_debug_done = False

        def _collate_fn(batch):
            pad_id = int(self.tokenizer.pad_token_id or 0)
            input_ids = left_pad_sequence(
                [b["input_ids"] for b in batch], padding_value=pad_id)
            attn = left_pad_sequence(
                [b["attention_mask"] for b in batch], padding_value=0)
            labels = input_ids.clone()
            labels[attn == 0] = -100

            if (self.debug and not self._collate_debug_done
                    and self.accelerator.is_main_process):
                self._collate_debug_done = True
                mismatch = ((input_ids == pad_id) ^ (attn == 0)).sum().item()
                print(f"[debug][collate] pad_id={pad_id}, bsz={len(batch)}, "
                      f"seq_len={input_ids.size(1)}, "
                      f"valid_labels={(labels != -100).sum().item()}, "
                      f"pad_mask_mismatch={mismatch}, "
                      f"left_padded={assert_left_padded(input_ids, attn)}")

            return {"input_ids": input_ids, "attention_mask": attn,
                    "labels": labels}
        return _collate_fn

    def prepare_data(self, txt_file: str):
        dataset = TextDatasetwchunk(
            txt_file=txt_file, tokenizer=self.tokenizer,
            max_length=self.max_length, chunk_size=self.chunk_size,
        )
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=self._make_collate_fn(), num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
            generator=g, persistent_workers=False, drop_last=False,
        )
        return dataset, loader

    # ── checkpointing ────────────────────────────────────────────────────
    def save_stage(self, stage_id: str, epoch: int) -> str:
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(self.output_dir, f"{stage_id}_epoch_{epoch}")
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = self.accelerator.get_state_dict(self.model)
        unwrapped.save_pretrained(
            save_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(save_path)
            gen_cfg = getattr(unwrapped, "generation_config", None)
            if gen_cfg is not None:
                gen_cfg.save_pretrained(save_path)
            cfg_path = os.path.join(save_path, "config.json")
            if not os.path.exists(cfg_path):
                unwrapped.config.to_json_file(cfg_path)
            print(f"[save] {save_path}")
        self.accelerator.wait_for_everyone()
        return save_path

    # ── training ─────────────────────────────────────────────────────────
    def train_one_stage(self, dataloader, stage_id: str) -> None:
        dataloader = self.accelerator.prepare(dataloader)

        updates_per_epoch = math.ceil(
            len(dataloader) / self.gradient_accumulation_steps)
        self.on_stage_start(
            stage_id, self.current_stage_idx,
            updates_per_epoch * self.num_epochs,
        )

        for epoch in range(self.num_epochs):
            set_seed(self.seed + epoch, deterministic=True)
            self.model.train()
            loss_sum, n_step = 0.0, 0

            bar = tqdm(
                dataloader,
                desc=f"{stage_id} | Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process, leave=False,
            )

            for step, batch in enumerate(bar):
                batch = self.pre_forward(batch)
                valid_cnt = (batch["labels"] != -100).sum().item()

                if self.debug and step == 0 and self.accelerator.is_main_process:
                    print(f"[debug][train] {stage_id} | "
                          f"bsz={batch['input_ids'].size(0)}, "
                          f"seq_len={batch['input_ids'].size(1)}, "
                          f"valid_labels={valid_cnt}, left_padded="
                          f"{assert_left_padded(batch['input_ids'], batch['attention_mask'])}")

                if valid_cnt == 0:
                    continue

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    task_loss = outputs.loss
                    penalty = self.extra_loss()
                    loss = task_loss if penalty is None else task_loss + penalty
                    self.accelerator.backward(loss)
                    # No gradient clipping. Unified across all methods.
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    self.on_optimizer_update()

                loss_val = float(loss.detach().item())
                self.global_update_step += 1
                loss_sum += loss_val
                n_step += 1

                bar.set_postfix({
                    "Loss": f"{loss_val:.4f}",
                    "Step": f"{step + 1}/{len(dataloader)}",
                    "gStep": self.global_update_step,
                })

            avg_loss = loss_sum / max(n_step, 1)
            if self.accelerator.is_main_process:
                print(f"\n[{stage_id}] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.save_stage(stage_id, epoch + 1)

    def train(self) -> None:
        for stage_idx, (stage_id, path) in enumerate(self.stages):
            self.current_stage_idx = stage_idx
            if self.accelerator.is_main_process:
                print(f"\n{'=' * 72}")
                print(f"[Stage {stage_idx + 1}/{len(self.stages)}] {stage_id}")
                print(f"  file          : {path}")
                print(f"  updates so far: {self.global_update_step}")
                print(f"{'=' * 72}")

            dataset, dataloader = self.prepare_data(path)
            self.train_one_stage(dataloader, stage_id)
            self.on_stage_end(dataset, stage_id, stage_idx)

            del dataset, dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
