import os
import gc
import random
import torch
import numpy as np
import wandb
from collections import deque
from tqdm import tqdm
from functools import partial

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.data import DataLoader, Dataset

from utils.dataset_loader import TextDatasetwchunk

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "777"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─────────────────────────────────────────────────────────────────────────────
# LEFT-pad utilities
#   F.pad(x, (pad_size, 0))  →  왼쪽에만 padding (= left-pad)
#   tokenizer.padding_side="left" 와 정확히 일치
# ─────────────────────────────────────────────────────────────────────────────
def left_pad_sequence(sequences, padding_value=0):
    """1-D tensor 리스트를 left-padding 으로 stack → (B, max_len)."""
    max_len = max(s.size(0) for s in sequences)
    padded = []
    for s in sequences:
        pad_size = max_len - s.size(0)
        if pad_size > 0:
            # (pad_size, 0): 마지막 차원 왼쪽에만 pad_size 만큼 추가
            s = torch.nn.functional.pad(s, (pad_size, 0), value=padding_value)
        padded.append(s)
    return torch.stack(padded, dim=0)


def left_pad_to_length(t: torch.Tensor, target_len: int, value):
    """2-D (B, L) 텐서를 target_len 까지 LEFT-padding."""
    cur_len = t.size(1)
    if cur_len >= target_len:
        return t
    pad_size = target_len - cur_len
    # 2-D 마지막 차원 기준 (left=pad_size, right=0)
    return torch.nn.functional.pad(t, (pad_size, 0), value=value)


def assert_left_padded(input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_id: int) -> bool:
    """
    각 row에서 (mask==0) 영역이 시퀀스 앞쪽에만 모여있는지 검증.
    left-pad 라면: 각 row는 [0,0,...,0,1,1,...,1] 모양이어야 함.
    """
    # mask == 0 인 위치들의 인덱스가 연속적으로 0..k-1 인지 확인
    mask = attention_mask
    bsz, L = mask.shape
    for i in range(bsz):
        zeros = (mask[i] == 0).nonzero(as_tuple=True)[0]
        ones  = (mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(zeros) == 0:
            continue
        if len(ones) == 0:
            continue
        # 모든 0이 모든 1보다 앞에 와야 함
        if zeros.max() >= ones.min():
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
class TokenReplayBuffer:
    """
    Stage 완료 후 샘플들을 저장하는 버퍼. (FIFO via deque(maxlen=...))
    각 item: {"input_ids": Tensor, "attention_mask": Tensor}
    """
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, samples: list):
        for s in samples:
            self.buffer.append({
                "input_ids": s["input_ids"].cpu(),
                "attention_mask": s["attention_mask"].cpu(),
            })

    def sample(self, n: int) -> list:
        n = min(n, len(self.buffer))
        if n == 0:
            return []
        return random.sample(list(self.buffer), n)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Mixed Batch (replay 비율만큼 current를 *대체* → 총 batch size 유지, ALL LEFT-PAD)
# ─────────────────────────────────────────────────────────────────────────────
def mix_batch_with_replay(
    current_batch: dict,
    replay_buffer: TokenReplayBuffer,
    replay_ratio: float,
    pad_id: int,
    device,
) -> dict:
    """
    current_batch 의 일부를 replay 샘플로 교체.
    - n_replay = int(current_bsz * replay_ratio)
    - n_keep   = current_bsz - n_replay
    → 최종 mixed batch 크기 == current_bsz (= self.batch_size 유지).
    All padding is LEFT-padding (tokenizer.padding_side="left" 와 일치).
    """
    if len(replay_buffer) == 0 or replay_ratio <= 0.0:
        return current_batch

    current_bsz = current_batch["input_ids"].size(0)
    n_replay = int(current_bsz * replay_ratio)
    n_replay = max(1, min(n_replay, current_bsz - 1)) if current_bsz >= 2 else 0
    if n_replay == 0:
        return current_batch
    n_keep = current_bsz - n_replay

    replay_samples = replay_buffer.sample(n_replay)
    if not replay_samples:
        return current_batch
    n_replay = len(replay_samples)
    n_keep = current_bsz - n_replay

    kept_input_ids = current_batch["input_ids"][:n_keep]
    kept_attn_mask = current_batch["attention_mask"][:n_keep]
    kept_labels    = current_batch["labels"][:n_keep]

    # ── replay 샘플 left-pad stack ────────────────────────────────────────
    replay_input_ids = left_pad_sequence(
        [s["input_ids"] for s in replay_samples], padding_value=pad_id
    ).to(device)
    replay_attn_mask = left_pad_sequence(
        [s["attention_mask"] for s in replay_samples], padding_value=0
    ).to(device)

    # ── seq_len 정렬: 짧은 쪽을 LEFT-pad로 늘림 ─────────────────────────
    cur_len = kept_input_ids.size(1) if n_keep > 0 else 0
    rep_len = replay_input_ids.size(1)
    target_len = max(cur_len, rep_len)

    if n_keep > 0 and cur_len < target_len:
        kept_input_ids = left_pad_to_length(kept_input_ids, target_len, pad_id)
        kept_attn_mask = left_pad_to_length(kept_attn_mask, target_len, 0)
        kept_labels    = left_pad_to_length(kept_labels, target_len, -100)
    if rep_len < target_len:
        replay_input_ids = left_pad_to_length(replay_input_ids, target_len, pad_id)
        replay_attn_mask = left_pad_to_length(replay_attn_mask, target_len, 0)

    replay_labels = replay_input_ids.clone()
    replay_labels[replay_attn_mask == 0] = -100

    if n_keep > 0:
        mixed_input_ids = torch.cat([kept_input_ids, replay_input_ids], dim=0)
        mixed_attn_mask = torch.cat([kept_attn_mask, replay_attn_mask], dim=0)
        mixed_labels    = torch.cat([kept_labels,    replay_labels],    dim=0)
    else:
        mixed_input_ids = replay_input_ids
        mixed_attn_mask = replay_attn_mask
        mixed_labels    = replay_labels

    return {
        "input_ids": mixed_input_ids,
        "attention_mask": mixed_attn_mask,
        "labels": mixed_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        model_name="/path/to/model",
        dataset_list=None,
        output_dir="/home/jovyan/sumin_data/saved_model/Adam_replay/",
        batch_size=4,
        seed=777,
        num_epochs=1,
        learning_rate=2e-5,
        max_length=256,
        chunk_size=64,
        gradient_accumulation_steps=16,
        num_workers=4,
        use_replay=True,
        replay_capacity=5000,
        replay_ratio=0.5,
        replay_sample_per_stage=2000,
        debug=True,
        use_wandb=True,
    ):
        if dataset_list is None:
            raise ValueError("dataset_list must be provided.")

        set_seed(seed)
        self.model_name = model_name
        self.tokenizer_name = model_name
        self.dataset_list = dataset_list
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers

        self.use_replay = use_replay
        self.replay_capacity = replay_capacity
        self.replay_ratio = replay_ratio
        self.replay_sample_per_stage = replay_sample_per_stage

        self.debug = debug
        self.use_wandb = use_wandb

        self.replay_buffer = TokenReplayBuffer(capacity=self.replay_capacity)

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy=auto_wrap_policy,
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            use_orig_params=True,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
            fsdp_plugin=fsdp_plugin,
        )
        self.device = self.accelerator.device

        if self.debug and self.accelerator.is_main_process:
            print(f"\n{'=' * 60}")
            print("[INFO] Adam + Replay Trainer Configuration (LEFT-PAD)")
            print(f"{'=' * 60}")
            print(f"Processes        : {self.accelerator.num_processes}")
            print(f"Device           : {self.device}")
            print(f"use_replay       : {self.use_replay}")
            print(f"replay_capacity  : {self.replay_capacity}")
            print(f"replay_ratio     : {self.replay_ratio}")
            print(f"replay_per_stage : {self.replay_sample_per_stage}")
            print(f"padding_side     : left (collate + replay 모두)")
            print(f"{'=' * 60}\n")

        self.wandb_initialized = False
        self._collate_debug_done = False
        self.tokenizer = None
        self.model = None
        self.optimizer = None  # ← prepare 1회 후 보관

        os.makedirs(self.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    def load_model_and_tokenizer(self):
        if self.accelerator.is_main_process:
            print(f"[INFO] Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.train()

        # ── FSDP prepare는 모델/옵티마이저 모두 단 1회 ────────────────────
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        if self.debug and self.accelerator.is_main_process:
            print(f"[DEBUG] Pad token ID : {self.tokenizer.pad_token_id}")
            print(f"[DEBUG] padding_side : {self.tokenizer.padding_side}")
            print(f"[DEBUG] FSDP prepared once for model + optimizer")

    # ─────────────────────────────────────────────────────────────────────
    def _make_collate_fn(self):
        self._collate_debug_done = False

        def _collate_fn(batch):
            pad_id = int(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            input_ids_list = [item["input_ids"] for item in batch]
            attention_mask_list = [item["attention_mask"] for item in batch]

            # ── LEFT-padding (tokenizer.padding_side="left" 와 일치) ──
            input_ids_padded = left_pad_sequence(
                input_ids_list, padding_value=pad_id
            )
            attention_mask_padded = left_pad_sequence(
                attention_mask_list, padding_value=0
            )
            labels_padded = input_ids_padded.clone()
            labels_padded[attention_mask_padded == 0] = -100

            if (
                self.debug
                and not self._collate_debug_done
                and self.accelerator.is_main_process
            ):
                self._collate_debug_done = True
                valid = (labels_padded != -100).sum().item()
                mismatch = (
                    (input_ids_padded == pad_id) ^ (attention_mask_padded == 0)
                ).sum().item()
                is_left = assert_left_padded(
                    input_ids_padded, attention_mask_padded, pad_id
                )
                print(
                    f"[debug][collate] pad_id={pad_id}, bsz={len(batch)}, "
                    f"seq_len={input_ids_padded.size(1)}, "
                    f"valid_labels={valid}, pad_mask_mismatch={mismatch}, "
                    f"left_padded={is_left}"
                )

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "labels": labels_padded,
            }

        return _collate_fn

    def prepare_data(self, txt_file: str):
        """(dataset, dataloader) 튜플 → buffer 업데이트 시 dataset 재사용."""
        dataset = TextDatasetwchunk(
            txt_file=txt_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chunk_size=self.chunk_size,
        )
        g = torch.Generator()
        g.manual_seed(self.seed)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._make_collate_fn(),
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=False,
            drop_last=False,
        )
        return dataset, loader

    # ─────────────────────────────────────────────────────────────────────
    def _update_replay_buffer(self, dataset, stage_idx: int, dataset_name: str):
        """
        stage 완료 후, 학습에 사용한 dataset을 재활용.
        모든 rank가 동일 인덱스를 뽑도록 stage별 결정적 RNG 사용 → buffer 일치.
        """
        if not self.use_replay:
            return

        if self.accelerator.is_main_process:
            print(
                f"[Replay] Collecting {self.replay_sample_per_stage} samples "
                f"from {dataset_name} into replay buffer..."
            )

        n_total = len(dataset)
        n_sample = min(self.replay_sample_per_stage, n_total)

        # 결정적 RNG (전역 random과 분리, 모든 rank에서 동일 시퀀스 보장)
        rng = random.Random(self.seed * 1_000_003 + stage_idx)
        indices = rng.sample(range(n_total), n_sample)

        samples = []
        for idx in indices:
            item = dataset[idx]
            samples.append({
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            })

        self.replay_buffer.add(samples)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print(
                f"[Replay] Buffer size: {len(self.replay_buffer)} / {self.replay_capacity} "
                f"(deterministic across ranks)"
            )

    # ─────────────────────────────────────────────────────────────────────
    def _init_wandb(self, dataset_name: str):
        if self.wandb_initialized:
            return

        if not self.use_wandb:
            wandb.init(mode="disabled")
            self.wandb_initialized = True
            return

        if self.accelerator.is_main_process:
            name_map = {
                "new-medical-kor-dataset": "kor-medical",
                "guidline_medical": "eng-medical",
                "new-legal-kor-dataset": "kor-legal",
                "eng-new-legal-dataset": "eng-legal",
            }
            wname = next(
                (v for k, v in name_map.items() if k in dataset_name),
                dataset_name,
            )
            run_label = "Adam+Replay" if self.use_replay else "Adam"
            wandb.init(
                project="pact-cpt",
                config={
                    "optimizer": run_label,
                    "learning_rate": self.learning_rate,
                    "model": self.model_name,
                    "use_replay": self.use_replay,
                    "replay_capacity": self.replay_capacity,
                    "replay_ratio": self.replay_ratio,
                    "replay_sample_per_stage": self.replay_sample_per_stage,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "max_length": self.max_length,
                    "seed": self.seed,
                    "padding_side": "left",
                },
                name=f"{run_label}_ep{self.num_epochs}_{wname}_{self.seed}",
                group="baseline",
            )
        else:
            wandb.init(mode="disabled")

        self.wandb_initialized = True

    # ─────────────────────────────────────────────────────────────────────
    def _save_model(self, dataset_name: str, epoch: int):
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(
            self.output_dir, f"{dataset_name}_epoch_{epoch}"
        )
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(
            save_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(save_path)
            if (
                hasattr(unwrapped, "generation_config")
                and unwrapped.generation_config is not None
            ):
                unwrapped.generation_config.save_pretrained(save_path)
            config_path = os.path.join(save_path, "config.json")
            if not os.path.exists(config_path):
                unwrapped.config.to_json_file(config_path)
            print(f"✅ Model saved: {save_path}")

        self.accelerator.wait_for_everyone()

    # ─────────────────────────────────────────────────────────────────────
    def train_on_dataset(self, dataloader: DataLoader, dataset_name: str):
        self._init_wandb(dataset_name)

        if self.accelerator.is_main_process:
            mode = "Adam+Replay" if self.use_replay else "Adam"
            print(f"[INFO] [{mode}] Training on: {dataset_name}")

        # model/optimizer는 이미 prepare 완료. dataloader만 prepare
        dataloader = self.accelerator.prepare(dataloader)

        pad_id = int(
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        global_step = 0

        for epoch in range(self.num_epochs):
            set_seed(self.seed + epoch)
            self.model.train()
            epoch_loss_sum = 0.0
            step_count = 0

            progress_bar = tqdm(
                dataloader,
                desc=f"{dataset_name} | Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_main_process,
                leave=False,
            )

            for step, batch in enumerate(progress_bar):
                pre_bsz = batch["input_ids"].size(0)

                batch = mix_batch_with_replay(
                    batch,
                    self.replay_buffer,
                    self.replay_ratio if self.use_replay else 0.0,
                    pad_id,
                    self.device,
                )
                post_bsz = batch["input_ids"].size(0)

                if self.debug and step == 0 and self.accelerator.is_main_process:
                    valid_cnt = (batch["labels"] != -100).sum().item()
                    lengths = batch["attention_mask"].sum(dim=1)
                    if self.use_replay and len(self.replay_buffer) > 0:
                        n_replay = int(pre_bsz * self.replay_ratio)
                        n_replay = max(1, min(n_replay, pre_bsz - 1)) if pre_bsz >= 2 else 0
                    else:
                        n_replay = 0
                    is_left = assert_left_padded(
                        batch["input_ids"], batch["attention_mask"], pad_id
                    )
                    print(
                        f"[debug][train] step=0, "
                        f"pre_bsz={pre_bsz}, post_bsz={post_bsz}, "
                        f"replay_in_batch={n_replay}, "
                        f"valid_labels={valid_cnt}, "
                        f"seq_len={batch['input_ids'].size(1)}, "
                        f"left_padded={is_left}"
                    )
                    print(
                        f"  lengths (min/mean/max)="
                        f"{int(lengths.min())}/"
                        f"{float(lengths.float().mean()):.1f}/"
                        f"{int(lengths.max())}"
                    )

                valid_cnt = (batch["labels"] != -100).sum().item()
                if valid_cnt == 0:
                    if self.debug and self.accelerator.is_main_process:
                        print(f"[warn] step {step}: no valid labels, skip")
                    continue

                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                loss_val = float(loss.detach().item())
                global_step += 1
                epoch_loss_sum += loss_val
                step_count += 1

                if self.accelerator.is_main_process:
                    wandb.log({
                        "train/step": global_step,
                        "train/loss": loss_val,
                        "train/valid_labels": valid_cnt,
                        "train/replay_buffer_size": len(self.replay_buffer),
                        "epoch": epoch + 1,
                    })

                progress_bar.set_postfix({
                    "Loss": f"{loss_val:.4f}",
                    "Step": f"{step + 1}/{len(dataloader)}",
                    "ReplayBuf": len(self.replay_buffer),
                })

            avg_loss = epoch_loss_sum / max(step_count, 1)
            if self.accelerator.is_main_process:
                print(f"\n[{dataset_name}] Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")
                wandb.log({"epoch/avg_loss": avg_loss, "epoch": epoch + 1})

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._save_model(dataset_name, epoch + 1)

    # ─────────────────────────────────────────────────────────────────────
    def train_across_datasets(self):
        for stage_idx, dataset_file in enumerate(self.dataset_list):
            dataset_name = os.path.basename(dataset_file).split(".")[0]

            if self.accelerator.is_main_process:
                print(f"\n{'=' * 60}")
                print(
                    f"[Stage {stage_idx + 1}/{len(self.dataset_list)}] {dataset_name}"
                )
                print(
                    f"  Replay buffer before stage: {len(self.replay_buffer)} samples"
                )
                print(f"{'=' * 60}")

            dataset, dataloader = self.prepare_data(dataset_file)
            self.train_on_dataset(dataloader, dataset_name)
            self._update_replay_buffer(
                dataset=dataset,
                stage_idx=stage_idx,
                dataset_name=dataset_name,
            )

            del dataset, dataloader
            gc.collect()

        if self.wandb_initialized:
            wandb.finish()


if __name__ == "__main__":
    trainer = Trainer(
        model_name="meta-llama/Llama-3.2-1B",
        dataset_list=[
            "/home/jovyan/sumin_data/cp4llm/utils/data_storage/new-medical-kor-dataset.txt",
            "/home/jovyan/sumin_data/cp4llm/utils/data_storage/guidline_medical.txt",
            "/home/jovyan/sumin_data/cp4llm/utils/data_storage/new-legal-kor-dataset.txt",
            "/home/jovyan/sumin_data/cp4llm/utils/data_storage/eng-new-legal-dataset.txt",
        ],
        output_dir="/home/jovyan/sumin_data/saved_model/adam_replay/",
        batch_size=4,
        num_epochs=1,
        learning_rate=2e-5,
        max_length=256,
        chunk_size=64,
        gradient_accumulation_steps=16,
        num_workers=4,
        use_replay=True,
        replay_capacity=5000,
        replay_ratio=0.5,
        replay_sample_per_stage=2000,
        seed=777,
        debug=True,
        use_wandb=True,
    )
    trainer.load_model_and_tokenizer()
    trainer.train_across_datasets()
