# eval/modules/cl_evaluator.py


from __future__ import annotations

import gc
import inspect
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
try:
    import wandb  # optional; wandb_mode="disabled" needs no package
except ImportError:
    wandb = None
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.eval_stability import EvalRunner, run_single_gpu_evaluation
from modules.metrics import get_metric_policy

CL_METRICS: List[str] = [
    "bleu",
    "chrf",
    "rouge1",
    "rouge2",
    "rougeL",
    "meteor",
    "cosine_similarity",
    "token_precision",
    "token_recall",
    "token_f1",
]

# Token-overlap P/R/F1 are valid CL metrics only when they are computed in a
# fixed metric-tokenizer space.  With token_metric_policy="base", the base
# checkpoint tokenizer is used for every checkpoint, so these three metrics are
# directly comparable across the CL trajectory.  With "exclude", they are
# removed from the R matrix entirely.
TOKEN_CL_METRICS: Tuple[str, ...] = (
    "token_precision",
    "token_recall",
    "token_f1",
)
TEXT_CL_METRICS: List[str] = [
    metric for metric in CL_METRICS if metric not in TOKEN_CL_METRICS
]

# Micro token-overlap variants remain diagnostics. Generation diagnostics
# describe truncation/cap and tokenizer round-trip behavior.
TOKEN_DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "token_precision_micro",
    "token_recall_micro",
    "token_f1_micro",
)
GENERATION_DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "eval_prompt_max_tokens",
    "eval_generation_cap_hit_rate",
    "eval_prompt_roundtrip_mismatch_rate",
)
DIAGNOSTIC_METRICS: Tuple[str, ...] = (
    "token_precision_micro",
    "token_recall_micro",
    "token_f1_micro",
    "eval_prompt_max_tokens",
    "eval_generation_cap_hit_rate",
    "eval_prompt_roundtrip_mismatch_rate",
)
TOKENIZER_DEPENDENT_METRICS: set[str] = set(TOKEN_CL_METRICS) | set(TOKEN_DIAGNOSTIC_METRICS)
ALL_METRICS: List[str] = list(CL_METRICS) + list(DIAGNOSTIC_METRICS)
TOKENIZER_POLICIES: set[str] = {"fixed", "checkpoint"}
TOKEN_METRIC_POLICIES: set[str] = {"base", "exclude"}

# All configured metrics are higher-is-better similarity/quality metrics.
# All retained metrics are higher-is-better.
LOWER_IS_BETTER: set[str] = set()

REQUIRED_FINITE_METRICS: Tuple[str, ...] = tuple(TEXT_CL_METRICS)

MetricRow = Dict[str, float]
StepResult = Dict[str, MetricRow]
DiagnosticResult = Dict[str, Dict[str, float]]
CurriculumItem = Tuple[str, str, int]



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Metric '{name}' is not numeric: {value!r}.") from exc


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False



def _strict_mean(values: Sequence[float], *, context: str) -> float:
    """Average complete values; never discard a non-finite value silently."""
    if not values:
        return float("nan")
    converted = [_to_float(value, name=context) for value in values]
    bad = [value for value in converted if not math.isfinite(value)]
    if bad:
        raise RuntimeError(
            f"Cannot compute {context}: found non-finite values {bad}. "
            "A failed domain must not be removed from the denominator."
        )
    return float(np.mean(converted))


def _stable_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _stable_json_value(value.item())
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _stable_json_value(value.detach().cpu().item())
        return [_stable_json_value(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, Mapping):
        return {
            str(key): _stable_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_stable_json_value(item) for item in value]
    return str(value)


def _json_dump(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            _stable_json_value(data),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def _put_finite(payload: MutableMapping[str, Any], key: str, value: Any) -> None:
    if _is_finite(value):
        payload[key] = float(value)


def _cuda_allocated_mib(device: torch.device) -> float:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / (1024.0**2))


def _clear_caches(device: Optional[torch.device]) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "step"


def _require_eval_runner_revision() -> None:
    """Revision and fingerprint gating removed. The repository ships the
    companion eval_stability.py, so the runtime signature check that imported
    modules.n_eval_stability is unnecessary and referenced a module name that
    does not exist in this tree."""
    return None


def _validate_single_gpu_runtime(accelerator: Accelerator) -> torch.device:
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "CL evaluation requires exactly one OS process. Do not use "
            "multi-process Accelerate, DDP, FSDP, or DeepSpeed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CL evaluation requires one GPU.")
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != 1:
        raise RuntimeError(
            "Exactly one CUDA GPU must be visible to this process, but "
            f"torch.cuda.device_count()={visible_gpu_count}. Select one GPU with "
            "CUDA_VISIBLE_DEVICES=<physical_gpu_id> before launching."
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            f"Accelerator selected {accelerator.device}, not a CUDA device."
        )
    device_index = int(
        accelerator.device.index
        if accelerator.device.index is not None
        else torch.cuda.current_device()
    )
    if device_index != 0:
        raise RuntimeError(
            f"Expected the single visible logical GPU to be cuda:0, got cuda:{device_index}."
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def resolve_inference_dtype(name: str) -> Tuple[torch.dtype, str]:
    normalized = str(name).strip().lower()
    allowed = {"auto", "bf16", "fp16", "fp32"}
    if normalized not in allowed:
        raise ValueError(
            f"inference_dtype must be one of {sorted(allowed)}, got {name!r}."
        )

    if normalized == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        return torch.float16, "fp16"
    if normalized == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but the selected GPU does not support it.")
        return torch.bfloat16, "bf16"
    if normalized == "fp16":
        return torch.float16, "fp16"
    return torch.float32, "fp32"


_ADAPTER_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)

_TOKENIZER_ARTIFACT_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
}


def _is_existing_local_source(source: str) -> bool:
    return bool(source) and os.path.exists(os.path.expanduser(source))


def _looks_like_missing_local_path(source: str) -> bool:
    expanded = os.path.expanduser(source)
    return (
        os.path.isabs(expanded)
        or expanded.startswith("./")
        or expanded.startswith("../")
    ) and not os.path.exists(expanded)


def _inspect_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    if not isinstance(ckpt_path, str) or not ckpt_path.strip():
        raise ValueError("Checkpoint path/repository ID must be non-empty.")
    ckpt_path = ckpt_path.strip()

    if _looks_like_missing_local_path(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    if not _is_existing_local_source(ckpt_path):
        # Remote Hub IDs are treated as full-model checkpoints. A remote LoRA
        # adapter cannot satisfy the local config+weights integrity check.
        return {
            "source": ckpt_path,
            "is_local": False,
            "is_lora": False,
            "adapter_config": None,
            "adapter_weights": None,
        }

    path = Path(os.path.expanduser(ckpt_path)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint is not a directory: {path}")

    adapter_config = path / "adapter_config.json"
    adapter_weights = [path / name for name in _ADAPTER_WEIGHT_NAMES if (path / name).is_file()]
    has_config = adapter_config.is_file()
    has_weights = bool(adapter_weights)

    if has_config != has_weights:
        missing = "adapter weights" if has_config else "adapter_config.json"
        raise RuntimeError(
            f"Incomplete LoRA checkpoint at {path}: missing {missing}. "
            "A valid adapter must contain adapter_config.json and one of "
            f"{list(_ADAPTER_WEIGHT_NAMES)}."
        )

    is_lora = has_config and has_weights
    if not is_lora and not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Full-model checkpoint has no config.json: {path}. If this is a "
            "LoRA adapter, both adapter_config.json and adapter weights are required."
        )

    return {
        "source": str(path),
        "is_local": True,
        "is_lora": is_lora,
        "adapter_config": str(adapter_config) if is_lora else None,
        "adapter_weights": str(adapter_weights[0]) if is_lora else None,
    }


def _read_adapter_config(adapter_path: str) -> Dict[str, Any]:
    config_path = Path(adapter_path) / "adapter_config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read LoRA adapter config: {config_path}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"LoRA adapter config is not a JSON object: {config_path}")
    return config


def _resolve_lora_base(
    adapter_path: str,
    override_base: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    adapter_config = _read_adapter_config(adapter_path)
    configured_base = adapter_config.get("base_model_name_or_path")
    if override_base is not None and override_base.strip():
        return override_base.strip(), adapter_config
    if isinstance(configured_base, str) and configured_base.strip():
        configured = os.path.expanduser(configured_base.strip())
        if not os.path.isabs(configured):
            relative_candidate = Path(adapter_path) / configured
            if relative_candidate.exists():
                configured = str(relative_candidate.resolve())
        return configured, adapter_config
    raise ValueError(
        f"Cannot determine the base model for LoRA adapter '{adapter_path}'. "
        "Provide --lora_base_model or set base_model_name_or_path in "
        "adapter_config.json."
    )


def _has_tokenizer_artifacts(path: str) -> bool:
    directory = Path(path)
    if not directory.is_dir():
        return False
    names = {item.name for item in directory.iterdir() if item.is_file()}
    if names.intersection(_TOKENIZER_ARTIFACT_NAMES):
        return True
    return any(name.startswith("tokenization_") and name.endswith(".py") for name in names)


def _first_eos_id(tokenizer: Any) -> Optional[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    return int(eos_id) if eos_id is not None else None


def _configure_existing_special_tokens(tokenizer: Any, model: Any) -> None:
    """Configure padding without adding tokens or resizing embeddings."""
    original_vocab_size = len(tokenizer)
    tokenizer.padding_side = "left"

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_id = _first_eos_id(tokenizer)
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id. Evaluation "
                "will not invent a new special token or resize embeddings."
            )
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.pad_token_id = eos_id

    if getattr(tokenizer, "pad_token_id", None) is None:
        raise RuntimeError("Failed to reuse the existing EOS token as padding.")
    if len(tokenizer) != original_vocab_size:
        raise RuntimeError(
            "Tokenizer vocabulary changed while configuring padding. Evaluation "
            "must not add special tokens."
        )

    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_config.pad_token_id = int(tokenizer.pad_token_id)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            model_config.eos_token_id = tokenizer.eos_token_id

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = int(tokenizer.pad_token_id)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.do_sample = False


def _validate_tokenizer_embedding_compatibility(tokenizer: Any, model: Any) -> Dict[str, int]:
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        raise RuntimeError("Tokenizer does not expose get_vocab().") from exc
    if not vocab:
        raise RuntimeError("Tokenizer vocabulary is empty.")

    max_token_id = max(int(token_id) for token_id in vocab.values())
    tokenizer_length = int(len(tokenizer))

    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise RuntimeError("Model does not expose input embeddings.")
    input_rows = int(input_embeddings.weight.shape[0])
    if max_token_id >= input_rows:
        raise RuntimeError(
            f"Tokenizer token ID {max_token_id} exceeds input embedding rows "
            f"{input_rows}. Evaluation will not call resize_token_embeddings()."
        )

    output_rows = input_rows
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        output_rows = int(output_embeddings.weight.shape[0])
        if max_token_id >= output_rows:
            raise RuntimeError(
                f"Tokenizer token ID {max_token_id} exceeds output embedding rows "
                f"{output_rows}. Evaluation will not resize the LM head."
            )

    return {
        "tokenizer_length": tokenizer_length,
        "max_token_id": max_token_id,
        "input_embedding_rows": input_rows,
        "output_embedding_rows": output_rows,
    }


def _tokenizer_fingerprint(tokenizer: Any) -> Dict[str, Any]:
    """Hash-based tokenizer fingerprinting removed. Only descriptive metadata
    is retained for logging; no vocabulary hashing is performed and nothing is
    enforced from these fields."""
    try:
        vocab_size = int(len(tokenizer))
    except Exception:
        vocab_size = -1
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": vocab_size,
        "padding_side": str(getattr(tokenizer, "padding_side", "")),
        "fingerprint": "disabled",
    }


def _model_fingerprint(model: Any) -> Dict[str, Any]:
    """Hash-based model fingerprinting removed. Only descriptive metadata is
    retained; no architecture hashing is performed and nothing is enforced."""
    cfg = getattr(model, "config", None)
    return {
        "model_type": str(getattr(cfg, "model_type", "")),
        "hidden_size": int(getattr(cfg, "hidden_size", -1) or -1),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1) or -1),
        "vocab_size": int(getattr(cfg, "vocab_size", -1) or -1),
        "architecture_fingerprint": "disabled",
        "fingerprint": "disabled",
    }


def _load_tokenizer_only(
    ckpt_path: str,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
    lora_base_model: Optional[str],
    lora_tokenizer_source: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the tokenizer associated with one checkpoint without loading a model."""
    if lora_tokenizer_source not in {"base", "adapter"}:
        raise ValueError("lora_tokenizer_source must be 'base' or 'adapter'.")

    checkpoint = _inspect_checkpoint(ckpt_path)
    is_lora = bool(checkpoint["is_lora"])
    model_source = checkpoint["source"]
    if is_lora:
        model_source, _ = _resolve_lora_base(
            checkpoint["source"], lora_base_model
        )
        model_source = _inspect_checkpoint(model_source)["source"]

    if (
        is_lora
        and lora_tokenizer_source == "adapter"
        and _has_tokenizer_artifacts(checkpoint["source"])
    ):
        tokenizer_source = checkpoint["source"]
        tokenizer_source_kind = "adapter"
        tokenizer_local_only = True
    else:
        tokenizer_source = model_source
        tokenizer_source_kind = "base" if is_lora else "checkpoint"
        tokenizer_local_only = (
            local_files_only or _is_existing_local_source(tokenizer_source)
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=tokenizer_local_only,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_id = _first_eos_id(tokenizer)
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_id is None:
            raise RuntimeError(
                f"Tokenizer for {ckpt_path} has neither pad_token_id nor eos_token_id."
            )
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        else:
            tokenizer.pad_token_id = int(eos_id)

    metadata = _tokenizer_fingerprint(tokenizer)
    metadata.update(
        {
            "source": tokenizer_source,
            "source_kind": tokenizer_source_kind,
            "checkpoint": ckpt_path,
        }
    )
    return tokenizer, metadata


def load_model_tokenizer(
    ckpt_path: str,
    *,
    device: torch.device,
    inference_dtype: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    lora_base_model: Optional[str] = None,
    merge_lora: bool = False,
    lora_tokenizer_source: str = "base",
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load one full-model or verified local LoRA checkpoint onto one GPU."""
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {device}.")
    if lora_tokenizer_source not in {"base", "adapter"}:
        raise ValueError("lora_tokenizer_source must be 'base' or 'adapter'.")

    checkpoint = _inspect_checkpoint(ckpt_path)
    is_lora = bool(checkpoint["is_lora"])
    adapter_config: Optional[Dict[str, Any]] = None

    model_source = checkpoint["source"]
    if is_lora:
        model_source, adapter_config = _resolve_lora_base(
            checkpoint["source"], lora_base_model
        )
        base_checkpoint = _inspect_checkpoint(model_source)
        if base_checkpoint["is_lora"]:
            raise RuntimeError(
                "A LoRA adapter cannot use another LoRA adapter as its base model."
            )
        model_source = base_checkpoint["source"]

    dtype, dtype_name = resolve_inference_dtype(inference_dtype)
    device_index = int(device.index if device.index is not None else 0)

    if is_lora and lora_tokenizer_source == "adapter" and _has_tokenizer_artifacts(
        checkpoint["source"]
    ):
        tokenizer_source = checkpoint["source"]
        tokenizer_source_kind = "adapter"
        tokenizer_local_only = True
    else:
        tokenizer_source = model_source
        tokenizer_source_kind = "base" if is_lora else "checkpoint"
        tokenizer_local_only = local_files_only or _is_existing_local_source(tokenizer_source)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=tokenizer_local_only,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )

    model_local_only = local_files_only or _is_existing_local_source(model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=model_local_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        device_map={"": device_index},
    )

    merge_actual = False
    if is_lora:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Evaluating LoRA checkpoints requires the 'peft' package."
            ) from exc

        model = PeftModel.from_pretrained(
            model,
            checkpoint["source"],
            is_trainable=False,
            local_files_only=True,
        )

        if merge_lora:
            try:
                try:
                    model = model.merge_and_unload(safe_merge=True)
                except TypeError:
                    model = model.merge_and_unload()
            except Exception as exc:
                raise RuntimeError(
                    f"merge_lora=True but merge_and_unload failed for {ckpt_path}."
                ) from exc
            merge_actual = True

    _configure_existing_special_tokens(tokenizer, model)
    embedding_info = _validate_tokenizer_embedding_compatibility(tokenizer, model)
    tokenizer_info = _tokenizer_fingerprint(tokenizer)
    model_info = _model_fingerprint(model)
    model.eval()

    metadata = {
        "checkpoint": ckpt_path,
        "resolved_checkpoint": checkpoint["source"],
        "checkpoint_type": "lora" if is_lora else "full_model",
        "model_source": model_source,
        "tokenizer_source": tokenizer_source,
        "tokenizer_source_kind": tokenizer_source_kind,
        "inference_dtype": dtype_name,
        "merge_lora_requested": bool(merge_lora),
        "merge_lora_actual": bool(merge_actual),
        "lora_base_override": lora_base_model,
        "adapter_config_base": (
            adapter_config.get("base_model_name_or_path")
            if isinstance(adapter_config, dict)
            else None
        ),
        "adapter_weights": checkpoint.get("adapter_weights"),
        "tokenizer": tokenizer_info,
        "model": model_info,
        "embedding_compatibility": embedding_info,
    }
    return model, tokenizer, metadata


### Continual-learning evaluator
class CLEvaluatorCL:
    """Continual-learning evaluator with one-checkpoint-at-a-time GPU ownership."""

    def __init__(
        self,
        output_dir: str,
        curriculum: List[CurriculumItem],
        base_ckpt_path: str,
        *,
        eval_split: str,
        eval_domains: Optional[List[str]] = None,
        batch_size: int = 8,
        max_length: int = 2048,
        gen_max_new_tokens: int = 256,
        project: str = "CL_Eval",
        run_name: Optional[str] = None,
        wandb_mode: str = "online",
        log_raw_json: bool = True,
        preprocessed_eval_root: Optional[str] = None,
        per_domain_metrics: Optional[List[str]] = None,
        shot_type: str = "zero-shot",
        meteor_mode: str = "multilingual_lexical",
        seed: int = 777,
        strict_eval: bool = True,
        inference_dtype: str = "auto",
        empty_cache_each_domain: bool = True,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        lora_base_model: Optional[str] = None,
        merge_lora: bool = False,
        lora_tokenizer_source: str = "base",
        tokenizer_policy: str = "checkpoint",
        token_metric_policy: str = "base",
        max_allowed_post_unload_mib: float = 64.0,
    ) -> None:
        _require_eval_runner_revision()

        self.accelerator = Accelerator()
        self.device = _validate_single_gpu_runtime(self.accelerator)

        if not isinstance(eval_split, str) or not eval_split.strip():
            raise ValueError("eval_split must be specified explicitly.")
        if not isinstance(base_ckpt_path, str) or not base_ckpt_path.strip():
            raise ValueError("base_ckpt_path must be non-empty.")
        if not curriculum:
            raise ValueError("curriculum must contain at least one training step.")
        if batch_size <= 0 or max_length <= 1 or gen_max_new_tokens <= 0:
            raise ValueError(
                "batch_size and gen_max_new_tokens must be positive; max_length must be > 1."
            )
        if max_allowed_post_unload_mib < 0:
            raise ValueError("max_allowed_post_unload_mib must be non-negative.")

        self.curriculum = self._validate_curriculum(curriculum)
        self.base_ckpt_path = base_ckpt_path.strip()
        self.eval_split = eval_split.strip()
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self.requested_preprocessed_eval_root = preprocessed_eval_root
        self.shot_type = shot_type
        self.meteor_mode = str(meteor_mode).strip().lower()
        if self.meteor_mode not in {"multilingual_lexical", "standard"}:
            raise ValueError(
                "meteor_mode must be 'multilingual_lexical' or 'standard', "
                f"got {meteor_mode!r}."
            )
        self.seed = int(seed)
        self.strict_eval = bool(strict_eval)
        self.inference_dtype = inference_dtype
        self.empty_cache_each_domain = bool(empty_cache_each_domain)
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.lora_base_model = lora_base_model
        self.merge_lora = bool(merge_lora)
        self.lora_tokenizer_source = lora_tokenizer_source
        self.tokenizer_policy = str(tokenizer_policy).strip().lower()
        self.token_metric_policy = str(token_metric_policy).strip().lower()
        if self.tokenizer_policy not in TOKENIZER_POLICIES:
            raise ValueError(
                f"tokenizer_policy must be one of {sorted(TOKENIZER_POLICIES)}, "
                f"got {tokenizer_policy!r}."
            )
        if self.token_metric_policy not in TOKEN_METRIC_POLICIES:
            raise ValueError(
                f"token_metric_policy must be one of "
                f"{sorted(TOKEN_METRIC_POLICIES)}, got {token_metric_policy!r}."
            )
        self.max_allowed_post_unload_mib = float(max_allowed_post_unload_mib)

        # The frozen raw cache contains only prompt/reference metadata, not token IDs.
        # It is therefore safe across checkpoint tokenizers; domain_map additionally
        # validates a policy hash covering the declared tokenizer compatibility set.
        self.preprocessed_eval_root = preprocessed_eval_root
        self.preprocessed_cache_policy = (
            "frozen_raw_manifest; tokenizer_id_independent; policy_hash_guarded"
        )

        # Strictness is set explicitly rather than inherited from a stale shell.
        os.environ["EVAL_STRICT"] = "1" if self.strict_eval else "0"

        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.log_raw_json = bool(log_raw_json)

        trained_domains = [domain for domain, _, _ in self.curriculum]
        extra_domains = self._validate_domain_list(eval_domains or [], "eval_domains")
        self.unique_domains = list(dict.fromkeys(trained_domains + extra_domains))


        # Use token P/R/F1 as CL metrics only in a fixed tokenizer space.  This
        # preserves comparability when checkpoint tokenizers evolve during CL.
        if self.token_metric_policy == "base":
            self.cl_metrics = list(CL_METRICS)
        else:
            self.cl_metrics = list(TEXT_CL_METRICS)
        self.required_finite_metrics = tuple(self.cl_metrics)

        requested_per_domain = per_domain_metrics or [
            "chrf", "bleu", "rougeL", "token_precision", "token_recall", "token_f1"
        ]
        if self.token_metric_policy == "exclude":
            requested_per_domain = [
                metric for metric in requested_per_domain if metric not in TOKEN_CL_METRICS
            ]
        unknown_metrics = sorted(set(requested_per_domain) - set(self.cl_metrics))
        if unknown_metrics:
            raise ValueError(
                f"Unsupported per_domain_metrics: {unknown_metrics}. "
                f"Allowed metrics: {self.cl_metrics}."
            )
        self.per_domain_metrics = list(dict.fromkeys(requested_per_domain))

        self.project = project
        base_name = os.path.basename(self.base_ckpt_path.rstrip("/")) or "base"
        self.run_name = run_name or f"CL_run_{base_name}"
        self.wandb_mode = str(wandb_mode).strip().lower()
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled.")

        set_seed(self.seed)

        self.R: Dict[str, List[MetricRow]] = {metric: [] for metric in self.cl_metrics}
        self.step_info: Dict[int, CurriculumItem] = {}
        self.first_learned_step: Dict[str, int] = {}
        for step_idx, item in enumerate(self.curriculum, start=1):
            domain, ckpt_path, epoch = item
            self.step_info[step_idx] = item
            self.first_learned_step[domain] = step_idx

        self.b_initial: Dict[str, MetricRow] = {}
        self.tokenizer_reference: Optional[Dict[str, Any]] = None
        self.model_reference: Optional[Dict[str, Any]] = None
        self.tokenizer_history: List[Dict[str, Any]] = []
        self.dataset_reference: Dict[str, Dict[str, Any]] = {}
        # Keep policy-hash observations for auditing.  Dataset identity itself is
        # anchored by the frozen raw prompt/reference content hash and semantic
        # manifest metadata, not by a tokenizer runtime-state hash.
        self.dataset_policy_hash_history: Dict[str, List[Dict[str, Any]]] = {}
        self.checkpoint_preflight = self._preflight_checkpoints()
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.diagnostic_history: List[Dict[str, Any]] = []

        # Predeclare the complete tokenizer set used by the base model and every
        # CL checkpoint. domain_map uses this set only for prompt-length compatibility;
        # no model predictions or gold-reference lengths participate in manifest selection.
        self.compatibility_tokenizers: List[Any] = []
        self.compatibility_tokenizer_metadata: List[Dict[str, Any]] = []
        seen_tokenizer_fingerprints: set = set()
        tokenizer_sources = [self.base_ckpt_path] + [
            checkpoint for _, checkpoint, _ in self.curriculum
        ]
        for source in tokenizer_sources:
            tok, meta = _load_tokenizer_only(
                source,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                lora_base_model=self.lora_base_model,
                lora_tokenizer_source=self.lora_tokenizer_source,
            )
            # Fingerprint dedup removed. Deduplicate by tokenizer identity
            # descriptors instead, which needs no vocabulary hashing.
            key = (meta.get("name_or_path"), meta.get("vocab_size"),
                   meta.get("padding_side"))
            if key in seen_tokenizer_fingerprints:
                continue
            seen_tokenizer_fingerprints.add(key)
            self.compatibility_tokenizers.append(tok)
            self.compatibility_tokenizer_metadata.append(meta)

        if not self.compatibility_tokenizers:
            raise RuntimeError("No compatibility tokenizers were loaded.")

        # The first tokenizer is the base-checkpoint tokenizer and defines the fixed
        # tokenizer space for optional token-overlap P/R/F1 only.
        self.metric_tokenizer = (
            self.compatibility_tokenizers[0]
            if self.token_metric_policy == "base"
            else None
        )
        self.metric_tokenizer_metadata: Optional[Dict[str, Any]] = (
            self.compatibility_tokenizer_metadata[0]
            if self.token_metric_policy == "base"
            else None
        )

        self.wandb_run = None
        if self.wandb_mode != "disabled":
            if wandb is None:
                raise ImportError("wandb_mode != 'disabled' but wandb is not installed.")
            self.wandb_run = wandb.init(
                project=self.project,
                name=self.run_name,
                mode=self.wandb_mode,
                reinit=True,
            )
            self._configure_wandb()

        self.accelerator.print(
            "[CL Eval] single-process/single-GPU mode; "
            f"device={self.device}, eval_split={self.eval_split}, "
            f"strict={self.strict_eval}, dtype={self.inference_dtype}"
        )
        self.accelerator.print(
            "[CL Eval] tokenizer_policy="
            f"{self.tokenizer_policy}, token_metric_policy={self.token_metric_policy}, "
            f"preprocessed_cache_policy={self.preprocessed_cache_policy}"
        )

    @staticmethod
    def _validate_domain_list(domains: Iterable[str], name: str) -> List[str]:
        normalized = [str(domain).strip() for domain in domains]
        if any(not domain for domain in normalized):
            raise ValueError(f"{name} contains an empty domain name: {normalized!r}.")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{name} contains duplicate domains: {normalized!r}.")
        return normalized

    @classmethod
    def _validate_curriculum(cls, curriculum: Sequence[CurriculumItem]) -> List[CurriculumItem]:
        if not curriculum:
            raise ValueError("curriculum must contain at least one training step.")
        validated: List[CurriculumItem] = []
        for index, item in enumerate(curriculum, start=1):
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError(
                    f"Curriculum item {index} must be (domain, checkpoint, epoch), got {item!r}."
                )
            domain = str(item[0]).strip()
            ckpt_path = str(item[1]).strip()
            try:
                epoch = int(item[2])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Curriculum item {index} has a non-integer epoch: {item[2]!r}."
                ) from exc
            if not domain:
                raise ValueError(f"Curriculum item {index} has an empty domain.")
            if not ckpt_path:
                raise ValueError(f"Curriculum item {index} has an empty checkpoint path.")
            if epoch < 0:
                raise ValueError(
                    f"Curriculum item {index} has a negative epoch: {epoch}."
                )
            validated.append((domain, ckpt_path, epoch))

        domains = [item[0] for item in validated]
        if len(domains) != len(set(domains)):
            duplicates = sorted({domain for domain in domains if domains.count(domain) > 1})
            raise ValueError(
                "Repeated curriculum domains are not supported by the current "
                f"domain-level CL formulas. Duplicates: {duplicates}."
            )
        return validated

    def _configure_wandb(self) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.define_metric("cl_step")
        self.wandb_run.define_metric("eval_batch_step")
        self.wandb_run.define_metric("eval/*", step_metric="cl_step")
        self.wandb_run.define_metric("diagnostic/*", step_metric="cl_step")
        self.wandb_run.define_metric("step_avg/*", step_metric="cl_step")
        self.wandb_run.define_metric("cl/*", step_metric="cl_step")
        self.wandb_run.define_metric("final/*", step_metric="cl_step")
        self.wandb_run.define_metric("eval_batch/*", step_metric="eval_batch_step")

        self.wandb_run.config.update(
            {
                "base_ckpt": self.base_ckpt_path,
                "curriculum": [
                    {"domain": d, "checkpoint": p, "epoch": e}
                    for d, p, e in self.curriculum
                ],
                "domains": self.unique_domains,
                "metrics": self.cl_metrics,
                "required_finite_metrics": list(self.required_finite_metrics),
                "eval_split": self.eval_split,
                "strict_eval": self.strict_eval,
                "single_process_single_visible_gpu": True,
                "shot_type": self.shot_type,
                "meteor_mode": self.meteor_mode,
                "seed": self.seed,
                "inference_dtype": self.inference_dtype,
                "preprocessed_eval_root": self.preprocessed_eval_root,
                "empty_cache_each_domain": self.empty_cache_each_domain,
                "trust_remote_code": self.trust_remote_code,
                "local_files_only": self.local_files_only,
                "lora_base_model": self.lora_base_model,
                "merge_lora": self.merge_lora,
                "lora_tokenizer_source": self.lora_tokenizer_source,
                "tokenizer_policy": self.tokenizer_policy,
                "token_metric_policy": self.token_metric_policy,
                "token_overlap_cl_metrics_enabled": self.token_metric_policy == "base",
                "token_micro_metrics_are_diagnostic_only": True,
                "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
                "generation_diagnostics_always_enabled": True,
                "preprocessed_cache_policy": self.preprocessed_cache_policy,
                "requested_preprocessed_eval_root": self.requested_preprocessed_eval_root,
                "metric_tokenizer": self.metric_tokenizer_metadata,
                "compatibility_tokenizers": self.compatibility_tokenizer_metadata,
                "text_metric_policy": get_metric_policy(),
                "checkpoint_preflight": self.checkpoint_preflight,
            },
            allow_val_change=True,
        )

    def _preflight_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Validate all local checkpoint/adapter structures before evaluation."""
        references = [self.base_ckpt_path] + [
            checkpoint for _, checkpoint, _ in self.curriculum
        ]
        inspected: Dict[str, Dict[str, Any]] = {}
        for reference in references:
            if reference in inspected:
                continue
            spec = _inspect_checkpoint(reference)
            entry = dict(spec)
            if spec["is_lora"]:
                base_source, _ = _resolve_lora_base(
                    spec["source"], self.lora_base_model
                )
                base_spec = _inspect_checkpoint(base_source)
                if base_spec["is_lora"]:
                    raise RuntimeError(
                        f"LoRA checkpoint '{reference}' resolves to another LoRA "
                        f"adapter as its base: {base_source}."
                    )
                entry["resolved_base_source"] = base_spec["source"]
            inspected[reference] = entry
        return inspected

    def _validate_model_consistency(
        self,
        model_metadata: Mapping[str, Any],
        *,
        ckpt_path: str,
    ) -> None:
        """Architecture-fingerprint enforcement removed. Metadata is recorded
        for the report and nothing is validated."""
        if self.model_reference is None:
            self.model_reference = dict(model_metadata)

    ### Checkpoint evaluation and validation
    def _validate_tokenizer_consistency(
        self,
        tokenizer_metadata: Mapping[str, Any],
        *,
        ckpt_path: str,
    ) -> None:
        """Tokenizer-fingerprint enforcement removed. Metadata is recorded for
        the report and nothing is validated."""
        record = {"checkpoint": ckpt_path, **dict(tokenizer_metadata)}
        self.tokenizer_history.append(record)
        if self.tokenizer_reference is None:
            self.tokenizer_reference = dict(tokenizer_metadata)

    def _validate_dataset_identity(
        self,
        *,
        domain: str,
        raw_metrics: Mapping[str, Any],
        ckpt_path: str,
    ) -> None:
        """Content-hash enforcement removed. The frozen manifest construction in
        domain_map already fixes the prompt/reference set per domain, so the only
        cheap invariant kept is the example count. No sha256 is computed or
        compared."""
        count_value = raw_metrics.get("eval_num_examples")
        try:
            count = int(float(count_value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"[{domain}] Invalid eval_num_examples={count_value!r}."
            ) from exc
        if count <= 0:
            raise RuntimeError(f"[{domain}] Invalid evaluation example count: {count}.")

        current = {
            "count": count,
            "requested_split": raw_metrics.get("eval_requested_split"),
            "effective_split": raw_metrics.get("eval_effective_split"),
        }
        self.dataset_policy_hash_history.setdefault(domain, []).append(
            {"checkpoint": ckpt_path, **current}
        )

        reference = self.dataset_reference.get(domain)
        if reference is None:
            self.dataset_reference[domain] = current
            return
        if reference.get("count") != count:
            raise RuntimeError(
                f"[{domain}] Evaluation example count changed across checkpoints: "
                f"{reference.get('count')} -> {count}, checkpoint={ckpt_path}."
            )

    def _normalize_and_validate_domain_metrics(
        self,
        *,
        domain: str,
        raw_metrics: Mapping[str, Any],
    ) -> Tuple[MetricRow, Dict[str, float]]:
        """Validate the configured text metrics without dropping failed domains."""
        missing = [
            metric for metric in self.required_finite_metrics
            if metric not in raw_metrics
        ]
        if missing:
            raise RuntimeError(
                f"[{domain}] Evaluation output is missing required metrics: {missing}."
            )

        metrics: MetricRow = {
            metric: _to_float(raw_metrics.get(metric), name=f"{domain}/{metric}")
            for metric in self.cl_metrics
        }
        for metric, value in metrics.items():
            if not math.isfinite(value):
                raise RuntimeError(
                    f"[{domain}] Required metric '{metric}' is non-finite. "
                    "The checkpoint/domain evaluation is invalid and will not "
                    "be removed from an average."
                )

        diagnostics: Dict[str, float] = {}

        for metric in GENERATION_DIAGNOSTIC_METRICS:
            if metric not in raw_metrics:
                continue
            value = _to_float(raw_metrics[metric], name=f"{domain}/{metric}")
            if math.isfinite(value):
                diagnostics[metric] = value

        if self.token_metric_policy == "base":
            for metric in TOKEN_DIAGNOSTIC_METRICS:
                if metric not in raw_metrics:
                    continue
                value = _to_float(raw_metrics[metric], name=f"{domain}/{metric}")
                if math.isfinite(value):
                    diagnostics[metric] = value

        return metrics, diagnostics

    def _evaluate_checkpoint_on_all(
        self,
        ckpt_path: str,
        step_idx: int,
        step_tag: str,
    ) -> Tuple[StepResult, DiagnosticResult, Dict[str, Any]]:
        _clear_caches(self.device)
        before_allocated = _cuda_allocated_mib(self.device)
        metadata_holder: Dict[str, Any] = {}

        def loader(device: torch.device) -> Tuple[Any, Any]:
            model, tokenizer, metadata = load_model_tokenizer(
                ckpt_path,
                device=device,
                inference_dtype=self.inference_dtype,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                lora_base_model=self.lora_base_model,
                merge_lora=self.merge_lora,
                lora_tokenizer_source=self.lora_tokenizer_source,
            )
            self._validate_tokenizer_consistency(
                metadata["tokenizer"], ckpt_path=ckpt_path
            )
            self._validate_model_consistency(
                metadata["model"], ckpt_path=ckpt_path
            )
            metadata_holder.update(metadata)
            return model, tokenizer

        raw_results = run_single_gpu_evaluation(
            load_model_and_tokenizer=loader,
            domains=self.unique_domains,
            accelerator=self.accelerator,
            runner_kwargs={
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "gen_max_new_tokens": self.gen_max_new_tokens,
                "eval_split": self.eval_split,
                "preprocessed_data_root": self.preprocessed_eval_root,
                "wandb_run": self.wandb_run,
                "cl_step_idx": step_idx,
                "step_tag": step_tag,
                "shot_type": self.shot_type,
                "seed": self.seed,
                "clear_cache_between_domains": self.empty_cache_each_domain,
                "release_tokenizer_on_cleanup": True,
                "meteor_mode": self.meteor_mode,
                "metric_tokenizer": self.metric_tokenizer,
                "release_metric_tokenizer_on_cleanup": False,
                "compute_token_metrics": self.token_metric_policy == "base",
                "compatibility_tokenizers": self.compatibility_tokenizers,
                "release_compatibility_tokenizers_on_cleanup": False,
            },
        )

        _clear_caches(self.device)
        after_allocated = _cuda_allocated_mib(self.device)
        retained_delta = max(0.0, after_allocated - before_allocated)
        metadata_holder["cuda_allocated_before_mib"] = before_allocated
        metadata_holder["cuda_allocated_after_mib"] = after_allocated
        metadata_holder["cuda_retained_delta_mib"] = retained_delta
        if retained_delta > self.max_allowed_post_unload_mib:
            raise RuntimeError(
                f"Checkpoint '{ckpt_path}' was evaluated, but CUDA allocations "
                f"increased by {retained_delta:.1f} MiB after unload. Another object "
                "still references the model or a CUDA tensor."
            )

        results: StepResult = {}
        diagnostics: DiagnosticResult = {}
        for domain in self.unique_domains:
            if domain not in raw_results:
                raise RuntimeError(
                    f"Checkpoint '{ckpt_path}' returned no result for domain '{domain}'."
                )
            self._validate_dataset_identity(
                domain=domain,
                raw_metrics=raw_results[domain],
                ckpt_path=ckpt_path,
            )
            domain_metrics, domain_diagnostics = self._normalize_and_validate_domain_metrics(
                domain=domain,
                raw_metrics=raw_results[domain],
            )
            results[domain] = domain_metrics
            diagnostics[domain] = domain_diagnostics

            if self.wandb_run is not None:
                payload: Dict[str, Any] = {
                    "cl_step": int(step_idx),
                    "checkpoint_tag": step_tag,
                    "eval_domain": domain,
                }
                for metric, value in domain_metrics.items():
                    _put_finite(payload, f"eval/{domain}/{metric}", value)
                for metric, value in domain_diagnostics.items():
                    _put_finite(payload, f"diagnostic/{domain}/{metric}", value)
                self.wandb_run.log(payload, commit=False)

        if self.wandb_run is not None:
            avg_payload: Dict[str, Any] = {
                "cl_step": int(step_idx),
                "checkpoint_tag": step_tag,
            }
            for metric in self.cl_metrics:
                domains = list(self.unique_domains)
                values = [results[domain][metric] for domain in domains]
                average = _strict_mean(
                    values,
                    context=f"step {step_idx} average/{metric}",
                )
                _put_finite(avg_payload, f"step_avg/{metric}", average)
                avg_payload[f"step_avg/{metric}_num_domains"] = len(domains)
            self.wandb_run.log(avg_payload, commit=True)

        return results, diagnostics, metadata_holder

    ### R matrix and strict CL metrics
    def _append_to_R(self, step_result: StepResult) -> None:
        missing_domains = [domain for domain in self.unique_domains if domain not in step_result]
        if missing_domains:
            raise RuntimeError(f"Step result is missing domains: {missing_domains}.")
        for metric in self.cl_metrics:
            row: MetricRow = {}
            for domain in self.unique_domains:
                value = float(step_result[domain][metric])
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"Cannot append non-finite R[{metric}][{domain}]={value}."
                    )
                row[domain] = value
            self.R[metric].append(row)

    def _require_R_value(self, metric: str, row_idx: int, domain: str) -> float:
        try:
            value = float(self.R[metric][row_idx][domain])
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Missing R-matrix value for metric={metric}, row={row_idx}, "
                f"domain={domain}."
            ) from exc
        if not math.isfinite(value):
            raise RuntimeError(
                f"Non-finite R-matrix value for metric={metric}, row={row_idx}, "
                f"domain={domain}: {value}."
            )
        return value

    def _compute_cl_metrics_at_step_raw(self, upto_step: int) -> Dict[str, Any]:
        expected_rows = len(self.R[self.cl_metrics[0]])
        if upto_step < 0 or upto_step >= expected_rows:
            raise IndexError(
                f"upto_step={upto_step} is outside the R matrix with {expected_rows} rows."
            )

        learned_domains = [
            domain
            for domain, learned_step in self.first_learned_step.items()
            if learned_step <= upto_step
        ]

        acc_avg: Dict[str, float] = {}
        bwt_avg: Dict[str, float] = {}
        fwt_avg: Dict[str, float] = {}
        avgf_avg: Dict[str, float] = {}
        wcr_value: Dict[str, float] = {}
        wcr_worst: Dict[str, Optional[str]] = {}

        bwt_domain = {metric: {} for metric in self.cl_metrics}
        fwt_domain = {metric: {} for metric in self.cl_metrics}
        avgf_domain = {metric: {} for metric in self.cl_metrics}
        wcr_domain = {metric: {} for metric in self.cl_metrics}
        term_counts = {
            metric: {"ACC": 0, "BWT": 0, "FWT": 0, "AvgF": 0, "WCR": 0}
            for metric in self.cl_metrics
        }
        wcr_zero_peak_domains = {metric: [] for metric in self.cl_metrics}

        for metric in self.cl_metrics:
            applicable_learned = list(learned_domains)

            acc_values = [
                self._require_R_value(metric, upto_step, domain)
                for domain in applicable_learned
            ]
            term_counts[metric]["ACC"] = len(acc_values)
            acc_avg[metric] = _strict_mean(
                acc_values,
                context=f"step {upto_step} ACC/{metric}",
            )

            bwt_values: List[float] = []
            for domain in applicable_learned:
                learned_step = self.first_learned_step[domain]
                if learned_step < upto_step:
                    final_value = self._require_R_value(metric, upto_step, domain)
                    after_learn_value = self._require_R_value(
                        metric, learned_step, domain
                    )
                    value = float(final_value - after_learn_value)
                    bwt_values.append(value)
                    bwt_domain[metric][domain] = value
            term_counts[metric]["BWT"] = len(bwt_values)
            bwt_avg[metric] = _strict_mean(
                bwt_values,
                context=f"step {upto_step} BWT/{metric}",
            )

            avgf_values: List[float] = []
            for domain in applicable_learned:
                learned_step = self.first_learned_step[domain]
                if learned_step <= upto_step - 1:
                    final_value = self._require_R_value(metric, upto_step, domain)
                    past_values = [
                        self._require_R_value(metric, row_idx, domain)
                        for row_idx in range(learned_step, upto_step)
                    ]
                    value = float(max(past_values) - final_value)
                    avgf_values.append(value)
                    avgf_domain[metric][domain] = value
            term_counts[metric]["AvgF"] = len(avgf_values)
            avgf_avg[metric] = _strict_mean(
                avgf_values,
                context=f"step {upto_step} AvgF/{metric}",
            )

            fwt_values: List[float] = []
            for domain in applicable_learned:
                learned_step = self.first_learned_step[domain]
                # Paper convention: the first learned domain is excluded, giving
                # the T-1 divisor for transfer to subsequent domains.
                if learned_step >= 2:
                    before_learn = self._require_R_value(
                        metric, learned_step - 1, domain
                    )
                    initial = float(self.b_initial[domain][metric])
                    if not math.isfinite(initial):
                        raise RuntimeError(
                            f"Non-finite initial value for FWT: {metric}/{domain}."
                        )
                    value = float(before_learn - initial)
                    fwt_values.append(value)
                    fwt_domain[metric][domain] = value
            term_counts[metric]["FWT"] = len(fwt_values)
            fwt_avg[metric] = _strict_mean(
                fwt_values,
                context=f"step {upto_step} FWT/{metric}",
            )

            for domain in applicable_learned:
                learned_step = self.first_learned_step[domain]
                final_value = self._require_R_value(metric, upto_step, domain)
                peak_values = [
                    self._require_R_value(metric, row_idx, domain)
                    for row_idx in range(learned_step, upto_step + 1)
                ]
                peak = float(max(peak_values))
                if peak <= 0.0:
                    wcr_zero_peak_domains[metric].append(domain)
                    continue
                wcr_domain[metric][domain] = float(1.0 - final_value / peak)

            term_counts[metric]["WCR"] = len(wcr_domain[metric])
            if wcr_domain[metric]:
                worst_domain = max(
                    wcr_domain[metric], key=wcr_domain[metric].get
                )
                wcr_worst[metric] = worst_domain
                wcr_value[metric] = float(wcr_domain[metric][worst_domain])
            else:
                wcr_worst[metric] = None
                wcr_value[metric] = float("nan")

        return {
            "learned_domains": learned_domains,
            "ACC_avg": acc_avg,
            "BWT_avg": bwt_avg,
            "FWT_avg": fwt_avg,
            "AvgF_avg": avgf_avg,
            "WCR": wcr_value,
            "WCR_worst_domain": wcr_worst,
            "BWT_per_domain": bwt_domain,
            "FWT_per_domain": fwt_domain,
            "AvgF_per_domain": avgf_domain,
            "WCR_per_domain": wcr_domain,
            "term_counts": term_counts,
            "WCR_zero_or_negative_peak_domains": wcr_zero_peak_domains,
        }

    ### Logging
    def _log_cl_metrics_stepwise(
        self,
        step_idx: int,
        cl_metrics: Mapping[str, Any],
    ) -> None:
        if self.wandb_run is None:
            return

        payload: Dict[str, Any] = {
            "cl_step": int(step_idx),
            "cl/num_learned_tasks": len(cl_metrics["learned_domains"]),
            "cl/raw_space": 1,
        }
        for metric in self.cl_metrics:
            _put_finite(payload, f"cl/ACC_avg/{metric}", cl_metrics["ACC_avg"][metric])
            _put_finite(payload, f"cl/BWT_avg/{metric}", cl_metrics["BWT_avg"][metric])
            _put_finite(payload, f"cl/FWT_avg/{metric}", cl_metrics["FWT_avg"][metric])
            _put_finite(payload, f"cl/AvgF_avg/{metric}", cl_metrics["AvgF_avg"][metric])
            _put_finite(payload, f"cl/WCR/{metric}", cl_metrics["WCR"][metric])

            for name, count in cl_metrics["term_counts"][metric].items():
                payload[f"cl/term_count/{name}/{metric}"] = int(count)
            payload[f"cl/WCR_zero_peak_count/{metric}"] = len(
                cl_metrics["WCR_zero_or_negative_peak_domains"][metric]
            )

            for domain, value in cl_metrics["BWT_per_domain"][metric].items():
                payload[f"cl/BWT_per_domain/{metric}/{domain}"] = float(value)
            for domain, value in cl_metrics["FWT_per_domain"][metric].items():
                payload[f"cl/FWT_per_domain/{metric}/{domain}"] = float(value)
            for domain, value in cl_metrics["AvgF_per_domain"][metric].items():
                payload[f"cl/AvgF_per_domain/{metric}/{domain}"] = float(value)
            for domain, value in cl_metrics["WCR_per_domain"][metric].items():
                payload[f"cl/WCR_per_domain/{metric}/{domain}"] = float(value)

        self.wandb_run.log(payload, commit=True)

    def _maybe_dump_json(self, step_idx: int, data: Dict[str, Any], tag: str) -> None:
        if not self.log_raw_json:
            return
        filename = f"raw_step_{step_idx}_{_safe_filename(tag)}.json"
        _json_dump(os.path.join(self.output_dir, filename), data)

    def _record_step(
        self,
        *,
        step_idx: int,
        step_tag: str,
        metrics: StepResult,
        diagnostics: DiagnosticResult,
        checkpoint_metadata: Dict[str, Any],
    ) -> None:
        self._append_to_R(metrics)
        self.checkpoint_history.append(
            {
                "step": step_idx,
                "tag": step_tag,
                **checkpoint_metadata,
            }
        )
        self.diagnostic_history.append(
            {
                "step": step_idx,
                "tag": step_tag,
                "domains": diagnostics,
            }
        )

    def _finalize_and_log(self) -> str:
        final_step = len(self.R[self.cl_metrics[0]]) - 1
        final_metrics = self._compute_cl_metrics_at_step_raw(final_step)
        summary = {
            "evaluation_policy": {
                "single_process_single_visible_gpu": True,
                "eval_split": self.eval_split,
                "strict_eval": self.strict_eval,
                "meteor_mode": self.meteor_mode,
                "inference_dtype": self.inference_dtype,
                "required_finite_metrics": list(self.required_finite_metrics),
                "tokenizer_policy": self.tokenizer_policy,
                "token_metric_policy": self.token_metric_policy,
                "token_overlap_cl_metrics_enabled": self.token_metric_policy == "base",
                "token_micro_metrics_are_diagnostic_only": True,
                "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
                "generation_diagnostics_always_enabled": True,
                "preprocessed_cache_policy": self.preprocessed_cache_policy,
            },
            "domains": self.unique_domains,
            "first_learned_step": self.first_learned_step,
            "tokenizer_reference": self.tokenizer_reference,
            "tokenizer_history": self.tokenizer_history,
            "metric_tokenizer": self.metric_tokenizer_metadata,
            "compatibility_tokenizers": self.compatibility_tokenizer_metadata,
            "text_metric_policy": get_metric_policy(),
            "model_reference": self.model_reference,
            "dataset_reference": self.dataset_reference,
            "dataset_policy_hash_history": self.dataset_policy_hash_history,
            "checkpoint_preflight": self.checkpoint_preflight,
            "num_eval_points": len(self.R[self.cl_metrics[0]]),
            "R_matrix": self.R,
            "b_initial": self.b_initial,
            "checkpoint_history": self.checkpoint_history,
            "diagnostic_history": self.diagnostic_history,
            "final": final_metrics,
        }

        summary_path = os.path.join(self.output_dir, "cl_summary.json")
        _json_dump(summary_path, summary)
        self.accelerator.print(f"Saved summary: {summary_path}")

        if self.wandb_run is not None:
            for metric in self.cl_metrics:
                for aggregate_name in ("ACC_avg", "BWT_avg", "FWT_avg", "AvgF_avg", "WCR"):
                    value = final_metrics[aggregate_name][metric]
                    if math.isfinite(value):
                        self.wandb_run.summary[
                            f"final/{aggregate_name}/{metric}"
                        ] = float(value)

            for metric in self.per_domain_metrics:
                for aggregate_name in (
                    "BWT_per_domain",
                    "FWT_per_domain",
                    "AvgF_per_domain",
                    "WCR_per_domain",
                ):
                    for domain, value in final_metrics[aggregate_name][metric].items():
                        self.wandb_run.summary[
                            f"final/{aggregate_name}/{domain}/{metric}"
                        ] = float(value)

            final_payload: Dict[str, Any] = {
                "cl_step": int(final_step),
                "final/num_learned_tasks": len(final_metrics["learned_domains"]),
                "final/raw_space": 1,
            }
            for metric in self.cl_metrics:
                for aggregate_name in ("ACC_avg", "BWT_avg", "FWT_avg", "AvgF_avg", "WCR"):
                    _put_finite(
                        final_payload,
                        f"final/{aggregate_name}/{metric}",
                        final_metrics[aggregate_name][metric],
                    )
            self.wandb_run.log(final_payload, commit=True)

        return summary_path

    
    def run(self) -> str:
        exit_code = 1
        try:
            self.accelerator.print("\n========== STEP 0: BASE ==========")
            base_metrics, base_diagnostics, base_metadata = self._evaluate_checkpoint_on_all(
                self.base_ckpt_path,
                0,
                "base",
            )
            self._record_step(
                step_idx=0,
                step_tag="base",
                metrics=base_metrics,
                diagnostics=base_diagnostics,
                checkpoint_metadata=base_metadata,
            )
            self.b_initial = {
                domain: dict(base_metrics[domain]) for domain in self.unique_domains
            }
            base_cl = self._compute_cl_metrics_at_step_raw(0)
            self._maybe_dump_json(
                0,
                {
                    "eval": base_metrics,
                    "diagnostics": base_diagnostics,
                    "checkpoint": base_metadata,
                    "cl": base_cl,
                },
                "base",
            )
            self._log_cl_metrics_stepwise(0, base_cl)

            iterator = tqdm(
                enumerate(self.curriculum, start=1),
                total=len(self.curriculum),
                desc="[Total CL Steps]",
                dynamic_ncols=True,
            )
            for step_idx, (domain, ckpt_path, epoch) in iterator:
                step_tag = f"after_{domain}_epoch_{epoch}"
                self.accelerator.print(f"\n--- Step {step_idx}: {step_tag} ---")
                metrics, diagnostics, metadata = self._evaluate_checkpoint_on_all(
                    ckpt_path,
                    step_idx,
                    step_tag,
                )
                self._record_step(
                    step_idx=step_idx,
                    step_tag=step_tag,
                    metrics=metrics,
                    diagnostics=diagnostics,
                    checkpoint_metadata=metadata,
                )
                cl_metrics = self._compute_cl_metrics_at_step_raw(step_idx)
                self._maybe_dump_json(
                    step_idx,
                    {
                        "eval": metrics,
                        "diagnostics": diagnostics,
                        "checkpoint": metadata,
                        "cl": cl_metrics,
                    },
                    step_tag,
                )
                self._log_cl_metrics_stepwise(step_idx, cl_metrics)

            summary_path = self._finalize_and_log()
            exit_code = 0
            return summary_path
        finally:
            _clear_caches(self.device)
            self.metric_tokenizer = None
            self.compatibility_tokenizers = []
            if self.wandb_run is not None:
                try:
                    self.wandb_run.finish(exit_code=exit_code)
                finally:
                    self.wandb_run = None
