# eval/modules/eval_stability.py
"""Strict single-process / single-GPU generation evaluation.

Paper-final evaluation contract
-------------------------------
1. ``domain_map`` returns a frozen RAW manifest containing only prompt/reference/id.
2. Gold references never enter model input construction and are never tokenized by
   the checkpoint tokenizer for generation.
3. Each checkpoint tokenizer encodes only the same frozen raw prompt text.
4. The manifest is compatibility-filtered against the predeclared tokenizer set
   (base + every CL checkpoint) before scoring.
5. Metrics compare decoded predictions against the frozen raw reference text.
6. Token-overlap P/R/F1, when enabled, use one fixed metric tokenizer.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.metrics import calculate_metrics

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from utils.n_domain_map import (  # noqa: E402
    EVAL_DUPLICATE_POLICY,
    EVAL_MANIFEST_TYPE,
    EVAL_SCHEMA_VERSION,
    domain_info,
    get_collate_fn,
    get_eval_policy,
    get_processed_dataset,
)

MetricDict = Dict[str, Any]
ModelAndTokenizerLoader = Callable[[torch.device], Tuple[Any, Any]]


def _nan_metrics() -> MetricDict:
    return {
        "rouge1": float("nan"),
        "rouge2": float("nan"),
        "rougeL": float("nan"),
        "bleu": float("nan"),
        "chrf": float("nan"),
        "meteor": float("nan"),
        "cosine_similarity": float("nan"),
        "token_precision": float("nan"),
        "token_recall": float("nan"),
        "token_f1": float("nan"),
        "token_precision_micro": float("nan"),
        "token_recall_micro": float("nan"),
        "token_f1_micro": float("nan"),
        "eval_num_examples": 0.0,
        "eval_content_sha256": None,
        "eval_duplicate_policy": EVAL_DUPLICATE_POLICY,
        "eval_prompt_max_tokens": float("nan"),
        "eval_generation_cap_hit_rate": float("nan"),
        "eval_prompt_roundtrip_mismatch_rate": float("nan"),
    }


def _strict() -> bool:
    return os.environ.get("EVAL_STRICT", "0") == "1"


def _canonical_eval_text(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def _roundtrip_compare_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _update_eval_content_hash(hasher: "hashlib._Hash", prompt: str, reference: str) -> None:
    for tag, value in ((b"P", prompt), (b"R", reference)):
        payload = _canonical_eval_text(value).encode("utf-8", errors="surrogatepass")
        hasher.update(tag)
        hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        hasher.update(payload)


def _unwrap_for_generate(model: Any, accelerator: Accelerator) -> Any:
    candidates = [model]
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        try:
            unwrapped = unwrap_model(model)
            if unwrapped is not model:
                candidates.append(unwrapped)
        except Exception:
            pass
    module = getattr(model, "module", None)
    if module is not None and module is not model:
        candidates.append(module)

    seen = set()
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if hasattr(candidate, "generate"):
            return candidate
    raise AttributeError(
        f"Model of type {type(model)} has no safe generate-capable view."
    )


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA, got {device}.")
    return int(device.index if device.index is not None else torch.cuda.current_device())


def _parse_cuda_device(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, torch.device):
        return _device_index(value) if value.type == "cuda" else None
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "cuda":
            return int(torch.cuda.current_device())
        if value.isdigit():
            return int(value)
        if value.startswith("cuda:") and value.split(":", 1)[1].isdigit():
            return int(value.split(":", 1)[1])
    return None


def _collect_model_cuda_devices(model: Any) -> Tuple[set[int], set[str]]:
    cuda_devices: set[int] = set()
    non_cuda: set[str] = set()
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for placement in device_map.values():
            idx = _parse_cuda_device(placement)
            if idx is None:
                non_cuda.add(str(placement))
            else:
                cuda_devices.add(idx)
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type == "cuda":
            cuda_devices.add(_device_index(tensor.device))
        else:
            non_cuda.add(str(tensor.device))
    return cuda_devices, non_cuda


def _distributed_wrapper_name(model: Any) -> Optional[str]:
    if isinstance(model, torch.nn.DataParallel):
        return "torch.nn.DataParallel"
    try:
        from torch.nn.parallel import DistributedDataParallel
        if isinstance(model, DistributedDataParallel):
            return "torch.nn.parallel.DistributedDataParallel"
    except Exception:
        pass
    name = f"{type(model).__module__}.{type(model).__name__}"
    if any(x in name for x in ("FullyShardedDataParallel", "FSDP", "DeepSpeedEngine")):
        return name
    return None


def _cuda_memory_mib(device: torch.device) -> Tuple[float, float]:
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0.0, 0.0
    return (
        float(torch.cuda.memory_allocated(device) / (1024.0**2)),
        float(torch.cuda.memory_reserved(device) / (1024.0**2)),
    )


def _clear_python_and_cuda_caches(device: Optional[torch.device]) -> None:
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


class EvalRunner:
    """Evaluate one causal LM against frozen raw prompt/reference manifests."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        accelerator: Accelerator,
        batch_size: int = 8,
        max_length: int = 2048,
        gen_max_new_tokens: int = 256,
        eval_split: Optional[str] = None,
        preprocessed_data_root: Optional[str] = None,
        wandb_run: Any = None,
        cl_step_idx: Optional[int] = None,
        step_tag: str = "",
        shot_type: str = "zero-shot",
        seed: int = 777,
        clear_cache_between_domains: bool = False,
        release_tokenizer_on_cleanup: bool = True,
        meteor_mode: str = "multilingual_lexical",
        metric_tokenizer: Optional[Any] = None,
        release_metric_tokenizer_on_cleanup: bool = False,
        compute_token_metrics: bool = True,
        compatibility_tokenizers: Optional[Sequence[Any]] = None,
        release_compatibility_tokenizers_on_cleanup: bool = False,
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not isinstance(max_length, int) or max_length <= 1:
            raise ValueError("max_length must be > 1.")
        if not isinstance(gen_max_new_tokens, int) or gen_max_new_tokens <= 0:
            raise ValueError("gen_max_new_tokens must be positive.")
        if eval_split is None or not str(eval_split).strip():
            raise ValueError("eval_split must be explicit and non-empty.")
        if model is None or tokenizer is None:
            raise ValueError("model/tokenizer must not be None.")

        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self.eval_split = str(eval_split).strip()
        self.preprocessed_data_root = preprocessed_data_root
        self.wandb_run = wandb_run
        self.cl_step_idx = cl_step_idx
        self.step_tag = step_tag
        self.shot_type = shot_type
        self.seed = int(seed)
        self.clear_cache_between_domains = bool(clear_cache_between_domains)
        self.release_tokenizer_on_cleanup = bool(release_tokenizer_on_cleanup)
        self.meteor_mode = str(meteor_mode).strip().lower()
        if self.meteor_mode not in {"multilingual_lexical", "standard"}:
            raise ValueError("meteor_mode must be multilingual_lexical or standard.")
        self.compute_token_metrics = bool(compute_token_metrics)
        self.metric_tokenizer = metric_tokenizer if metric_tokenizer is not None else tokenizer
        self.release_metric_tokenizer_on_cleanup = bool(release_metric_tokenizer_on_cleanup)
        self.compatibility_tokenizers = list(compatibility_tokenizers or [tokenizer])
        if not self.compatibility_tokenizers:
            raise ValueError("compatibility_tokenizers cannot be empty.")
        self.release_compatibility_tokenizers_on_cleanup = bool(
            release_compatibility_tokenizers_on_cleanup
        )
        self._closed = False

        self.eval_device = self._validate_single_process_single_gpu()
        torch.cuda.set_device(self.eval_device)

        config = getattr(self.model, "config", None)
        if config is not None and bool(getattr(config, "is_encoder_decoder", False)):
            raise ValueError("This evaluator supports decoder-only causal LMs only.")

        self.tokenizer.padding_side = "left"
        self.pad_id = self._configure_padding_token()
        if config is not None:
            config.pad_token_id = self.pad_id
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.pad_token_id = self.pad_id
            generation_config.do_sample = False

        self.collate_fn = get_collate_fn(collate_type="smart", pad_id=self.pad_id)
        print(
            "[EvalRunner] frozen-raw-manifest; "
            f"device={self.eval_device}, split={self.eval_split}, max_length={self.max_length}, "
            f"gen_max_new_tokens={self.gen_max_new_tokens}, tokenizers_for_preflight="
            f"{len(self.compatibility_tokenizers)}"
        )

    def _validate_single_process_single_gpu(self) -> torch.device:
        if self.accelerator.num_processes != 1:
            raise RuntimeError("Evaluation requires exactly one process.")
        wrapper = _distributed_wrapper_name(self.model)
        if wrapper is not None:
            raise RuntimeError(f"Distributed/multi-GPU wrapper detected: {wrapper}.")
        if not torch.cuda.is_available() or self.accelerator.device.type != "cuda":
            raise RuntimeError("Evaluation requires CUDA.")
        cuda_devices, non_cuda = _collect_model_cuda_devices(self.model)
        if non_cuda:
            raise RuntimeError(f"Model has non-CUDA placements: {sorted(non_cuda)}")
        if len(cuda_devices) != 1:
            raise RuntimeError(f"Model must occupy exactly one CUDA device: {sorted(cuda_devices)}")
        model_idx = next(iter(cuda_devices))
        accel_idx = _device_index(self.accelerator.device)
        if model_idx != accel_idx:
            raise RuntimeError(f"Model cuda:{model_idx} != Accelerator cuda:{accel_idx}.")
        return torch.device(f"cuda:{model_idx}")

    def _configure_padding_token(self) -> int:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if isinstance(eos_id, (list, tuple)):
                eos_id = eos_id[0] if eos_id else None
            if eos_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token
            else:
                self.tokenizer.pad_token_id = int(eos_id)
            pad_id = self.tokenizer.pad_token_id
        return int(pad_id)

    def _ensure_open(self) -> None:
        if self._closed or self.model is None:
            raise RuntimeError("EvalRunner has already been cleaned up.")

    def _domain_eval_policy(self, domain: str) -> Dict[str, Any]:
        if domain not in domain_info:
            raise KeyError(f"Unknown domain {domain!r}.")
        return dict(get_eval_policy(domain, self.eval_split, self.max_length))

    def _load_dataset_for_domain(self, domain: str):
        policy = self._domain_eval_policy(domain)
        print(
            f"[EvalRunner:{domain}] frozen_manifest split="
            f"{policy['requested_split']}->{policy['effective_split']}, "
            f"prompt_budget={policy['effective_max_length']}, "
            f"truncate={policy['truncate_strategy']}, task={policy['task_type']}"
        )
        return get_processed_dataset(
            domain=domain,
            tokenizer=self.tokenizer,  # evaluation mode does not use it for manifest construction
            max_length=self.max_length,
            mode="evaluation",
            # Pass the user's requested split, not the already-resolved split.
            # get_processed_dataset() must hash the same requested->effective policy
            # that EvalRunner reports in its metadata.
            split=self.eval_split,
            shot_type=self.shot_type,
            preprocessed_root=self.preprocessed_data_root,
            seed=self.seed,
            compatibility_tokenizers=self.compatibility_tokenizers,
        )

    def _encode_tokens(self, text: Optional[str]) -> List[int]:
        if text is None or not str(text).strip():
            return []
        return self.metric_tokenizer.encode(str(text).strip(), add_special_tokens=False)

    @staticmethod
    def _overlap_count(pred_toks: List[int], gold_toks: List[int]) -> Tuple[int, int, int]:
        common = Counter(pred_toks) & Counter(gold_toks)
        return sum(common.values()), len(pred_toks), len(gold_toks)

    @staticmethod
    def _prf(num_same: int, pred_len: int, gold_len: int) -> Tuple[float, float, float]:
        if pred_len == 0 or gold_len == 0 or num_same == 0:
            return 0.0, 0.0, 0.0
        p = num_same / pred_len
        r = num_same / gold_len
        f1 = 2.0 * p * r / (p + r) if p + r > 0 else 0.0
        return float(p), float(r), float(f1)

    def _compute_token_metrics(self, predictions: Sequence[str], labels: Sequence[str]) -> MetricDict:
        if not predictions or len(predictions) != len(labels):
            return {k: float("nan") for k in (
                "token_precision", "token_recall", "token_f1",
                "token_precision_micro", "token_recall_micro", "token_f1_micro",
            )}
        ps, rs, fs = [], [], []
        total_same = total_pred = total_gold = 0
        for pred, gold in zip(predictions, labels):
            same, plen, glen = self._overlap_count(self._encode_tokens(pred), self._encode_tokens(gold))
            p, r, f = self._prf(same, plen, glen)
            ps.append(p); rs.append(r); fs.append(f)
            total_same += same; total_pred += plen; total_gold += glen
        mp, mr, mf = self._prf(total_same, total_pred, total_gold)
        return {
            "token_precision": float(sum(ps) / len(ps)),
            "token_recall": float(sum(rs) / len(rs)),
            "token_f1": float(sum(fs) / len(fs)),
            "token_precision_micro": mp,
            "token_recall_micro": mr,
            "token_f1_micro": mf,
        }

    @staticmethod
    def _validate_raw_batch(batch: Mapping[str, Any], domain: str) -> int:
        required = (
            "eval_prompt_text", "eval_reference_text", "eval_example_id",
            "eval_schema_version", "eval_policy_hash",
        )
        missing = [k for k in required if k not in batch]
        if missing:
            raise RuntimeError(f"[{domain}] frozen manifest batch missing {missing}.")
        n = len(batch["eval_prompt_text"])
        if n == 0:
            raise RuntimeError(f"[{domain}] empty raw manifest batch.")
        for key in required:
            if len(batch[key]) != n:
                raise RuntimeError(f"[{domain}] raw metadata length mismatch for {key}.")
        if any(str(v) != EVAL_SCHEMA_VERSION for v in batch["eval_schema_version"]):
            raise RuntimeError(f"[{domain}] stale/mixed schema version in batch.")
        if len(set(str(v) for v in batch["eval_policy_hash"])) != 1:
            raise RuntimeError(f"[{domain}] mixed manifest policy hashes in one batch.")
        if any(not str(x).strip() for x in batch["eval_prompt_text"]):
            raise RuntimeError(f"[{domain}] empty frozen prompt.")
        if any(not str(x).strip() for x in batch["eval_reference_text"]):
            raise RuntimeError(f"[{domain}] empty frozen reference.")
        return n

    def evaluate(self, domain: str) -> MetricDict:
        self._ensure_open()
        domain = str(domain).strip()
        if not domain:
            raise ValueError("domain must be non-empty.")
        policy = self._domain_eval_policy(domain)

        dataset = dataloader = gen_model = None
        try:
            dataset = self._load_dataset_for_domain(domain)
            dataset_size = len(dataset)
            if dataset_size == 0:
                message = f"[{domain}] frozen evaluation manifest is empty."
                if _strict():
                    raise RuntimeError(message)
                return _nan_metrics()

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=0,
                pin_memory=False,
            )
            self.model.eval()
            gen_model = _unwrap_for_generate(self.model, self.accelerator)

            predictions: List[str] = []
            references: List[str] = []
            processed = 0
            generation_cap_hits = 0
            prompt_roundtrip_mismatches = 0
            prompt_max_tokens = 0
            content_hasher = hashlib.sha256()
            seen_policy_hash: Optional[str] = None
            seen_example_ids: set[str] = set()

            with torch.inference_mode():
                iterator = tqdm(dataloader, desc=f"[Eval:{domain}]", leave=False, dynamic_ncols=True)
                for batch in iterator:
                    batch_n = self._validate_raw_batch(batch, domain)
                    prompts = [str(x).strip() for x in batch["eval_prompt_text"]]
                    refs = [str(x).strip() for x in batch["eval_reference_text"]]
                    example_ids = [str(x).strip() for x in batch["eval_example_id"]]
                    if len(set(example_ids)) != len(example_ids):
                        raise RuntimeError(
                            f"[{domain}] duplicate eval_example_id values within one batch."
                        )
                    repeated_ids = seen_example_ids.intersection(example_ids)
                    if repeated_ids:
                        sample = sorted(repeated_ids)[:8]
                        raise RuntimeError(
                            f"[{domain}] duplicate eval_example_id values across manifest batches: "
                            f"{sample}. The frozen manifest must obey "
                            f"duplicate_policy={EVAL_DUPLICATE_POLICY!r}."
                        )
                    seen_example_ids.update(example_ids)
                    policy_hash = str(batch["eval_policy_hash"][0])
                    if seen_policy_hash is None:
                        seen_policy_hash = policy_hash
                    elif policy_hash != seen_policy_hash:
                        raise RuntimeError(f"[{domain}] manifest policy changed within one evaluation.")

                    # Critical contract: ONLY frozen prompts are encoded by the checkpoint tokenizer.
                    encoded = self.tokenizer(
                        prompts,
                        add_special_tokens=True,
                        truncation=False,
                        padding=True,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"]
                    attention_mask = encoded["attention_mask"]
                    lengths = attention_mask.sum(dim=1).tolist()
                    if lengths:
                        prompt_max_tokens = max(prompt_max_tokens, max(int(x) for x in lengths))
                    over = [i for i, length in enumerate(lengths) if int(length) > int(policy["effective_max_length"])]
                    if over:
                        raise RuntimeError(
                            f"[PROMPT-PREFLIGHT-BUG][{domain}] frozen prompt exceeds declared "
                            f"budget with current tokenizer; local_indices={over[:8]}. "
                            "The manifest should have been filtered against all checkpoint tokenizers."
                        )

                    # Prompt-only tokenizer round-trip diagnostic. Gold references are untouched.
                    decoded_prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    for raw, decoded in zip(prompts, decoded_prompts):
                        if _roundtrip_compare_text(raw) != _roundtrip_compare_text(decoded):
                            prompt_roundtrip_mismatches += 1

                    input_ids = input_ids.to(self.eval_device, non_blocking=True)
                    attention_mask = attention_mask.to(self.eval_device, non_blocking=True)
                    gen_input_len = int(input_ids.size(1))

                    kwargs: Dict[str, Any] = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": self.gen_max_new_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                        "num_return_sequences": 1,
                        "pad_token_id": self.pad_id,
                        "return_dict_in_generate": False,
                    }
                    eos_id = getattr(self.tokenizer, "eos_token_id", None)
                    if eos_id is not None:
                        kwargs["eos_token_id"] = eos_id

                    generated_full = gen_model.generate(**kwargs)
                    if not torch.is_tensor(generated_full) or generated_full.ndim != 2:
                        raise RuntimeError(f"[{domain}] generate() must return rank-2 tensor.")
                    if generated_full.size(0) != batch_n:
                        raise RuntimeError(f"[{domain}] generation batch-size mismatch.")
                    if generated_full.size(1) < gen_input_len:
                        raise RuntimeError(f"[{domain}] generated sequence shorter than prompt width.")
                    if not torch.equal(generated_full[:, :gen_input_len], input_ids):
                        raise RuntimeError(f"[{domain}] generate() did not preserve prompt prefix.")

                    generated_only = generated_full[:, gen_input_len:].detach().cpu()
                    eos_values = set()
                    if isinstance(eos_id, (list, tuple, set)):
                        eos_values = {int(x) for x in eos_id}
                    elif eos_id is not None:
                        eos_values = {int(eos_id)}

                    for pred_ids, prompt, ref in zip(generated_only, prompts, refs):
                        if int(pred_ids.numel()) >= self.gen_max_new_tokens:
                            ids_list = [int(x) for x in pred_ids.tolist()]
                            if not eos_values or not any(x in eos_values for x in ids_list):
                                generation_cap_hits += 1
                        pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
                        predictions.append(pred_text)
                        references.append(ref)
                        _update_eval_content_hash(content_hasher, prompt, ref)

                    processed += batch_n
                    del batch, encoded, input_ids, attention_mask, generated_full, generated_only

            if processed != dataset_size or len(predictions) != dataset_size:
                raise RuntimeError(
                    f"[{domain}] manifest count mismatch: dataset={dataset_size}, processed={processed}, "
                    f"predictions={len(predictions)}."
                )

            metrics: MetricDict = dict(
                calculate_metrics(predictions, references, meteor_mode=self.meteor_mode)
            )
            if self.compute_token_metrics:
                metrics.update(self._compute_token_metrics(predictions, references))

            metrics.update({
                "eval_num_examples": float(dataset_size),
                "eval_content_sha256": content_hasher.hexdigest(),
                "eval_policy_hash": seen_policy_hash,
                "eval_requested_split": policy["requested_split"],
                "eval_effective_split": policy["effective_split"],
                "eval_requested_max_length": float(policy["requested_max_length"]),
                "eval_effective_max_length": float(policy["effective_max_length"]),
                "eval_truncate_source": bool(policy["truncate_source"]),
                "eval_truncate_strategy": policy["truncate_strategy"],
                "eval_max_eval_samples": policy["max_eval_samples"],
                "eval_schema_version": policy["schema_version"],
                "eval_manifest_type": policy["manifest_type"],
                "eval_duplicate_policy": policy["duplicate_policy"],
                "eval_prompt_compatibility": policy["prompt_compatibility"],
                "eval_reference_used_for_prompt_construction": bool(
                    policy["reference_used_for_prompt_construction"]
                ),
                "eval_language": policy["language"],
                "eval_family": policy["eval_family"],
                "eval_task_type": policy["task_type"],
                "eval_prompt_max_tokens": float(prompt_max_tokens),
                "eval_generation_cap_hit_rate": float(generation_cap_hits / dataset_size),
                "eval_prompt_roundtrip_mismatch_rate": float(
                    prompt_roundtrip_mismatches / dataset_size
                ),
            })
            return metrics
        finally:
            gen_model = dataloader = dataset = None
            gc.collect()
            if self.clear_cache_between_domains and not self._closed:
                try:
                    torch.cuda.synchronize(self.eval_device)
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def evaluate_domains(self, domains: Iterable[str], *, cleanup_after: bool = True) -> Dict[str, MetricDict]:
        self._ensure_open()
        domain_list = [domains.strip()] if isinstance(domains, str) else [str(x).strip() for x in domains]
        if not domain_list or any(not x for x in domain_list):
            raise ValueError("domains must contain non-empty names.")
        if len(set(domain_list)) != len(domain_list):
            raise ValueError(f"Duplicate domains: {domain_list}")
        results: Dict[str, MetricDict] = {}
        try:
            for domain in domain_list:
                results[domain] = self.evaluate(domain)
            return results
        finally:
            if cleanup_after:
                self.cleanup()

    def evaluate_and_cleanup(self, domain: str) -> MetricDict:
        return self.evaluate_domains([domain], cleanup_after=True)[domain]

    def cleanup(self) -> None:
        if self._closed:
            return
        device = self.eval_device
        before_alloc, before_reserved = _cuda_memory_mib(device)
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

        model_ref = self.model
        tokenizer_ref = self.tokenizer if self.release_tokenizer_on_cleanup else None
        metric_ref = (
            self.metric_tokenizer
            if self.release_metric_tokenizer_on_cleanup and self.metric_tokenizer is not self.tokenizer
            else None
        )
        compatibility_refs = (
            self.compatibility_tokenizers
            if self.release_compatibility_tokenizers_on_cleanup
            else None
        )

        self.model = None
        if self.release_tokenizer_on_cleanup:
            self.tokenizer = None
        self.metric_tokenizer = None
        self.compatibility_tokenizers = []
        self.collate_fn = None
        self._closed = True

        del model_ref
        if tokenizer_ref is not None:
            del tokenizer_ref
        if metric_ref is not None:
            del metric_ref
        if compatibility_refs is not None:
            del compatibility_refs

        accelerator_ref = self.accelerator
        free_memory = getattr(accelerator_ref, "free_memory", None)
        clear = getattr(accelerator_ref, "clear", None)
        if callable(free_memory):
            try:
                free_memory()
            except Exception:
                pass
        elif callable(clear):
            try:
                clear()
            except Exception:
                pass
        self.accelerator = None
        del accelerator_ref
        _clear_python_and_cuda_caches(device)
        after_alloc, after_reserved = _cuda_memory_mib(device)
        print(
            "[EvalRunner.cleanup] "
            f"allocated={before_alloc:.1f}->{after_alloc:.1f} MiB, "
            f"reserved={before_reserved:.1f}->{after_reserved:.1f} MiB"
        )

    close = cleanup

    def __enter__(self) -> "EvalRunner":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.cleanup()
        return False


def run_single_gpu_evaluation(
    load_model_and_tokenizer: ModelAndTokenizerLoader,
    domains: Iterable[str],
    *,
    runner_kwargs: Mapping[str, Any],
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, MetricDict]:
    if not callable(load_model_and_tokenizer):
        raise TypeError("load_model_and_tokenizer must be callable.")
    forbidden = {"model", "tokenizer", "accelerator"}
    overlap = forbidden.intersection(runner_kwargs)
    if overlap:
        raise ValueError(f"runner_kwargs contains managed keys: {sorted(overlap)}")
    if "eval_split" not in runner_kwargs or not str(runner_kwargs["eval_split"]).strip():
        raise ValueError("runner_kwargs must include eval_split.")

    accelerator = accelerator or Accelerator()
    if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
        raise RuntimeError("run_single_gpu_evaluation requires one CUDA process.")
    domain_list = [domains.strip()] if isinstance(domains, str) else [str(x).strip() for x in domains]

    model = tokenizer = bundle = None
    runner: Optional[EvalRunner] = None
    try:
        bundle = load_model_and_tokenizer(accelerator.device)
        if not isinstance(bundle, tuple) or len(bundle) != 2:
            raise TypeError("loader must return (model, tokenizer).")
        model, tokenizer = bundle
        bundle = None
        runner = EvalRunner(model=model, tokenizer=tokenizer, accelerator=accelerator, **dict(runner_kwargs))
        model = tokenizer = None
        return runner.evaluate_domains(domain_list, cleanup_after=True)
    finally:
        if runner is not None:
            runner.cleanup()
        runner = None
        model = tokenizer = bundle = None
        _clear_python_and_cuda_caches(
            accelerator.device if accelerator.device.type == "cuda" else None
        )
