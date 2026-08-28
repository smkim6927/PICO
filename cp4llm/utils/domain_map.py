#cp4llm/utils/domain_map.py

import os
import hashlib
import json
from typing import Tuple, Union, Optional
import re
from functools import partial
from itertools import chain

from datasets import (
    load_dataset,
    Dataset,
    load_from_disk,
    get_dataset_split_names,
)
import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def _num_shots_from_type(shot_type: str) -> int:
    if not shot_type:
        return 0
    st = str(shot_type).strip().lower().replace(" ", "")
    if st in {"zero-shot", "zeroshot", "0-shot", "0shot", "0"}:
        return 0
    if st in {"3-shot", "3shot", "three-shot", "threeshot", "3"}:
        return 3
    m = re.match(r"^(\d+)[-]?shot$", st)
    if m:
        return int(m.group(1))
    return 0

def _parse_medical_reasoning(example):
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text.strip().startswith("Q:"):
        context = input_text.split("Q:", 1)[1].strip()
    else:
        context = input_text.strip()

    question = f"Q: {context}"

    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    
    if answer_match:
        raw_answer = answer_match.group(1).strip()
        
        answer = raw_answer.split(':')[0].strip()
        
        answer = answer.replace(".", "")
    else:
        answer = output_text.strip().split(':')[0].strip()

    return {"text": question, "target": answer}

def _parse_gsm8k(example):
    question = example.get("question", "")
    answer_full = example.get("answer", "")

    m = re.search(r"####\s*([^\n]+)", answer_full)
    if m:
        final_ans = m.group(1).strip()
    else:
        final_ans = answer_full.strip()

    return {"text": question.strip(), "target": final_ans}


# ★ ES: parser for qanastek/ELRC-Medical-V2 (config "en-es").
# Standard HF translation format: {"translation": {"en": ..., "es": ...}}.
# Flat en/es columns are accepted as a fallback. Anything else fails LOUDLY
# with the actual row keys, so a schema drift never becomes a silent NaN.
def _parse_elrc_medical_en_es(example):
    tr = example.get("translation")
    if isinstance(tr, dict) and "en" in tr and "es" in tr:
        return {"text": str(tr["en"]).strip(), "target": str(tr["es"]).strip()}
    if "en" in example and "es" in example:
        return {"text": str(example["en"]).strip(),
                "target": str(example["es"]).strip()}
    raise KeyError(
        f"[es_medical] unexpected ELRC-Medical-V2 row schema. "
        f"Row keys: {list(example.keys())}. Expected translation.en/es."
    )


# ★ ES: parser for SINAI/ALIA-es-legal-administrative-cqa.
# CQA rows: question/answer with optional context. Field names are matched
# from candidates (Spanish and English variants, SQuAD-style answers dict).
# Context, when present, is prepended to the question text; create_prompt
# then wraps it as Pregunta/Respuesta.
def _parse_alia_legal_cqa(example):
    q = None
    for k in ("question", "pregunta", "query", "instruction"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            break
    a = None
    for k in ("answer", "respuesta", "answers", "output", "response"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            a = v.strip()
            break
        if isinstance(v, list) and v and isinstance(v[0], str):
            a = v[0].strip()
            break
        if isinstance(v, dict):
            t = v.get("text")
            if isinstance(t, list) and t:
                a = str(t[0]).strip()
                break
            if isinstance(t, str) and t.strip():
                a = t.strip()
                break
    ctx = None
    for k in ("context", "contexto", "passage"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            ctx = v.strip()
            break
    if q is None or a is None:
        raise KeyError(
            f"[es_legal] unexpected ALIA-cqa row schema. "
            f"Row keys: {list(example.keys())}. "
            f"Expected question/answer (or pregunta/respuesta), got neither."
        )
    text = f"{ctx}\n\n{q}" if ctx else q
    return {"text": text, "target": a}


domain_info = {
    "kor_medical": {
        "pretrain_data_path": "utils/data_storage/new-medical-kor-dataset.txt",
        "dataset_name": "ChuGyouk/medical-o1-reasoning-SFT-Ko",
        "encoding": "utf-8",
        "columns": {"text": "Question", "target": "Response"},
        "task_type": "qa",
        "parser_fn": None,
    },
    "eng_medical": {
        "pretrain_data_path": "utils/data_storage/guidline_medical.txt",
        "dataset_name": "Shekswess/medical_gemma_instruct_dataset",
        "encoding": "utf-8",
        # 이 데이터셋은 features: ['output', 'input', 'instruction']
        # columns 를 그대로 쓰는 대신 parser_fn에서 text/target을 만들어냄.
        "columns": {"text": "input", "target": "output"},
        "task_type": "qa",
        "parser_fn": None,
    },
    "kor_legal": {
        "pretrain_data_path": "utils/data_storage/new-legal-kor-dataset.txt",
        "dataset_name": "jihye-moon/LawQA-Ko",
        "encoding": "utf-8",
        "columns": {"text": "question", "target": "answer"},
        "task_type": "qa",
        "parser_fn": None,
    },

    "eng_legal": {
        "pretrain_data_path": "utils/data_storage/eng-new-legal-dataset.txt",
        "dataset_name": "joelniklaus/legal_case_document_summarization",
        "encoding": "utf-8",
        "columns": {"text": "judgement", "target": "summary"},
        "task_type": "summarization",
        "parser_fn": None,
        "truncate_source": True,
    },
    
    "math": {
        "pretrain_data_path": None,
        "dataset_name": "openai/gsm8k",
        "hf_config": "main",
        "encoding": "utf-8",
        "columns": {"text": "question", "target": "answer"},
        "task_type": "math",
        "parser_fn": _parse_gsm8k,
    },

    "es_medical": {
        "pretrain_data_path": "utils/data_storage/es-medical-dataset.txt",
        "dataset_name": "qanastek/ELRC-Medical-V2",
        "hf_config": "en-es",
        "hf_split": "train",
        "hf_trust_remote_code": True,
        "encoding": "utf-8",
        "columns": {"text": "translation", "target": "translation"},
        "task_type": "translation",
        "parser_fn": _parse_elrc_medical_en_es,
        "truncate_source": True,
    },

    "es_legal": {
        "pretrain_data_path": "utils/data_storage/es-legal-dataset.txt",
        "dataset_name": "SINAI/ALIA-es-legal-administrative-cqa",
        "hf_split": "parlamint_es_an",  # EvalRunner 의 split="train" 을 오버라이드
        "encoding": "utf-8",
        "columns": {"text": "question", "target": "answer"},
        "task_type": "qa",
        "parser_fn": _parse_alia_legal_cqa,
        "truncate_source": True,        # ParlaMint 발화 context 가 길 수 있음
    },
}

EVAL_SCHEMA_VERSION = "cl_eval_v5_frozen_raw_manifest_dedup"
EVAL_MANIFEST_TYPE = "frozen_raw_prompt_reference"
EVAL_DUPLICATE_POLICY = "deduplicate_exact_source_reference_keep_first"


_FROZEN_MANIFEST_CACHE = {}
_DATASET_SPLIT_CACHE = {}


def _infer_language_and_family(domain: str):
    if domain.startswith("kor_"):
        language = "ko"
    elif domain.startswith("es_"):
        language = "es"
    else:
        language = "en"
    family = domain.split("_", 1)[1] if "_" in domain else domain
    return language, family


def _available_dataset_splits(domain: str):
    """Return available HF split names for ``domain`` with process-local caching.

    This inspects dataset metadata only. It does not use model outputs, references,
    or metric values.  An explicit nonstandard ``hf_split`` in ``domain_info`` is
    still authoritative because some datasets expose neither ``test`` nor ``train``
    under conventional names (e.g. a named corpus partition).
    """
    if domain not in domain_info:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(domain_info)}")

    info = domain_info[domain]
    dataset_name = info["dataset_name"]
    hf_config = info.get("hf_config")
    cache_key = (dataset_name, hf_config)
    cached = _DATASET_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    trust_remote = info.get("hf_trust_remote_code")
    kwargs = {} if trust_remote is None else {"trust_remote_code": bool(trust_remote)}

    try:
        if hf_config is not None:
            names = get_dataset_split_names(dataset_name, hf_config, **kwargs)
        else:
            names = get_dataset_split_names(dataset_name, **kwargs)
    except TypeError:
        # Compatibility with datasets releases where trust_remote_code is no
        # longer accepted by split discovery. Dataset loading itself still uses
        # the domain-specific load policy below.
        if hf_config is not None:
            names = get_dataset_split_names(dataset_name, hf_config)
        else:
            names = get_dataset_split_names(dataset_name)

    names = [str(name) for name in names]
    if not names:
        raise RuntimeError(f"[{domain}] Dataset reports no available splits.")
    _DATASET_SPLIT_CACHE[cache_key] = tuple(names)
    return names


def _resolve_eval_split(domain: str, requested_split: str) -> tuple[str, list[str], str]:
    """Resolve the evaluation split deterministically.

    Policy:
      1) If ``domain_info[domain]['hf_split']`` is explicitly set, use it.
         This preserves required nonstandard partitions such as ``parlamint_es_an``.
      2) Otherwise use the requested split when it exists.
      3) For the normal paper setting ``requested_split='test'``, fall back to
         ``train`` when the dataset has no test split.
      4) For any other unavailable requested split, prefer ``test`` and then
         ``train``.  If neither exists, fail loudly rather than silently choosing
         an unrelated partition.
    """
    requested = str(requested_split).strip()
    if not requested:
        raise ValueError("requested_split must be non-empty.")

    info = domain_info[domain]
    explicit = info.get("hf_split")
    available = _available_dataset_splits(domain)

    if explicit is not None:
        explicit = str(explicit).strip()
        if explicit not in available:
            raise RuntimeError(
                f"[{domain}] Explicit hf_split={explicit!r} is unavailable. "
                f"Available splits: {available}."
            )
        return explicit, available, "explicit_hf_split"

    if requested in available:
        return requested, available, "requested_split_available"

    if requested == "test" and "train" in available:
        return "train", available, "test_unavailable_fallback_to_train"

    if "test" in available:
        return "test", available, "requested_unavailable_fallback_to_test"
    if "train" in available:
        return "train", available, "requested_unavailable_fallback_to_train"

    raise RuntimeError(
        f"[{domain}] Requested split {requested!r} is unavailable and dataset "
        f"has neither 'test' nor 'train'. Available splits: {available}. "
        "Add an explicit hf_split in domain_info for this dataset."
    )


def get_eval_policy(domain: str, requested_split: str, requested_max_length: int):
    """Return the immutable evaluation policy for one domain.

    ``effective_max_length`` is a PROMPT-only budget. Gold-reference length is
    deliberately absent from every selection/truncation decision. Split choice
    is resolved against the dataset's actual split metadata before manifest
    construction so the frozen policy records the split that was truly used.
    """
    if domain not in domain_info:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(domain_info)}")
    if not str(requested_split).strip():
        raise ValueError("requested_split must be non-empty.")
    if int(requested_max_length) <= 1:
        raise ValueError("requested_max_length must be > 1.")

    info = domain_info[domain]
    effective_split, available_splits, split_resolution = _resolve_eval_split(
        domain, requested_split
    )
    language, family = _infer_language_and_family(domain)
    return {
        "requested_split": str(requested_split).strip(),
        "effective_split": effective_split,
        "available_splits": available_splits,
        "split_resolution": split_resolution,
        "requested_max_length": int(requested_max_length),
        "effective_max_length": int(info.get("eval_max_length", requested_max_length)),
        "truncate_source": bool(info.get("truncate_source", False)),
        "truncate_strategy": (
            "all_tokenizers_fixed_source_char_prefix"
            if info.get("truncate_source", False)
            else "all_tokenizers_drop_over_prompt_budget"
        ),
        "max_eval_samples": info.get("max_eval_samples"),
        "schema_version": EVAL_SCHEMA_VERSION,
        "manifest_type": EVAL_MANIFEST_TYPE,
        "duplicate_policy": EVAL_DUPLICATE_POLICY,
        "prompt_compatibility": "all_declared_checkpoint_tokenizers",
        "reference_used_for_prompt_construction": False,
        "language": language,
        "eval_family": family,
        "task_type": str(info.get("task_type", "unknown")),
    }


def _extract_eval_pair(example, domain):
    """Return normalized raw (source, reference) text for one row."""
    info = domain_info[domain]
    parser_fn = info.get("parser_fn")
    if parser_fn is not None:
        parsed = parser_fn(example)
        if not isinstance(parsed, dict):
            raise TypeError(
                f"[{domain}] parser_fn must return dict, got {type(parsed).__name__}."
            )
        source = parsed.get("text")
        reference = parsed.get("target")
    else:
        source = example.get(info["columns"]["text"])
        reference = example.get(info["columns"]["target"])

    source = source.strip() if isinstance(source, str) else None
    reference = reference.strip() if isinstance(reference, str) else None
    return source, reference


def create_prompt(text, domain, shot_type="zero-shot", shot_examples=None):
    """Create a prompt from source text only.

    The current example's gold reference is never an argument and therefore
    cannot enter the model prompt.  Few-shot demonstrations contain their own
    answers by definition; those demonstration rows are excluded from scoring.
    """
    info = domain_info[domain]
    task_type = info["task_type"]
    text_col, target_col = info["columns"]["text"], info["columns"]["target"]

    few_shot_context = ""
    if shot_type != "zero-shot" and shot_examples:
        for ex in shot_examples:
            if info.get("parser_fn"):
                parsed_ex = info["parser_fn"](ex)
                ex_text, ex_target = parsed_ex["text"], parsed_ex["target"]
            else:
                ex_text, ex_target = ex[text_col], ex[target_col]

            if task_type == "summarization":
                few_shot_context += f"Judgement: {ex_text}\nSummary: {ex_target}\n\n"
            elif task_type == "translation":
                few_shot_context += f"Texto: {ex_text}\nTraducción: {ex_target}\n\n"
            else:
                if domain.startswith("kor_"):
                    q_str, a_str = ("질문", "답변")
                elif domain.startswith("es_"):
                    q_str, a_str = ("Pregunta", "Respuesta")
                else:
                    q_str, a_str = ("Question", "Answer")
                few_shot_context += f"{q_str}: {ex_text}\n{a_str}: {ex_target}\n\n"

    if task_type == "summarization":
        return (
            "Summarize the following legal judgement:\n\n"
            f"{few_shot_context}Judgement: {text}\nSummary:"
        )
    if task_type == "math":
        return (
            "Solve the following math word problem.\n\n"
            f"{few_shot_context}Question: {text}\nAnswer:"
        )
    if task_type == "translation":
        return (
            "Traduce el siguiente texto médico al español.\n\n"
            f"{few_shot_context}Texto: {text}\nTraducción:"
        )

    if domain.startswith("kor_"):
        q_str, a_str = ("질문", "답변")
        instr = "아래 질문에 답하세요:"
    elif domain.startswith("es_"):
        q_str, a_str = ("Pregunta", "Respuesta")
        instr = "Responde a la siguiente pregunta:"
    else:
        q_str, a_str = ("Question", "Answer")
        instr = "Answer the following question:"
    return f"{instr}\n\n{few_shot_context}{q_str}: {text}\n{a_str}:"


def _stable_tokenizer_backend_hash(tokenizer) -> Optional[str]:
    """Hash tokenizer behavior while ignoring mutable runtime padding/truncation state.

    Hugging Face fast tokenizers may mutate ``backend_tokenizer`` padding and
    truncation configuration when ``__call__`` is used.  Hashing ``to_str()``
    verbatim therefore makes an evaluation-policy hash depend on call history,
    even when vocabulary and tokenization behavior are unchanged.

    The frozen-manifest compatibility policy only needs the stable tokenizer
    behavior that can affect prompt tokenization, so top-level runtime
    ``padding``/``truncation`` fields are removed before hashing.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        return None
    try:
        raw = backend.to_str()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = dict(parsed)
            parsed.pop("padding", None)
            parsed.pop("truncation", None)
            raw = json.dumps(
                parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
        return hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()
    except Exception:
        return None


def _stable_tokenizer_vocab_hash(tokenizer) -> Optional[str]:
    """Deterministic vocabulary fingerprint used when auditing prompt compatibility."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    hasher = hashlib.sha256()
    try:
        items = sorted(vocab.items(), key=lambda item: (int(item[1]), str(item[0])))
        for token, token_id in items:
            hasher.update(str(int(token_id)).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(str(token).encode("utf-8", errors="surrogatepass"))
            hasher.update(b"\0")
    except Exception:
        return None
    return hasher.hexdigest()


def _tokenizer_audit_id(tokenizer) -> dict:
    """Stable tokenizer identity for frozen-manifest compatibility hashing.

    Deliberately excludes ``name_or_path`` and mutable backend padding/truncation
    state.  Two tokenizers with identical behavior should produce the same audit
    identifier regardless of where they were loaded from or what padding call
    happened previously in the process.
    """
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "length": int(len(tokenizer)),
        "vocab_hash": _stable_tokenizer_vocab_hash(tokenizer),
        "backend_behavior_hash": _stable_tokenizer_backend_hash(tokenizer),
        "special_token_ids": {
            "pad": getattr(tokenizer, "pad_token_id", None),
            "eos": getattr(tokenizer, "eos_token_id", None),
            "bos": getattr(tokenizer, "bos_token_id", None),
            "unk": getattr(tokenizer, "unk_token_id", None),
        },
    }


def _manifest_policy_hash(domain: str, policy: dict, compatibility_tokenizers) -> str:
    payload = {
        "domain": domain,
        "policy": policy,
        "tokenizers": [_tokenizer_audit_id(tok) for tok in compatibility_tokenizers],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prompt_token_length(prompt: str, tokenizer) -> int:
    ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )["input_ids"]
    return int(len(ids))


def _prompt_fits_all(prompt: str, compatibility_tokenizers, prompt_budget: int) -> bool:
    return all(
        _prompt_token_length(prompt, tok) <= int(prompt_budget)
        for tok in compatibility_tokenizers
    )


def _fit_source_prompt_only(
    source: str,
    domain: str,
    shot_type: str,
    shot_examples,
    compatibility_tokenizers,
    prompt_budget: int,
):
    """Fit source using ONLY prompt text and the declared tokenizer set.

    Gold/reference text is intentionally not accepted by this function.
    For truncating domains, the maximal Unicode-character prefix that fits every
    declared checkpoint tokenizer is selected deterministically.
    """
    full_prompt = create_prompt(source, domain, shot_type, shot_examples)
    if _prompt_fits_all(full_prompt, compatibility_tokenizers, prompt_budget):
        return source, full_prompt

    if not domain_info[domain].get("truncate_source", False):
        return None, None

    empty_prompt = create_prompt("", domain, shot_type, shot_examples)
    if not _prompt_fits_all(empty_prompt, compatibility_tokenizers, prompt_budget):
        return None, None

    lo, hi = 0, len(source)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = source[:mid].rstrip()
        prompt = create_prompt(candidate, domain, shot_type, shot_examples)
        if _prompt_fits_all(prompt, compatibility_tokenizers, prompt_budget):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    fitted = source[:best].rstrip()
    if not fitted:
        return None, None
    prompt = create_prompt(fitted, domain, shot_type, shot_examples)
    if not _prompt_fits_all(prompt, compatibility_tokenizers, prompt_budget):
        raise RuntimeError(f"[{domain}] prompt-only source fitting invariant failed.")
    return fitted, prompt


def _stable_example_id(domain: str, source: str, reference: str) -> str:
    payload = f"{domain}\0{source}\0{reference}".encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()[:24]


def _validate_raw_manifest_cache(ds, *, expected_policy_hash: str, local_path: str):
    required = {
        "eval_prompt_text",
        "eval_reference_text",
        "eval_example_id",
        "eval_schema_version",
        "eval_policy_hash",
    }
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise RuntimeError(
            f"[STALE-EVAL-CACHE] {local_path!r} is missing {missing}. "
            f"Rebuild it with schema {EVAL_SCHEMA_VERSION}."
        )
    versions = set(str(v) for v in ds["eval_schema_version"])
    if versions != {EVAL_SCHEMA_VERSION}:
        raise RuntimeError(
            f"[STALE-EVAL-CACHE] schema={versions}, expected={EVAL_SCHEMA_VERSION}."
        )
    policy_hashes = set(str(v) for v in ds["eval_policy_hash"])
    if policy_hashes != {expected_policy_hash}:
        raise RuntimeError(
            "[STALE-EVAL-CACHE] tokenizer/prompt compatibility policy changed. "
            "Rebuild the frozen raw manifest before paper evaluation."
        )

    if len(ds) == 0:
        raise RuntimeError(f"[STALE-EVAL-CACHE] {local_path!r} is empty.")
    ids = [str(v) for v in ds["eval_example_id"]]
    if any(not value.strip() for value in ids):
        raise RuntimeError(
            f"[STALE-EVAL-CACHE] {local_path!r} contains empty eval_example_id values."
        )
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"[STALE-EVAL-CACHE] {local_path!r} contains duplicate eval_example_id "
            f"values under duplicate_policy={EVAL_DUPLICATE_POLICY!r}. Rebuild it."
        )


class PretrainingDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        raw_dataset = load_dataset(
            "text", data_files=self.data_path, split="train", streaming=True
        )
        token_buffer = []
        for example in raw_dataset:
            text = example.get("text")
            if text:
                token_buffer.extend(
                    self.tokenizer(text, truncation=False)["input_ids"]
                )
                while len(token_buffer) >= self.max_length:
                    chunk = token_buffer[: self.max_length]
                    token_buffer = token_buffer[self.max_length :]
                    yield {
                        "input_ids": torch.tensor(chunk, dtype=torch.long),
                        "attention_mask": torch.tensor([1] * len(chunk), dtype=torch.long),
                        "labels": torch.tensor(chunk, dtype=torch.long),
                    }


def _load_raw_dataset(domain: str, split_eff: str):
    info = domain_info[domain]
    dataset_name = info["dataset_name"]
    hf_config = info.get("hf_config")
    trust_remote = info.get("hf_trust_remote_code")
    kwargs = {} if trust_remote is None else {"trust_remote_code": bool(trust_remote)}
    if hf_config is not None:
        return load_dataset(dataset_name, hf_config, split=split_eff, **kwargs)
    return load_dataset(dataset_name, split=split_eff, **kwargs)


def _choose_few_shot_examples(raw_dataset, domain: str, n_shots: int, seed: int):
    """Select demonstrations deterministically; scoring rows are later excluded."""
    if n_shots <= 0:
        return None, set()
    chosen = []
    chosen_ids = set()
    for example in raw_dataset.shuffle(seed=seed):
        source, reference = _extract_eval_pair(example, domain)
        if not source or not reference:
            continue
        ex_id = _stable_example_id(domain, source, reference)
        if ex_id in chosen_ids:
            continue
        chosen.append(example)
        chosen_ids.add(ex_id)
        if len(chosen) >= n_shots:
            break
    if len(chosen) < n_shots:
        raise RuntimeError(
            f"[{domain}] requested {n_shots}-shot but found only {len(chosen)} valid demonstrations."
        )
    return chosen, chosen_ids


def get_processed_dataset(
    domain,
    tokenizer,
    max_length,
    mode="pretrain",
    split="train",
    shot_type="zero-shot",
    preprocessed_root: Optional[str] = None,
    seed: int = 777,
    compatibility_tokenizers=None,
):
    """Return pretraining data or a frozen RAW evaluation manifest.

    Evaluation mode returns NO model token IDs and NO labels.  The rows contain
    only frozen raw prompt/reference metadata.  ``compatibility_tokenizers`` is
    the predeclared tokenizer set for base + every CL checkpoint and is used only
    for prompt-length compatibility.  Reference text never participates in prompt
    construction, truncation, or compatibility filtering.
    """
    if domain not in domain_info:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(domain_info)}")
    info = domain_info[domain]

    if mode == "pretrain":
        data_path = info.get("pretrain_data_path")
        if not data_path:
            raise ValueError(f"Domain {domain!r} has no pretrain_data_path.")
        return PretrainingDataset(data_path, tokenizer, max_length)
    if mode != "evaluation":
        raise ValueError("mode must be 'pretrain' or 'evaluation'.")

    tokenizers = list(compatibility_tokenizers or [tokenizer])
    if not tokenizers:
        raise ValueError("compatibility_tokenizers must contain at least one tokenizer.")

    policy = get_eval_policy(domain, split, max_length)
    split_eff = policy["effective_split"]
    prompt_budget = int(policy["effective_max_length"])
    policy_hash = _manifest_policy_hash(domain, policy, tokenizers)
    n_shots = _num_shots_from_type(shot_type)
    shot_suffix = "" if n_shots == 0 else f"_{n_shots}shot"
    manifest_cache_key = (
        domain, split_eff, str(shot_type), int(seed), policy_hash
    )
    cached_manifest = _FROZEN_MANIFEST_CACHE.get(manifest_cache_key)
    if cached_manifest is not None:
        return cached_manifest

    # A raw-manifest cache is tokenizer-ID independent and safe to share across
    # checkpoints, but only when its compatibility policy hash matches exactly.
    if preprocessed_root is not None:
        candidates = (
            [
                os.path.join(preprocessed_root, f"{domain}_{split_eff}"),
                os.path.join(preprocessed_root, f"{domain}_{split_eff}_0shot"),
            ]
            if n_shots == 0
            else [os.path.join(preprocessed_root, f"{domain}_{split_eff}{shot_suffix}")]
        )
        local_path = next((p for p in candidates if os.path.exists(p)), None)
        if local_path is not None:
            ds = load_from_disk(local_path)
            try:
                _validate_raw_manifest_cache(
                    ds, expected_policy_hash=policy_hash, local_path=local_path
                )
            except RuntimeError as exc:
                # Do not abort the whole CL run because an on-disk manifest was
                # created with an older/non-stable audit hash.  Reject that cache
                # and rebuild the frozen manifest from the source dataset below.
                # The stale cache is never used for scoring.
                print(
                    f"[get_processed_dataset] Rejecting stale frozen raw manifest "
                    f"{local_path}: {exc}. Rebuilding in memory."
                )
                ds = None
            else:
                print(f"[get_processed_dataset] Using frozen raw manifest: {local_path}")
                _FROZEN_MANIFEST_CACHE[manifest_cache_key] = ds
                return ds

    raw_dataset = _load_raw_dataset(domain, split_eff)
    n_cap = info.get("max_eval_samples")
    if n_cap is not None and len(raw_dataset) > int(n_cap):
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(int(n_cap)))
        print(
            f"[get_processed_dataset] {domain!r}: deterministic candidate cap={n_cap}, seed={seed}."
        )

    shot_examples, shot_ids = _choose_few_shot_examples(
        raw_dataset, domain, n_shots, seed
    )

    prompts, references, ids, versions, policy_hashes = [], [], [], [], []
    removed_invalid = removed_prompt_budget = removed_shot = removed_duplicate = 0
    seen_scoring_ids = set()

    for example in raw_dataset:
        source, reference = _extract_eval_pair(example, domain)
        # Valid-reference check is the only reference-based inclusion rule.  It
        # verifies that scoring is possible; length/content never shape the prompt.
        if not source or not reference:
            removed_invalid += 1
            continue
        ex_id = _stable_example_id(domain, source, reference)
        if ex_id in shot_ids:
            # Exclude every exact duplicate of a demonstration so that a repeated
            # raw row cannot leak the few-shot answer into the scored set.
            removed_shot += 1
            continue
        if ex_id in seen_scoring_ids:
            # Some source datasets contain repeated identical source/reference rows
            # (e.g. LawQA-Ko). Scoring each copy would overweight that example and
            # violates the manifest's unique-ID invariant. Keep the first occurrence
            # deterministically and drop later exact duplicates.
            removed_duplicate += 1
            continue
        seen_scoring_ids.add(ex_id)

        _, prompt = _fit_source_prompt_only(
            source=source,
            domain=domain,
            shot_type=shot_type,
            shot_examples=shot_examples,
            compatibility_tokenizers=tokenizers,
            prompt_budget=prompt_budget,
        )
        if prompt is None:
            removed_prompt_budget += 1
            continue

        prompts.append(prompt.strip())
        references.append(reference.strip())
        ids.append(ex_id)
        versions.append(EVAL_SCHEMA_VERSION)
        policy_hashes.append(policy_hash)

    if not prompts:
        raise RuntimeError(
            f"[{domain}] frozen manifest is empty: invalid={removed_invalid}, "
            f"prompt_budget={removed_prompt_budget}, shots={removed_shot}, "
            f"duplicates={removed_duplicate}."
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(f"[{domain}] duplicate eval_example_id values in manifest.")

    print(
        f"[get_processed_dataset] {domain!r}: frozen manifest n={len(prompts)}, "
        f"removed_invalid={removed_invalid}, removed_prompt_budget={removed_prompt_budget}, "
        f"excluded_shots={removed_shot}, removed_duplicates={removed_duplicate}, "
        f"duplicate_policy={EVAL_DUPLICATE_POLICY}."
    )
    manifest = Dataset.from_dict(
        {
            "eval_prompt_text": prompts,
            "eval_reference_text": references,
            "eval_example_id": ids,
            "eval_schema_version": versions,
            "eval_policy_hash": policy_hashes,
        }
    )
    _FROZEN_MANIFEST_CACHE[manifest_cache_key] = manifest
    return manifest


def get_collate_fn(collate_type: str = "smart", pad_id: int = 0):
    """Dual-mode collator.

    Frozen evaluation manifests are returned as raw metadata lists.  Legacy
    tensor rows are retained only for non-manifest callers/pretraining utilities.
    """
    raw_keys = (
        "eval_prompt_text",
        "eval_reference_text",
        "eval_example_id",
        "eval_schema_version",
        "eval_policy_hash",
    )

    def collate(batch):
        if not batch:
            return {}
        if "eval_prompt_text" in batch[0]:
            for key in raw_keys:
                if any(key not in item for item in batch):
                    raise RuntimeError(f"Frozen manifest batch missing key {key!r}.")
            return {key: [item[key] for item in batch] for key in raw_keys}

        def as_long(x):
            return x.long() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.long)

        ids = [as_long(item["input_ids"]) for item in batch]
        masks = [as_long(item.get("attention_mask", torch.ones_like(ids[i]))) for i, item in enumerate(batch)]
        labels = [as_long(item.get("labels", ids[i])) for i, item in enumerate(batch)]
        max_len = max(int(x.numel()) for x in ids)
        out_ids = torch.full((len(ids), max_len), int(pad_id), dtype=torch.long)
        out_mask = torch.zeros((len(ids), max_len), dtype=torch.long)
        out_labels = torch.full((len(ids), max_len), -100, dtype=torch.long)
        for i, (x, m, y) in enumerate(zip(ids, masks, labels)):
            active = m.bool()
            x, y = x[active], y[active]
            n = int(x.numel())
            out_ids[i, -n:] = x
            out_mask[i, -n:] = 1
            out_labels[i, -n:] = y
        return {"input_ids": out_ids, "attention_mask": out_mask, "labels": out_labels}

    if collate_type not in {"smart", "simple"}:
        raise ValueError(f"Unknown collate_type: {collate_type}")
    return collate
