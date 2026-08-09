#cp4llm/utils/domain_map.py

import os
from typing import Tuple, Union, Optional
import re
from functools import partial
from itertools import chain

from datasets import load_dataset, Dataset, load_from_disk  
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
    """Parse eng_medical rows and extract the answer label."""
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
    """Parse GSM8K rows and extract the final answer after '####'."""
    question = example.get("question", "")
    answer_full = example.get("answer", "")
    m = re.search(r"####\s*([^\n]+)", answer_full)
    if m:
        final_ans = m.group(1).strip()
    else:
        final_ans = answer_full.strip()
    return {"text": question.strip(), "target": final_ans}

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
        "choice_accuracy": False,
    },
    "eng_medical": {
        "pretrain_data_path": "utils/data_storage/guidline_medical.txt",
        "dataset_name": "Shekswess/medical_gemma_instruct_dataset",
        "encoding": "utf-8",
        "columns": {"text": "input", "target": "output"},
        "task_type": "qa",
        "parser_fn": None,
        "choice_accuracy": False,
    },
    "kor_legal": {
        "pretrain_data_path": "utils/data_storage/new-legal-kor-dataset.txt",
        "dataset_name": "jihye-moon/LawQA-Ko",
        "encoding": "utf-8",
        "columns": {"text": "question", "target": "answer"},
        "task_type": "qa",
        "parser_fn": None,
        "choice_accuracy": False,
    },
    "eng_legal": {
        "pretrain_data_path": "utils/data_storage/eng-new-legal-dataset.txt",
        "dataset_name": "joelniklaus/legal_case_document_summarization",
        "encoding": "utf-8",
        "columns": {"text": "judgement", "target": "summary"},
        "task_type": "summarization",
        "parser_fn": None,
        "eval_max_length": 512,
        "truncate_source": True,
        "max_eval_samples": 3000,
        "choice_accuracy": False,
    },
    "math": {
        "pretrain_data_path": None,  
        "dataset_name": "openai/gsm8k",
        "hf_config": "main",             
        "encoding": "utf-8",
        "columns": {"text": "question", "target": "answer"},
        "task_type": "math",
        "parser_fn": _parse_gsm8k,
        "choice_accuracy": False,
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
        "eval_max_length": 512,
        "truncate_source": True,
        "max_eval_samples": 500,        
    },
    "es_legal": {
        "pretrain_data_path": "utils/data_storage/es-legal-dataset.txt",
        "dataset_name": "SINAI/ALIA-es-legal-administrative-cqa",
        "hf_split": "parlamint_es_an",  
        "encoding": "utf-8",
        "columns": {"text": "question", "target": "answer"},
        "task_type": "qa",
        "parser_fn": _parse_alia_legal_cqa,
        "eval_max_length": 1024,
        "truncate_source": True,        
        "max_eval_samples": 500,
    },
}

_VALIDATED_EVAL_CACHE_KEYS = set()


def _extract_eval_pair(example, domain):
    """Return normalized (source_text, target_text) for an evaluation row."""
    info = domain_info[domain]
    parser_fn = info.get("parser_fn")

    if parser_fn is not None:
        parsed = parser_fn(example)
        if not isinstance(parsed, dict):
            raise TypeError(
                f"[{domain}] parser_fn must return a dict, got {type(parsed).__name__}."
            )
        text = parsed.get("text")
        target = parsed.get("target")
    else:
        text_col = info["columns"]["text"]
        target_col = info["columns"]["target"]
        text = example.get(text_col)
        target = example.get(target_col)

    
    text = text.strip() if isinstance(text, str) else None
    target = target.strip() if isinstance(target, str) else None
    return text, target


def _reference_token_ids_and_text(target, tokenizer):
    """Tokenize a target and return (ids, decoded_non_special_text)."""
    if not isinstance(target, str) or not target.strip():
        return [], ""

    target_ids = tokenizer(
        target.strip(),
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )["input_ids"]
    target_ids = [int(token_id) for token_id in target_ids]
    decoded = (
        tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        if target_ids else ""
    )
    return target_ids, decoded


def _valid_raw_eval_row(example, tokenizer, domain):
    """Return True only for rows with non-empty source and decodable target."""
    text, target = _extract_eval_pair(example, domain)
    if not text or not target:
        return False
    target_ids, decoded = _reference_token_ids_and_text(target, tokenizer)
    return bool(target_ids) and bool(decoded)


def _cached_reference_issue(example, tokenizer):
    """Return None for a valid cached row, otherwise a concise failure reason."""
    input_ids = example.get("input_ids")
    attention_mask = example.get("attention_mask")
    labels = example.get("labels")

    if input_ids is None or attention_mask is None or labels is None:
        return "missing input_ids, attention_mask, or labels"
    if not (len(input_ids) == len(attention_mask) == len(labels)):
        return (
            "shape mismatch: "
            f"input_ids={len(input_ids)}, attention_mask={len(attention_mask)}, "
            f"labels={len(labels)}"
        )

    reference_ids = [
        int(label_id)
        for label_id, mask_value in zip(labels, attention_mask)
        if int(mask_value) == 1 and int(label_id) != -100
    ]
    if not reference_ids:
        return "no supervised target token IDs"

    decoded = tokenizer.decode(
        reference_ids,
        skip_special_tokens=True,
    ).strip()
    if not decoded:
        preview_ids = reference_ids[:12]
        preview_tokens = tokenizer.convert_ids_to_tokens(preview_ids)
        return (
            "supervised IDs decode to empty text; "
            f"ids={preview_ids}, tokens={preview_tokens}"
        )
    return None


def _validate_preprocessed_eval_cache(ds, tokenizer, domain, local_path):
    """
    Validate a loaded preprocessed cache once per process/tokenizer signature.

    Invalid cached rows are not silently dropped. Silently changing the
    denominator during paper evaluation would make results hard to audit. The
    caller must rebuild the cache from the filtered raw dataset.
    """
    cache_key = (
        os.path.realpath(local_path),
        tokenizer.__class__.__name__,
        str(getattr(tokenizer, "name_or_path", "")),
        int(len(tokenizer)),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    )
    if cache_key in _VALIDATED_EVAL_CACHE_KEYS:
        return

    bad_count = 0
    bad_examples = []
    for index, example in enumerate(ds):
        issue = _cached_reference_issue(example, tokenizer)
        if issue is not None:
            bad_count += 1
            if len(bad_examples) < 8:
                bad_examples.append((index, issue))

    if bad_count:
        raise RuntimeError(
            f"[INVALID-EVAL-CACHE][{domain}] '{local_path}' contains "
            f"{bad_count} row(s) with invalid references. First failures: "
            f"{bad_examples}. This cache was built before empty-target "
            "validation (or with a mismatched tokenizer). Move/delete this "
            "cache and rebuild it from the raw dataset with the patched "
            "domain_map.py. Do not suppress EMPTY-REF in EvalRunner."
        )

    _VALIDATED_EVAL_CACHE_KEYS.add(cache_key)


def create_prompt(text, domain, shot_type="zero-shot", shot_examples=None):
    info = domain_info[domain]
    task_type = info["task_type"]
    text_col, target_col = info["columns"]["text"], info["columns"]["target"]

    few_shot_context = ""
    if shot_type != "zero-shot" and shot_examples:
        for ex in shot_examples:
            if info.get('parser_fn'):
                parsed_ex = info['parser_fn'](ex)
                ex_text, ex_target = parsed_ex['text'], parsed_ex['target']
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
            f"Summarize the following legal judgement:\n\n"
            f"{few_shot_context}"
            f"Judgement: {text}\nSummary:"
        )
    elif task_type == "math":
        return (
            f"Solve the following math word problem.\n\n"
            f"{few_shot_context}"
            f"Question: {text}\nAnswer:"
        )
    elif task_type == "translation":   
        return (
            f"Traduce el siguiente texto médico al español.\n\n"
            f"{few_shot_context}"
            f"Texto: {text}\nTraducción:"
        )
    else:
        
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

def _fit_source_to_budget(text, target, domain, tokenizer, max_length,
                          shot_type="zero-shot", shot_examples=None):
    skeleton = create_prompt("", domain, shot_type=shot_type,
                             shot_examples=shot_examples)
    skel_ids = tokenizer(skeleton, truncation=False,
                         add_special_tokens=True)["input_ids"]
    t_ids = tokenizer(target, truncation=False,
                      add_special_tokens=False)["input_ids"]
    extra = 1 if tokenizer.eos_token_id is not None else 0

    room = max_length - len(skel_ids) - len(t_ids) - extra - 2
    if room <= 8:
        return None

    text_ids = tokenizer(text, truncation=False,
                         add_special_tokens=False)["input_ids"]
    if len(text_ids) <= room:
        return text

    for shave in (0, 8, 16):
        cand = tokenizer.decode(text_ids[:max(room - shave, 1)],
                                skip_special_tokens=True)
        p_ids = tokenizer(
            create_prompt(cand, domain, shot_type=shot_type,
                          shot_examples=shot_examples),
            truncation=False, add_special_tokens=True)["input_ids"]
        if len(p_ids) + len(t_ids) + extra <= max_length:
            return cand
    return None

class PretrainingDataset(IterableDataset):
    """Stream text data and yield fixed-length token chunks for LM pretraining."""
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        
        raw_dataset = load_dataset("text", data_files=self.data_path, split='train', streaming=True)
        
        buffer_size = 100_000  
        token_buffer = []

        for example in raw_dataset:
            text = example.get("text")
            if text:
                token_buffer.extend(self.tokenizer(text, truncation=False)["input_ids"])
                while len(token_buffer) >= self.max_length:
                    chunk = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]
                    yield {
                        "input_ids": torch.tensor(chunk, dtype=torch.long),
                        "attention_mask": torch.tensor([1] * len(chunk), dtype=torch.long),
                        "labels": torch.tensor(chunk, dtype=torch.long),
                    }

def preprocess_for_evaluation(
    examples, tokenizer, domain, max_length,
    shot_type="zero-shot", shot_examples=None
):
    """Build CausalLM evaluation inputs with prompt labels masked as -100."""
    info = domain_info[domain]
    text_col, target_col = info["columns"]["text"], info["columns"]["target"]

    texts, targets = [], []
    
    keys = list(examples.keys())
    rows = [dict(zip(keys, vals)) for vals in zip(*examples.values())]

    for row_idx, ex in enumerate(rows):
        text, target = _extract_eval_pair(ex, domain)
        target_ids, decoded_target = _reference_token_ids_and_text(target, tokenizer)

        if not text:
            raise ValueError(
                f"[{domain}] preprocess_for_evaluation received an empty or "
                f"non-string source at batch row {row_idx}. The raw validity "
                "filter should have removed it."
            )
        if not target or not target_ids or not decoded_target:
            raise ValueError(
                f"[{domain}] preprocess_for_evaluation received an empty or "
                f"special-token-only target at batch row {row_idx}. The raw "
                "validity filter should have removed it."
            )

        texts.append(text)
        targets.append(target)


    if info.get("truncate_source"):
        fitted_texts = []
        for t, tgt in zip(texts, targets):
            fitted = _fit_source_to_budget(
                t, tgt, domain, tokenizer, max_length,
                shot_type=shot_type, shot_examples=shot_examples,
            )
            fitted_texts.append(fitted if fitted is not None else t)
        texts = fitted_texts

    prompts = [create_prompt(t, domain, shot_type, shot_examples) for t in texts]

    
    prompt_enc = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    target_enc = tokenizer(
        targets,
        add_special_tokens=False,   
        truncation=False,
        padding=False,
    )

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    eos_id = tokenizer.eos_token_id

    for p_ids, t_ids in zip(prompt_enc["input_ids"], target_enc["input_ids"]):
        
        combined = p_ids + t_ids + ([eos_id] if eos_id is not None else [])
        
        
        labels = ([-100] * len(p_ids)) + t_ids + ([eos_id] if eos_id is not None else [])

        
        combined = combined[:max_length]
        labels = labels[:max_length]

        attn = [1] * len(combined)

        input_ids_list.append(combined)
        attention_mask_list.append(attn)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def make_length_filter(tokenizer, domain, max_length, shot_type="zero-shot", shot_examples=None):
    info = domain_info[domain]
    text_col, target_col = info["columns"]["text"], info["columns"]["target"]
    parser_fn = info.get("parser_fn")

    def _filter(example):
        text, target = _extract_eval_pair(example, domain)
        if not text or not target:
            return False

        t_ids, decoded_target = _reference_token_ids_and_text(target, tokenizer)
        if not t_ids or not decoded_target:
            return False

        if info.get("truncate_source"):
            fitted = _fit_source_to_budget(
                text, target, domain, tokenizer, max_length,
                shot_type=shot_type, shot_examples=shot_examples,
            )
            return fitted is not None

        prompt = create_prompt(text, domain, shot_type=shot_type, shot_examples=shot_examples)

        p_ids = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
        
        extra = 1 if tokenizer.eos_token_id is not None else 0
        return (len(p_ids) + len(t_ids) + extra) <= max_length

    return _filter


def _strip_right_padding(example, pad_id: int):
    """Remove legacy right padding using attention_mask."""
    input_ids = example.get("input_ids", [])
    attention_mask = example.get("attention_mask", [])
    labels = example.get("labels", [])

    if attention_mask:
        last_one = 0
        for i, v in enumerate(attention_mask):
            if v == 1:
                last_one = i + 1
        true_len = last_one
    else:
        true_len = len(input_ids)
        while true_len > 0 and input_ids[true_len - 1] == pad_id:
            true_len -= 1

    true_len = max(true_len, 0)

    if input_ids:
        example["input_ids"] = input_ids[:true_len]
    if attention_mask:
        example["attention_mask"] = attention_mask[:true_len]
    if labels:
        example["labels"] = labels[:true_len]

    return example

def get_processed_dataset(
    domain,
    tokenizer,
    max_length,
    mode="pretrain",
    split="train",
    shot_type="zero-shot",
    preprocessed_root: Optional[str] = None,
    seed: int = 777,
):
    info = domain_info[domain]

    if mode == "pretrain":
        data_path = info.get("pretrain_data_path")
        if not data_path:
            raise ValueError(f"pretrain_data_path is not defined for domain '{domain}'.")
        return PretrainingDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)

    if mode != "evaluation":
        raise ValueError("mode must be either 'pretrain' or 'evaluation'.")
    split_eff = info.get("hf_split", split)
    max_len_eff = int(info.get("eval_max_length", max_length))

    n_shots = _num_shots_from_type(shot_type)
    shot_suffix = "" if n_shots == 0 else f"_{n_shots}shot"
    
    if preprocessed_root is not None:
        
        candidate_paths = []
        if n_shots == 0:
            
            candidate_paths = [
                os.path.join(preprocessed_root, f"{domain}_{split}"),
                os.path.join(preprocessed_root, f"{domain}_{split}_0shot"),
            ]
        else:
            candidate_paths = [os.path.join(preprocessed_root, f"{domain}_{split}{shot_suffix}")]

        local_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if local_path is not None:
            print(f"[get_processed_dataset] Using preprocessed eval dataset from: {local_path}")
            ds = load_from_disk(local_path)

            pad_id = int(tokenizer.pad_token_id or 0)
            ds = ds.map(lambda ex: _strip_right_padding(ex, pad_id), batched=False)
            if "attention_mask" in ds.column_names:
                ds = ds.filter(lambda ex: sum(ex["attention_mask"]) > 0)

            _validate_preprocessed_eval_cache(
                ds=ds,
                tokenizer=tokenizer,
                domain=domain,
                local_path=local_path,
            )
            return ds

    dataset_name = info.get("dataset_name")
    hf_config = info.get("hf_config", None)

    _trc = info.get("hf_trust_remote_code", None)
    _load_kw = {} if _trc is None else {"trust_remote_code": bool(_trc)}

    if hf_config is not None:
        raw_dataset = load_dataset(dataset_name, hf_config, split=split_eff,
                                   **_load_kw)                                 
    else:
        raw_dataset = load_dataset(dataset_name, split=split_eff, **_load_kw)  

    n_cap = info.get("max_eval_samples")
    if n_cap is not None and len(raw_dataset) > int(n_cap):
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(int(n_cap)))
        print(f"[get_processed_dataset] '{domain}': subsampled to {n_cap} "
              f"examples (seed={seed}) before filtering.")

    
    shot_examples = None
    if n_shots > 0:
        shot_examples = []
        
        base_filter = make_length_filter(
            tokenizer=tokenizer,
            domain=domain,
            max_length=max_len_eff,     
            shot_type="zero-shot",
            shot_examples=None,
        )

        
        shuffled = raw_dataset.shuffle(seed=seed)
        for ex in shuffled:
            if base_filter(ex):
                shot_examples.append(ex)
                if len(shot_examples) >= n_shots:
                    break
        if len(shot_examples) == 0:
            shot_examples = None

    length_filter = make_length_filter(
        tokenizer=tokenizer,
        domain=domain,
        max_length=max_len_eff,         
        shot_type=shot_type,
        shot_examples=shot_examples,
    )
    n_before_filter = len(raw_dataset)
    raw_dataset = raw_dataset.filter(length_filter)
    n_removed = n_before_filter - len(raw_dataset)
    if n_removed:
        print(
            f"[get_processed_dataset] '{domain}': removed {n_removed} row(s) "
            "because the source/reference was empty, the reference decoded "
            "to no non-special text, or prompt+target exceeded the token budget."
        )

    preprocess_fn = partial(
        preprocess_for_evaluation,
        tokenizer=tokenizer,
        domain=domain,
        max_length=max_len_eff,         
        shot_type=shot_type,
        shot_examples=shot_examples,
    )

    return raw_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )


def get_collate_fn(collate_type: str = "smart", pad_id: int = 0):
    """Return either smart left-padding collation or simple stacking."""

    def _to_long_tensor(x):
        if torch.is_tensor(x):
            return x.long()
        return torch.tensor(x, dtype=torch.long)

    def _extract_valid_by_mask(ids_1d: torch.Tensor, mask_1d: torch.Tensor):
        if mask_1d is None or mask_1d.numel() == 0:
            return ids_1d
        if ids_1d.numel() != mask_1d.numel():
            L = min(ids_1d.numel(), mask_1d.numel())
            ids_1d = ids_1d[:L]
            mask_1d = mask_1d[:L]
        return ids_1d[mask_1d == 1]

    def _left_pad_2d(seqs, pad_value: int):
        if len(seqs) == 0:
            return torch.empty((0, 0), dtype=torch.long)

        lengths = [int(s.numel()) for s in seqs]
        max_len = max(lengths) if lengths else 0

        out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            l = int(s.numel())
            if l > 0:
                out[i, -l:] = s
        return out

    def smart_collate(batch):
        raw_input_ids = [_to_long_tensor(item["input_ids"]) for item in batch]

        if "attention_mask" in batch[0]:
            raw_attention = [_to_long_tensor(item["attention_mask"]) for item in batch]
        else:
            raw_attention = [torch.ones_like(x) for x in raw_input_ids]

        if "labels" in batch[0]:
            raw_labels = [_to_long_tensor(item["labels"]) for item in batch]
        else:
            raw_labels = [x.clone() for x in raw_input_ids]

        input_ids_list = []
        labels_list = []
        attention_list = []

        for ids, am, lb in zip(raw_input_ids, raw_attention, raw_labels):
            ids_valid = _extract_valid_by_mask(ids, am)
            lb_valid = _extract_valid_by_mask(lb, am)

            if lb_valid.numel() != ids_valid.numel():
                L = min(lb_valid.numel(), ids_valid.numel())
                ids_valid = ids_valid[:L]
                lb_valid = lb_valid[:L]

            am_valid = torch.ones_like(ids_valid)

            input_ids_list.append(ids_valid)
            labels_list.append(lb_valid)
            attention_list.append(am_valid)

        input_ids = _left_pad_2d(input_ids_list, pad_value=pad_id)
        attention_mask = _left_pad_2d(attention_list, pad_value=0)
        labels = _left_pad_2d(labels_list, pad_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def simple_collate(batch):
        input_ids = torch.stack([_to_long_tensor(item["input_ids"]) for item in batch])
        attention_mask = (
            torch.stack([_to_long_tensor(item["attention_mask"]) for item in batch])
            if "attention_mask" in batch[0]
            else torch.ones_like(input_ids)
        )

        if "labels" in batch[0]:
            labels = torch.stack([_to_long_tensor(item["labels"]) for item in batch])
        else:
            labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    if collate_type == "smart":
        return smart_collate
    if collate_type == "simple":
        return simple_collate
    raise ValueError(f"Unknown collate_type: {collate_type}")
