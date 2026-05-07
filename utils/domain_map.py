#cp4llm/utils/domain_map.py

import os
from typing import Tuple, Union, Optional
import re
from functools import partial
from itertools import chain

from datasets import load_dataset, Dataset, load_from_disk  # ★ load_from_disk 추가
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
    """
    eng_medical (mamachang/medical-reasoning) 전용 parser.
    수정 사항: <answer> 태그 안에서 "E: Description" 형태일 경우 "E"만 추출하도록 로직 추가.
    """
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # 1. Question 구성 (이전과 동일)
    if input_text.strip().startswith("Q:"):
        context = input_text.split("Q:", 1)[1].strip()
    else:
        context = input_text.strip()

    question = f"Q: {context}"

    # 2. Target(정답) 추출 및 클래스 분리 (핵심 수정 부분)
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    
    if answer_match:
        # 예: "E: Subarachnoid hemorrhage"
        raw_answer = answer_match.group(1).strip()
        
        # 콜론(:)을 기준으로 쪼개서 앞부분(알파벳)만 가져옴
        # 만약 "E"라고만 적혀 있어도 split(':')[0]은 "E"를 반환하므로 안전함
        answer = raw_answer.split(':')[0].strip()
        
        # (옵션) 혹시 모를 점(.) 제거 (예: "E." -> "E")
        answer = answer.replace(".", "")
    else:
        # 태그가 없는 경우 fallback
        answer = output_text.strip().split(':')[0].strip()

    return {"text": question, "target": answer}

def _parse_gsm8k(example):
    """
    GSM8K(openai/gsm8k) 전용 parser.

    - HF 데이터셋 openai/gsm8k 의 컬럼은 question / answer 로 구성됨.
      question: 문제 문장
      answer: 풀이 + 마지막 줄에 '#### 최종답' 형식의 문자열  

    - text  : question 그대로 사용
    - target: answer에서 '####' 뒤의 최종 답만 잘라서 사용
    """
    question = example.get("question", "")
    answer_full = example.get("answer", "")

    # '#### 정답' 패턴에서 정답만 추출
    m = re.search(r"####\s*([^\n]+)", answer_full)
    if m:
        final_ans = m.group(1).strip()
    else:
        # 혹시 ####가 없는 예외 케이스가 있다면 전체를 사용
        final_ans = answer_full.strip()

    return {"text": question.strip(), "target": final_ans}

# 각 도메인별 메타 정보: pretrain 파일, HF dataset, 컬럼명, 태스크 타입, parser 등
domain_info = {
    # 한국어 의료 QA
    "kor_medical": {
        "pretrain_data_path": "utils/data_storage/new-medical-kor-dataset.txt",
        "dataset_name": "ChuGyouk/medical-o1-reasoning-SFT-Ko",
        # 이 데이터셋은 Question / Reasoning / Final_answer 구조 기반으로 만들어졌고,
        # 우리가 평가에 쓸 때는 Question(질문) / Response(최종 답변) 같은 필드를 사용하는 셈.
        # 실제 HF viewer에서 features 이름은 JSON 구조에 따라 다를 수 있으므로,
        # 필요하면 여기 columns 값을 실제 컬럼명에 맞춰 한 번 더 확인해주면 좋음. :contentReference[oaicite:1]{index=1}
        "encoding": "utf-8",
        "columns": {"text": "Question", "target": "Response"},
        "task_type": "qa",
        "parser_fn": None,
    },

    # 영어 의료 reasoning (HF: mamachang/medical-reasoning)
    # "parser_fn": _parse_medical_reasoning,
    # Shekswess/medical_gemma_instruct_dataset
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

    # 한국어 법률 QA (LawQA-Ko)
    "kor_legal": {
        "pretrain_data_path": "utils/data_storage/new-legal-kor-dataset.txt",
        "dataset_name": "jihye-moon/LawQA-Ko",
        "encoding": "utf-8",
        # 공개 예제 코드에서 question 컬럼을 쓰는 걸 확인할 수 있음. :contentReference[oaicite:3]{index=3}
        "columns": {"text": "question", "target": "answer"},
        "task_type": "qa",
        "parser_fn": None,
    },

    # 영어 법률 판결문 요약 (legal_case_document_summarization)
    "eng_legal": {
        "pretrain_data_path": "utils/data_storage/eng-new-legal-dataset.txt",
        "dataset_name": "joelniklaus/legal_case_document_summarization",
        "encoding": "utf-8",
        # 이 데이터셋은 judgement(긴 판결문), summary(요약) 구조를 가진 법률 요약 데이터셋임. :contentReference[oaicite:4]{index=4}
        "columns": {"text": "judgement", "target": "summary"},
        "task_type": "summarization",
        "parser_fn": None,
    },
    
    "math": {
        "pretrain_data_path": None,  # pretrain 용으로는 사용하지 않을 것이므로 None
        "dataset_name": "openai/gsm8k",
        "hf_config": "main",             # openai/gsm8k 의 subset(main / socratic 중 main 사용)
        "encoding": "utf-8",
        # 원본 컬럼 이름(question, answer) – parser_fn 이 최종 text/target을 만들어줌
        "columns": {"text": "question", "target": "answer"},
        "task_type": "math",
        "parser_fn": _parse_gsm8k,
    },    
}


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
            else:
                q_str, a_str = ("질문", "답변") if domain.startswith("kor_") else ("Question", "Answer")
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
    else:
        q_str, a_str = ("질문", "답변") if domain.startswith("kor_") else ("Question", "Answer")
        instr = "아래 질문에 답하세요:" if domain.startswith("kor_") else "Answer the following question:"
        return f"{instr}\n\n{few_shot_context}{q_str}: {text}\n{a_str}:"


# ===================== PretrainingDataset (pretrain mode) =====================

class PretrainingDataset(IterableDataset):
    """
    LM pretraining 용 스트리밍 IterableDataset.
    text 파일을 load_dataset(\"text\")로 스트리밍하고,
    토큰을 모아 max_length 단위로 잘라서 반환.
    """
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        # 스트리밍으로 파일을 읽기 시작
        raw_dataset = load_dataset("text", data_files=self.data_path, split='train', streaming=True)
        
        buffer_size = 100_000  # 한 번에 처리할 토큰 버퍼 크기 (사용 안 하지만 남겨둠)
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


# ===================== 평가용 전처리 함수 =====================

def preprocess_for_evaluation(
    examples, tokenizer, domain, max_length,
    shot_type="zero-shot", shot_examples=None
):
    """
    CausalLM 평가용 전처리:
    - prompt + target을 하나의 input_ids로 결합
    - labels는 prompt 구간 -100 마스크, target 구간만 정답
    """
    info = domain_info[domain]
    text_col, target_col = info["columns"]["text"], info["columns"]["target"]

    texts, targets = [], []
    # batched=True 입력(dict[list]) -> row dict로 변환
    keys = list(examples.keys())
    rows = [dict(zip(keys, vals)) for vals in zip(*examples.values())]

    for ex in rows:
        if info.get("parser_fn"):
            parsed = info["parser_fn"](ex)
            texts.append(parsed["text"])
            targets.append(parsed["target"])
        else:
            texts.append(ex[text_col])
            targets.append(ex[target_col])

    prompts = [create_prompt(t, domain, shot_type, shot_examples) for t in texts]

    # ✅ padding 없이 개별 토큰 목록 확보
    prompt_enc = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    target_enc = tokenizer(
        targets,
        add_special_tokens=False,   # 보통 target엔 special token 안 붙임
        truncation=False,
        padding=False,
    )

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    eos_id = tokenizer.eos_token_id

    for p_ids, t_ids in zip(prompt_enc["input_ids"], target_enc["input_ids"]):
        # prompt + target (+ eos)
        combined = p_ids + t_ids + ([eos_id] if eos_id is not None else [])
        
        # labels: prompt는 -100, target(+eos)만 학습 신호
        labels = ([-100] * len(p_ids)) + t_ids + ([eos_id] if eos_id is not None else [])

        # ✅ max_length 자르기
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
        if parser_fn is not None:
            parsed = parser_fn(example)
            text = parsed["text"]
            target = parsed["target"]
        else:
            text = example[text_col]
            target = example[target_col]

        prompt = create_prompt(text, domain, shot_type=shot_type, shot_examples=shot_examples)

        p_ids = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
        t_ids = tokenizer(target, truncation=False, add_special_tokens=False)["input_ids"]

        extra = 1 if tokenizer.eos_token_id is not None else 0
        return (len(p_ids) + len(t_ids) + extra) <= max_length

    return _filter


def _strip_right_padding(example, pad_id: int):
    """
    과거에 right padding된 채로 save_to_disk된 샘플을
    attention_mask 기준으로 '진짜 길이'로 되돌린다.
    """
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
            raise ValueError(f"Domain '{domain}'에 대한 pretrain_data_path가 정의되지 않았습니다.")
        return PretrainingDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)

    if mode != "evaluation":
        raise ValueError("mode는 'pretrain' 또는 'evaluation' 이어야 합니다.")
    n_shots = _num_shots_from_type(shot_type)
    shot_suffix = "" if n_shots == 0 else f"_{n_shots}shot"
    # 1) preprocessed 우선
    if preprocessed_root is not None:
        # few-shot일 때 zero-shot 캐시를 잘못 재사용하지 않도록 suffix 분리
        candidate_paths = []
        if n_shots == 0:
            # 기존 호환: {domain}_{split} 우선, 없으면 {domain}_{split}_0shot도 허용
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

            return ds

    # 2) HF raw 로딩 (★ config 지원)
    dataset_name = info.get("dataset_name")
    hf_config = info.get("hf_config", None)

    if hf_config is not None:
        raw_dataset = load_dataset(dataset_name, hf_config, split=split)
    else:
        raw_dataset = load_dataset(dataset_name, split=split)

    # 2.5) few-shot 예시 샘플링 (도메인별 1회 고정)
    shot_examples = None
    if n_shots > 0:
        shot_examples = []
        # shot 예시 자체가 너무 긴 경우를 줄이기 위해(최소한) zero-shot 기준 길이 필터로 선별
        base_filter = make_length_filter(
            tokenizer=tokenizer,
            domain=domain,
            max_length=max_length,
            shot_type="zero-shot",
            shot_examples=None,
        )

        # reproducible sampling: shuffle(seed=seed) 권장 패턴 :contentReference[oaicite:1]{index=1}
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
        max_length=max_length,
        shot_type=shot_type,
        shot_examples=shot_examples,
    )
    raw_dataset = raw_dataset.filter(length_filter)

    preprocess_fn = partial(
        preprocess_for_evaluation,
        tokenizer=tokenizer,
        domain=domain,
        max_length=max_length,
        shot_type=shot_type,
        shot_examples=shot_examples,
    )

    return raw_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )

def get_collate_fn(collate_type: str = "smart", pad_id: int = 0):
    """
    - 'smart': ✅ 강력 권장
        1) attention_mask 기반 unpad (저장본 right padding 제거)
        2) '진짜 left padding'
    - 'simple': 동일 길이 가정
    """

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
