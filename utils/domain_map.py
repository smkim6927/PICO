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


def _parse_medical_reasoning(example):
    """
    eng_medical (mamachang/medical-reasoning) 전용 parser.
    이 데이터셋은 features: ['output', 'input', 'instruction'] 구조를 가짐. :contentReference[oaicite:0]{index=0}
    우리가 쓰는 건 input(질문+컨텍스트) / output(답변) 이라서 아래처럼 매핑하는 게 맞음.
    """
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    # input에서 'Q:' 이후를 질문으로 사용 (없으면 전체)
    question = input_text.split("Q:", 1)[1].strip() if "Q:" in input_text else input_text

    # output에서 <answer>...</answer> 태그만 추출 (없으면 전체 출력 사용)
    answer_match = re.search(r'<answer>(.*?)</answer>', output_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else output_text

    return {"text": question, "target": answer}


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
    "eng_medical": {
        "pretrain_data_path": "utils/data_storage/guidline_medical.txt",
        "dataset_name": "mamachang/medical-reasoning",
        "encoding": "utf-8",
        # 이 데이터셋은 features: ['output', 'input', 'instruction']이므로 :contentReference[oaicite:2]{index=2}
        # columns 를 그대로 쓰는 대신 parser_fn에서 text/target을 만들어냄.
        "columns": {"text": "input", "target": "output"},
        "task_type": "qa",
        "parser_fn": _parse_medical_reasoning,
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
):
    info = domain_info[domain]

    if mode == "pretrain":
        data_path = info.get("pretrain_data_path")
        if not data_path:
            raise ValueError(f"Domain '{domain}'에 대한 pretrain_data_path가 정의되지 않았습니다.")
        return PretrainingDataset(data_path=data_path, tokenizer=tokenizer, max_length=max_length)

    if mode != "evaluation":
        raise ValueError("mode는 'pretrain' 또는 'evaluation' 이어야 합니다.")

    # 1) preprocessed 우선
    if preprocessed_root is not None:
        local_path = os.path.join(preprocessed_root, f"{domain}_{split}")
        if os.path.exists(local_path):
            print(f"[get_processed_dataset] Using preprocessed eval dataset from: {local_path}")
            ds = load_from_disk(local_path)

            # ✅ 과거 right padding 제거
            pad_id = int(tokenizer.pad_token_id or 0)
            ds = ds.map(lambda ex: _strip_right_padding(ex, pad_id), batched=False)
            if "attention_mask" in ds.column_names:
                ds = ds.filter(lambda ex: sum(ex["attention_mask"]) > 0)

            return ds

    # 2) HF raw
    dataset_name = info.get("dataset_name")
    raw_dataset = load_dataset(dataset_name, split=split)

    length_filter = make_length_filter(
        tokenizer=tokenizer,
        domain=domain,
        max_length=max_length,
        shot_type=shot_type,
        shot_examples=None,
    )
    raw_dataset = raw_dataset.filter(length_filter)

    preprocess_fn = partial(
        preprocess_for_evaluation,
        tokenizer=tokenizer,
        domain=domain,
        max_length=max_length,
        shot_type=shot_type,
        shot_examples=None,
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
                out[i, -l:] = s  # ✅ 오른쪽 정렬 => left padding
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
