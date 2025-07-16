import os, re
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from accelerate import Accelerator
import wandb
import random
from tqdm import tqdm
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimizer.upgd import UPGD32
from utils.metrics import calculate_metrics


seed=777
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(seed)

##################################
# 1) 도메인 매핑 정보 + 전처리 함수
##################################
domain_info = {
    "eng_legal": {
        "dataset_name": "joelniklaus/legal_case_document_summarization",
        "text_column": "judgement",
        "target_column": "summary",
        "task_type": "summarization",
    },
    "eng_medical_old": {
        "dataset_name": "openlifescienceai/medmcqa",
        "text_column": "question",  # Column containing the questions
        "target_column": "opa",    # For simplicity, using option A as placeholder label
        "task_type": "qa",         # multiple-choice QA
    },
    "eng_medical":{
        "dataset_name": "mamachang/medical-reasoning", 
        "text_column": "input",
        "target_column": "output",
        "task_type": "qa",
    },
    "kor_legal": {
        "dataset_name": "jihye-moon/LawQA-Ko", 
        "text_column": "question",
        "target_column": "answer",
        "task_type": "qa",         # 일반 QA
    },
    "kor_medical": {
        "dataset_name": "ChuGyouk/medical-o1-reasoning-SFT-Ko",
        "text_column": "Question",
        "target_column": "Response",
        "task_type": "qa",         # 일반 QA
    },
}


def create_prompt(text, domain, shot_type="zero-shot", shot_examples=None):
    """
    도메인과 shot_examples(1,3,5 shot 등)에 따라 적절한 프롬프트를 생성한다.
    """
    info = domain_info[domain]
    task_type = info["task_type"]

    few_shot_context = ""
    if shot_type != "zero-shot" and shot_examples is not None:
        for ex in shot_examples:
            ex_text = ex[info["text_column"]]
            ex_target = ex[info["target_column"]]

            # 도메인/태스크마다 어떻게 예시를 보여줄지 달리 설정
            if domain == "eng_legal" and task_type == "summarization":
                few_shot_context += f"Judgement: {ex_text}\nSummary: {ex_target}\n\n"
            elif domain == "eng_medical":  # multiple-choice QA 예시일 수도 있음
                few_shot_context += f"Question: {ex_text}\nAnswer: {ex_target}\n\n"
            elif domain == "kor_legal":
                few_shot_context += f"질문: {ex_text}\n답변: {ex_target}\n\n"
            elif domain == "kor_medical":
                few_shot_context += f"질문: {ex_text}\n답변: {ex_target}\n\n"
            else:
                few_shot_context += f"Input: {ex_text}\nOutput: {ex_target}\n\n"

    # 실제 프롬프트 생성
    if domain == "eng_legal" and task_type == "summarization":
        prompt = (
            "Summarize the following legal judgement:\n\n"
            f"{few_shot_context}"
            f"Judgement: {text}\nSummary:"
        )
    elif domain == "eng_medical" and task_type == "qa":
        prompt = (
            f"Answer the following medical question in English:\n\n"
            f"{few_shot_context}"
            f"Question: {text}\nAnswer:"
        )
    elif domain == "kor_legal" and task_type == "qa":
        prompt = (
            f"아래 법률 관련 질문에 답하세요:\n\n"
            f"{few_shot_context}"
            f"질문: {text}\n답변:"
        )
    elif domain == "kor_medical" and task_type == "qa":
        prompt = (
            f"아래 의료 관련 질문에 답하세요:\n\n"
            f"{few_shot_context}"
            f"질문: {text}\n답변:"
        )
    else:
        prompt = (
            f"{few_shot_context}"
            f"Input: {text}\nOutput:"
        )

    return prompt


def law_preprocess_function(examples, domain, shot_type, shot_examples, tokenizer, max_length=512):
    """
    Summarization(예: eng_legal) 혹은 일반 QA(예: kor_legal)에 사용하는 전처리.
    """
    info = domain_info[domain]
    text_column = info["text_column"]
    target_column = info["target_column"]

    texts = examples[text_column]
    targets = examples[target_column]

    # domain/task에 맞는 prompt 생성
    prompts = [
        create_prompt(t, domain, shot_type, shot_examples)
        for t in texts
    ]

    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]

    if model_inputs["attention_mask"].dim() == 1:
        model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(0)

    return model_inputs


# def medi_preprocess_function(examples, domain, shot_type, shot_examples, tokenizer, max_length=1024):
#     """
#     eng_medical (openlifescienceai/medmcqa) 전용 전처리 함수.
#     Multiple-choice이므로 opa/opb/opc/opd/cop 컬럼을 처리.
#     """
#     questions = examples["question"]
#     options = [examples["opa"], examples["opb"], examples["opc"], examples["opd"]]
#     correct_answers = examples["cop"]

#     prompts = []
#     for i, question in enumerate(questions):
#         prompt = (
#             f"Question: {question}\n"
#             f"Options:\n"
#             f"A) {options[0][i]}\n"
#             f"B) {options[1][i]}\n"
#             f"C) {options[2][i]}\n"
#             f"D) {options[3][i]}\n"
#             f"Answer:"
#         )
#         prompts.append(prompt)

#     model_inputs = tokenizer(
#         prompts,
#         max_length=max_length,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )
#     labels = tokenizer(
#         [options[correct_answers[i] - 1][i] for i in range(len(correct_answers))],
#         max_length=max_length,
#         truncation=True,
#         padding="max_length",
#         return_tensors="pt"
#     )
#     model_inputs["labels"] = labels["input_ids"]
    
#     return model_inputs

def medi_preprocess_function_new(examples, domain, shot_type, shot_examples, tokenizer, max_length=1024):
    """
    mamachang/medical-reasoning 데이터셋 전용 전처리 함수.
    """
    inputs = examples["input"]
    outputs = examples["output"]
    instructions = examples["instruction"]
    
    questions = []
    answers = []
    
    for i, (input_text, output_text) in enumerate(zip(inputs, outputs)):
        try:
            # Input text에서 질문과 선택지 추출
            question = input_text
            if "Q:" in input_text:
                question = input_text.split("Q:", 1)[1].strip()
            
            # Output text에서 정답 추출
            answer = output_text
            answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            
            questions.append(question)
            answers.append(answer)
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            questions.append(input_text)
            answers.append(output_text)

    # 프롬프트 생성
    prompts = [
        create_prompt(q, domain, shot_type, shot_examples)
        for q in questions
    ]

    # 토크나이징 - input
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 토크나이징 - labels 
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            answers,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    model_inputs["labels"] = labels["input_ids"]
    
    # Ensure all tensors are properly shaped
    if model_inputs["attention_mask"].dim() == 1:
        model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(0)
    if model_inputs["labels"].dim() == 1:
        model_inputs["labels"] = model_inputs["labels"].unsqueeze(0)
        
    return model_inputs

def parse_input_text(input_text):
    """
    input 텍스트에서 질문과 선택지 파싱
    """
    # Q: 로 시작하는 질문 부분과 선택지 딕셔너리 부분 분리
    if "Q:" in input_text:
        parts = input_text.split("Q:", 1)[1]  # Q: 이후 부분만 가져오기
        
        # 선택지 딕셔너리 찾기 (중괄호로 둘러싸인 부분)
        options_match = re.search(r'\{[^}]+\}', parts)
        
        if options_match:
            options_str = options_match.group(0)
            question = parts[:options_match.start()].strip()
            
            # 선택지 딕셔너리 파싱
            try:
                # 안전한 파싱을 위해 ast.literal_eval 사용
                options = ast.literal_eval(options_str)
            except:
                # ast.literal_eval 실패 시 정규식으로 파싱
                options = parse_options_regex(options_str)
                
        else:
            # 선택지를 찾지 못한 경우
            question = parts.strip()
            options = {}
            
    else:
        question = input_text.strip()
        options = {}
    
    return {
        "question": question,
        "options": options
    }

def parse_options_regex(options_str):
    """
    정규식을 사용한 선택지 파싱 (백업 방법)
    """
    options = {}
    # 'A': 'text', 'B': 'text' 형태 파싱
    pattern = r"'([A-E])'\s*:\s*'([^']+)'"
    matches = re.findall(pattern, options_str)
    
    for letter, text in matches:
        options[str(letter)] = str(text)        
        print(f'letter:{letter}\t option:{options[letter]}')
    
    return options

def extract_answer_from_output(output_text):
    """
    output에서 <answer> 태그 내의 정답 추출
    """
    # <answer>...</answer> 태그에서 내용 추출
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # 정답에서 첫 번째 문자(A, B, C, D, E) 추출
        answer_letter_match = re.match(r'^([A-E])', answer_content)
        
        if answer_letter_match:
            return answer_letter_match.group(1)
    
    return None


def qa_preprocess_function(examples, domain, shot_type, shot_examples, tokenizer, max_length=512):
    """
    kor_medical 등 단순 question-answer 형태 처리 전용 함수.
    """
    info = domain_info[domain]
    text_column = info["text_column"]
    target_column = info["target_column"]

    texts = examples[text_column]
    targets = examples[target_column]

    prompts = [
        create_prompt(t, domain, shot_type, shot_examples)
        for t in texts
    ]

    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]

    if model_inputs["attention_mask"].dim() == 1:
        model_inputs["attention_mask"] = model_inputs["attention_mask"].unsqueeze(0)

    return model_inputs


def get_dataset(domain, split="train"):
    info = domain_info[domain]
    dataset_name = info["dataset_name"]
    if not dataset_name:
        raise ValueError(f"Domain '{domain}'에 해당하는 dataset_name이 없음.")
    ds = load_dataset(dataset_name, split=split)
    return ds

##################################
# 2) Eval 클래스
##################################
class Eval:
    def __init__(self, model_name, accelerator, batch_size=1, max_length=512, num_epochs=3):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.max_length = max_length
        # self.optimizer = UPGD32(self.model.parameters(), lr=1e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.num_epochs = num_epochs

    def evaluate(self, domain="eng_legal", shot_type="zero-shot"):
        """
        도메인 + shot_type 기반으로 데이터셋 로드 및 학습/평가 수행.
        """
        # if self.accelerator.is_main_process:
            # wandb.init(project="Evaluation", name=f"{domain}_{shot_type}", config={"shots": shot_type})

        # 1) 데이터셋 로드
        dataset = get_dataset(domain, split="train")

        # 2) shot examples 준비
        shot_examples = None
        if shot_type != "zero-shot":
            if shot_type == "1shot":
                shot_examples = list(dataset.select(range(1)))
            elif shot_type == "3shot":
                shot_examples = list(dataset.select(range(3)))
            elif shot_type == "5shot":
                shot_examples = list(dataset.select(range(5)))
            elif shot_type == "10shot":
                shot_examples = list(dataset.select(range(10)))

        # 3) 전처리 함수 분기
        def fn(x):
            if domain == "eng_legal":
                return law_preprocess_function(
                    x, domain, shot_type, shot_examples,
                    self.tokenizer, self.max_length
                )
            elif domain == "eng_medical":
                return medi_preprocess_function_new(
                    x, domain, shot_type, shot_examples,
                    self.tokenizer, self.max_length
                )
            elif domain == "kor_legal":
                return law_preprocess_function(
                    x, domain, shot_type, shot_examples,
                    self.tokenizer, self.max_length
                )
            elif domain == "kor_medical":
                return qa_preprocess_function(
                    x, domain, shot_type, shot_examples,
                    self.tokenizer, self.max_length
                )
            else:
                raise ValueError(f"Unsupported domain: {domain}")

        tokenized_dataset = dataset.map(
            fn,
            batched=True,
            remove_columns=dataset.column_names
        )

        # 4) Distributed Sampler
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=True
        )

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=lambda data: {
                "input_ids": torch.stack([torch.tensor(f["input_ids"]) if not torch.is_tensor(f["input_ids"]) else f["input_ids"] for f in data]),
                "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) if not torch.is_tensor(f["attention_mask"]) else f["attention_mask"] for f in data]),
                "labels": torch.stack([torch.tensor(f["labels"]) if not torch.is_tensor(f["labels"]) else f["labels"] for f in data])
            },
            drop_last=True
        )

        # 5) Accelerator prepare
        self.model, self.optimizer, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, dataloader
        )

        for epoch in range(self.num_epochs):
            self.model.train()
            sampler.set_epoch(epoch)

            progress_bar = tqdm(
                dataloader,
                desc=f"[{domain}/{shot_type}] Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
                leave=True
            )

            # 에폭 동안 로컬에 임시 저장할 리스트
            local_input_ids = []
            local_label_ids = []
            local_prediction_ids = []
            local_losses = []
            plasticity_scores = []

            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # 업데이트 전 loss 계산
                with torch.no_grad():
                    out_before = self.model(
                        input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )
                    loss_before = out_before.loss.item()

                # 학습 단계
                out = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 업데이트 후 loss 계산
                with torch.no_grad():
                    out_after = self.model(
                        input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )
                    loss_after = out_after.loss.item()

                plasticity = max(1 - (loss_after / max(loss_before, 1e-6)), 0)
                plasticity_scores.append(plasticity)

                logits = out.logits
                preds = torch.argmax(logits, dim=-1)

                progress_bar.set_postfix({"loss": loss.item()})

                # 로컬 저장 (input_ids, labels, preds)
                for i in range(len(input_ids)):
                    input_text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    label_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)

                    # Ensure preds[i] is a tensor
                    pred_item = preds[i]
                    if not isinstance(pred_item, torch.Tensor):
                        pred_item = torch.tensor(pred_item, device=self.accelerator.device)
                    pred_text = self.tokenizer.decode(pred_item, skip_special_tokens=True)

                    in_ids = self.tokenizer.encode(input_text, return_tensors="pt")[0].to(self.accelerator.device)
                    lb_ids = self.tokenizer.encode(label_text, return_tensors="pt")[0].to(self.accelerator.device)
                    pd_ids = self.tokenizer.encode(pred_text, return_tensors="pt")[0].to(self.accelerator.device)

                    local_input_ids.append(in_ids)
                    local_label_ids.append(lb_ids)
                    local_prediction_ids.append(pd_ids)
                    local_losses.append(torch.tensor(loss.item(), device=self.accelerator.device))

            # ---------- EPOCH EVALUATION (같은 데이터셋으로 평가) ----------
            self.model.eval()

            if len(local_input_ids) > 0:
                local_input_ids = pad_sequence(
                    local_input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                )
                local_label_ids = pad_sequence(
                    local_label_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                )
                local_prediction_ids = pad_sequence(
                    local_prediction_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                )
                local_losses = torch.stack(local_losses, dim=0)

            padded_input_ids = self.accelerator.pad_across_processes(
                local_input_ids, 
                pad_index=self.tokenizer.pad_token_id,
                dim=0
            )
            padded_label_ids = self.accelerator.pad_across_processes(
                local_label_ids, 
                pad_index=self.tokenizer.pad_token_id,
                dim=0
            )
            padded_pred_ids = self.accelerator.pad_across_processes(
                local_prediction_ids, 
                pad_index=self.tokenizer.pad_token_id,
                dim=0
            )
            padded_losses = self.accelerator.pad_across_processes(local_losses, dim=0)

            gathered_input_ids = self.accelerator.gather(padded_input_ids)
            gathered_label_ids = self.accelerator.gather(padded_label_ids)
            gathered_pred_ids = self.accelerator.gather(padded_pred_ids)
            gathered_losses = self.accelerator.gather(padded_losses)

            plasticity_tensor = torch.tensor(plasticity_scores, device=self.accelerator.device)
            padded_plasticity = self.accelerator.pad_across_processes(plasticity_tensor, dim=0)
            gathered_plasticity = self.accelerator.gather(padded_plasticity)

            if self.accelerator.is_main_process:
                all_losses = gathered_losses.cpu().tolist()
                plasticity_list = gathered_plasticity.cpu().tolist()

                all_inputs = [
                    self.tokenizer.decode(row, skip_special_tokens=True)
                    for row in gathered_input_ids
                ]
                all_labels = [
                    self.tokenizer.decode(row, skip_special_tokens=True)
                    for row in gathered_label_ids
                ]
                all_preds = [
                    self.tokenizer.decode(row, skip_special_tokens=True)
                    for row in gathered_pred_ids
                ]

                # 메트릭 계산
                metrics = calculate_metrics(
                    predictions=all_preds,
                    labels=all_labels,
                    losses=all_losses,
                    inputs=all_inputs
                )

                wandb.log({
                    "epoch": epoch + 1,
                    "domain": domain,
                    "shot_type": shot_type,
                    "Average Loss": metrics["loss"],
                    "ROUGE-1": metrics["rouge1"],
                    "ROUGE-2": metrics["rouge2"],
                    "ROUGE-L": metrics["rougeL"],
                    "BLEU": metrics["bleu"],
                    "METEOR": metrics["meteor"],
                    "F1": metrics["f1"],
                    "R²": metrics["r2"],
                    "Exact Match": metrics["exact_match"],
                    "Groundedness": metrics["groundedness"],
                    "Cosine Similarity": metrics["cosine_similarity"],
                    "Jaccard Similarity": metrics["jaccard_similarity"],
                    "Plasticity": sum(plasticity_list) / len(plasticity_list) if plasticity_list else 0.0,
                })

                # 추가: 모델의 질문에 대한 답변(입력, 예측, 정답)을 wandb 테이블로 기록
                num_samples = min(5, len(all_inputs))
                sample_data = []
                for i in range(num_samples):
                    sample_data.append([all_inputs[i], all_preds[i], all_labels[i]])
                example_table = wandb.Table(data=sample_data, columns=["Input", "Prediction", "Ground Truth"])
                wandb.log({"sample_predictions": example_table})

        if self.accelerator.is_main_process:
            wandb.finish()


def main():
    set_seed(777)
    accelerator = Accelerator()

    evaluator = Eval(
        model_name="/home/jovyan/sumin_data/results/saved_model/krmd-krlaw-egmd-eglaw",
        accelerator=accelerator,
        batch_size=1,
        max_length=512,
        num_epochs=1
    )
    accelerator.print(f"Starting evaluation on {accelerator.num_processes} GPUs")

    # 각 도메인별로 zero-shot, 1shot, 3shot 등 실행 예시
    # evaluator.evaluate(domain="eng_legal", shot_type="zero-shot")
    # evaluator.evaluate(domain="eng_legal", shot_type="1shot")
    # evaluator.evaluate(domain="eng_legal", shot_type="3shot")
    # evaluator.evaluate(domain="eng_legal", shot_type="5shot")
    # evaluator.evaluate(domain="eng_legal", shot_type="10shot")

    # evaluator.evaluate(domain="eng_medical", shot_type="zero-shot")
    evaluator.evaluate(domain="eng_medical", shot_type="1shot")
    # evaluator.evaluate(domain="eng_medical", shot_type="3shot")
    # evaluator.evaluate(domain="eng_medical", shot_type="5shot")
    # evaluator.evaluate(domain="eng_medical", shot_type="10shot")

    # evaluator.evaluate(domain="kor_legal", shot_type="zero-shot")
    # evaluator.evaluate(domain="kor_legal", shot_type="1shot")
    # evaluator.evaluate(domain="kor_legal", shot_type="3shot")
    # evaluator.evaluate(domain="kor_legal", shot_type="5shot")
    # evaluator.evaluate(domain="kor_legal", shot_type="10shot")

    # evaluator.evaluate(domain="kor_medical", shot_type="zero-shot")
    # evaluator.evaluate(domain="kor_medical", shot_type="1shot")
    # evaluator.evaluate(domain="kor_medical", shot_type="3shot")
    # evaluator.evaluate(domain="kor_medical", shot_type="5shot")
    # evaluator.evaluate(domain="kor_medical", shot_type="10shot")

    accelerator.print("Done!")

if __name__ == "__main__":
    main()
