# eval/metrics.py

import math
import torch
import numpy as np
from sklearn.metrics import f1_score, r2_score   # r2_score는 사용하지 않지만 그대로 둠
from rouge_score import rouge_scorer
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###############################################################
# 1. 연속 학습(Continual Learning) 관련 지표
###############################################################

def compute_avgf(accuracies_max, accuracies_final, current_task):
    """
    Average Forgetting (AvgF)

    - accuracies_max[i]   : 태스크 i에 대해 지금까지 관측한 최대 성능 (예: max_t R_{t,i})
    - accuracies_final[i] : 현재 시점(current_task)에서 태스크 i의 성능 (예: R_{T,i})
    - current_task        : 지금까지 학습 완료된 태스크 개수 (1-based, T)

    보편적인 정의:
        F_T = (1/(T-1)) * sum_{i=1..T-1} ( max_{t<=T} R_{t,i} - R_{T,i} )
    를 accuracy 버전으로 구현한 것.
    """
    if current_task <= 1:
        return 0.0
    
    forgetting_sum = 0.0
    # 태스크 인덱스를 0-based로 간주: 0 .. current_task-2
    for i in range(current_task - 1):
        forgetting_sum += (accuracies_max[i] - accuracies_final[i])
    return forgetting_sum / (current_task - 1)


def compute_bwt(accuracies_final, accuracies_init, current_task):
    """
    Backward Transfer (BWT)

    - accuracies_init[i]  : 태스크 i를 '처음 학습 직후'의 성능 (예: R_{i,i})
    - accuracies_final[i] : 현재 시점(current_task)에서 태스크 i의 성능 (예: R_{T,i})
    - current_task        : 지금까지 학습 완료된 태스크 개수 (1-based, T)

    Lopez-Paz & Ranzato (2017)의 정의와 같은 형태:
        BWT_T = (1/(T-1)) * sum_{i=1..T-1} ( R_{T,i} - R_{i,i} )
    """
    if current_task <= 1:
        return 0.0
    
    transfer_sum = 0.0
    for i in range(current_task - 1):
        transfer_sum += (accuracies_final[i] - accuracies_init[i])
    return transfer_sum / (current_task - 1)


def compute_fisher_information(model, dataloader, criterion):
    """
    Fisher Information을 추정하는 함수.

    - Empirical Fisher: E[ (∂ log p(y|x) / ∂θ)^2 ] 를
      배치 단위 gradient의 제곱을 평균내는 방식으로 근사.:contentReference[oaicite:3]{index=3}

    구현상 선택:
    - 각 배치에서 loss = CrossEntropyLoss(ignore_index=-100, reduction='mean')
    - 그에 대한 gradient를 g_batch 라고 할 때,
      Fisher ≈ (1 / #batches) * sum_batch g_batch^2
    - IterableDataset 지원을 위해 dataloader.dataset 길이에 의존하지 않음.
    """
    # Fisher 텐서 초기화
    fisher_information = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    model.eval()
    device = next(model.parameters()).device

    num_batches = 0

    for batch in dataloader:
        # 배치를 디바이스로 이동
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        labels = batch["labels"]

        # CrossEntropyLoss(ignore_index=-100, reduction='mean' 혹은 'sum') 을 criterion으로 가정
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher_information:
                fisher_information[name] += param.grad.detach().pow(2)

        num_batches += 1

    if num_batches > 0:
        for name in fisher_information:
            fisher_information[name] /= float(num_batches)
    
    return fisher_information


def compute_delta_fisher(fisher_t, fisher_t_minus_1):
    """
    Fisher Information 변화량(L1 norm)을 계산.
    두 시점 t, t-1 에서의 Fisher 텐서 차이의 L1 norm 합.
    """
    delta = 0.0
    for name in fisher_t:
        if name in fisher_t_minus_1:
            delta += torch.sum(
                torch.abs(fisher_t[name] - fisher_t_minus_1[name])
            ).item()
    return delta


def update_ema_parameters(ema_params, current_params, alpha=0.999):
    """
    Exponential Moving Average(EMA) 파라미터 업데이트.

    ema_new = alpha * ema_old + (1 - alpha) * current
    """
    if ema_params is None:
        # 첫 스텝인 경우 현재 파라미터를 그대로 EMA로 사용
        return {name: param.clone().detach() for name, param in current_params.items()}
    
    updated_ema = {}
    for name, param in current_params.items():
        if name in ema_params:
            updated_ema[name] = alpha * ema_params[name] + (1 - alpha) * param
        else:
            updated_ema[name] = param.clone().detach()
    
    return updated_ema


def compute_ema_drift(ema_params, current_params):
    """
    현재 파라미터와 EMA 파라미터 간의 Drift를 계산.

    drift = ||θ_t - θ_ema||_2 / ||θ_ema||_2

    - 분모는 EMA 파라미터의 L2 노름으로 정규화.
    """
    if ema_params is None:
        return 0.0

    ema_norm_sq = 0.0
    diff_norm_sq = 0.0
    
    for name, param in current_params.items():
        if name in ema_params:
            ema = ema_params[name]
            diff = param - ema
            diff_norm_sq += diff.pow(2).sum().item()
            ema_norm_sq += ema.pow(2).sum().item()
    
    if ema_norm_sq <= 0.0:
        return 0.0

    ema_norm = math.sqrt(ema_norm_sq)
    diff_norm = math.sqrt(diff_norm_sq)
    
    return float(diff_norm / ema_norm)


def calculate_plasticity(loss_before, loss_after, eps=1e-8):
    """
    새로운 태스크 학습 전/후 손실 감소를 기반으로 "가소성(plasticity)"을 계산.

    plasticity = clip( (loss_before - loss_after) / (loss_before + eps), 0, 1 )

    - 0 이하면 0
    - 1 이상이면 1
    """
    delta_loss = loss_before - loss_after
    relative_improvement = delta_loss / (loss_before + eps)
    plasticity_metric = max(0.0, min(1.0, relative_improvement))
    return plasticity_metric


class ContinualLearningMetrics:
    """연속 학습 관련 지표들을 관리하고 계산하는 클래스."""
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracies_max = [0.0] * num_tasks      # 각 태스크별 최고 성능
        self.accuracies_init = [0.0] * num_tasks     # 태스크 학습 직후 성능 R_{i,i}
        self.accuracies_current = [0.0] * num_tasks  # 현재 시점 성능 R_{T,i}
        self.fisher_history = []
        self.ema_params = None
        
    def update_accuracy(self, task_id, accuracy, is_init=False):
        """
        태스크별 정확도 업데이트.

        - task_id: 0-based index
        - is_init=True: 해당 태스크를 처음 학습 직후의 성능(R_{i,i})일 때
        """
        self.accuracies_current[task_id] = accuracy
        if is_init:
            self.accuracies_init[task_id] = accuracy
        if accuracy > self.accuracies_max[task_id]:
            self.accuracies_max[task_id] = accuracy
    
    def compute_all_metrics(self, model, current_task, dataloader=None, criterion=None):
        """
        모든 연속 학습 지표를 계산.

        - current_task: 지금까지 학습 완료된 태스크 개수 (1-based)
        - dataloader, criterion 이 주어지면 Fisher 기반 지표도 갱신.
        """
        metrics = {}
        metrics['AvgF'] = compute_avgf(
            self.accuracies_max, self.accuracies_current, current_task
        )
        metrics['BWT'] = compute_bwt(
            self.accuracies_current, self.accuracies_init, current_task
        )
        
        # Fisher 기반 지표 (옵션)
        if dataloader is not None and criterion is not None:
            current_fisher = compute_fisher_information(model, dataloader, criterion)
            self.fisher_history.append(current_fisher)
            if len(self.fisher_history) > 1:
                metrics['DeltaFisher'] = compute_delta_fisher(
                    self.fisher_history[-1], self.fisher_history[-2]
                )
        
        # EMA Drift
        current_params = {name: param.data for name, param in model.named_parameters()}
        self.ema_params = update_ema_parameters(self.ema_params, current_params)
        metrics['EMA_Drift'] = compute_ema_drift(self.ema_params, current_params)
        
        return metrics

###############################################################
# 2. 텍스트 생성 품질 관련 지표
###############################################################

def calculate_groundedness(prediction, context):
    """
    생성된 응답(prediction)이 주어진 문맥(context)에 얼마나 기반하는지 계산.

    단순히:
        groundedness = (# prediction 토큰 중 context에도 등장하는 토큰 수) / (# prediction 토큰 수)
    """
    pred_tokens = set(prediction.lower().split())
    context_tokens = set(context.lower().split())
    if not pred_tokens:
        return 0.0
    common_tokens = pred_tokens.intersection(context_tokens)
    return len(common_tokens) / len(pred_tokens)


def calculate_metrics(predictions, labels, losses, inputs):
    """
    다양한 텍스트 생성 품질 지표(ROUGE, BLEU, METEOR 등)를 계산.

    Args:
        predictions: List[str] - 생성된 텍스트
        labels:      List[str] - 정답 텍스트
        losses:      List[float] - (옵션) per-example loss (평균만 사용)
        inputs:      List[str] - 프롬프트/컨텍스트 텍스트 (groundedness 용)
    """
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    exact_matches = []
    groundedness_scores = []
    cosine_similarities = []
    jaccard_similarities = []

    # HuggingFace evaluate의 BLEU, METEOR 사용:contentReference[oaicite:4]{index=4}
    bleu_metric = evaluate.load('bleu')
    meteor_metric = evaluate.load('meteor')

    for pred, label, inp in zip(predictions, labels, inputs):
        pred_str = pred if isinstance(pred, str) else str(pred)
        label_str = label if isinstance(label, str) else str(label)
        inp_str = inp if isinstance(inp, str) else str(inp)

        # ROUGE (F-measure)
        r_score = rouge_scorer_obj.score(label_str, pred_str)
        for key in rouge_scores:
            rouge_scores[key].append(r_score[key].fmeasure)

        # Exact Match
        exact_matches.append(1 if pred_str.strip() == label_str.strip() else 0)

        # Groundedness
        groundedness_scores.append(calculate_groundedness(pred_str, inp_str))
        
        # Jaccard & Cosine Similarity
        pred_tokens = set(pred_str.lower().split())
        label_tokens = set(label_str.lower().split())
        union = pred_tokens.union(label_tokens)
        intersection = pred_tokens.intersection(label_tokens)
        jaccard_similarities.append(
            len(intersection) / len(union) if union else 0.0
        )
        
        if not pred_str.strip() or not label_str.strip():
            cosine_similarities.append(0.0)
        else:
            try:
                vectorizer = CountVectorizer().fit([pred_str.strip(), label_str.strip()])
                vectors = vectorizer.transform([pred_str.strip(), label_str.strip()])
                cosine_similarities.append(cosine_similarity(vectors)[0, 1])
            except ValueError:
                # 어휘가 비어서 CountVectorizer 에러나는 경우 등
                cosine_similarities.append(0.0)

    # BLEU / METEOR 는 전체 샘플에 대해 한 번만 compute
    references_for_eval = [[lbl] for lbl in labels]
    bleu_results = bleu_metric.compute(
        predictions=predictions, references=references_for_eval
    )
    meteor_results = meteor_metric.compute(
        predictions=predictions, references=references_for_eval
    )
    
    # loss 리스트가 비어있으면 0.0 으로 처리
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    results = {
        "loss": avg_loss,
        "rouge1": float(np.mean(rouge_scores['rouge1'])) if rouge_scores['rouge1'] else 0.0,
        "rouge2": float(np.mean(rouge_scores['rouge2'])) if rouge_scores['rouge2'] else 0.0,
        "rougeL": float(np.mean(rouge_scores['rougeL'])) if rouge_scores['rougeL'] else 0.0,
        "bleu": float(bleu_results.get('bleu', 0.0)),
        "meteor": float(meteor_results.get('meteor', 0.0)),
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "groundedness": float(np.mean(groundedness_scores)) if groundedness_scores else 0.0,
        "cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0.0,
        "jaccard_similarity": float(np.mean(jaccard_similarities)) if jaccard_similarities else 0.0,
    }
    return results
