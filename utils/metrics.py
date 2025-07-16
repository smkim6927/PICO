import torch
import numpy as np
from sklearn.metrics import f1_score, r2_score
from rouge_score import rouge_scorer
from nltk.translate import bleu_score, meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###############################################################
# 1. 연속 학습(Continual Learning) 관련 지표
###############################################################

def compute_avgf(accuracies_max, accuracies_final, current_task):
    """Average Forgetting (AvgF)을 계산합니다."""
    if current_task <= 1:
        return 0.0
    
    forgetting_sum = sum(accuracies_max[i] - accuracies_final[i] 
                        for i in range(current_task - 1))
    return forgetting_sum / (current_task - 1)

def compute_bwt(accuracies_final, accuracies_init, current_task):
    """Backward Transfer (BWT)를 계산합니다."""
    if current_task <= 1:
        return 0.0
    
    transfer_sum = sum(accuracies_final[i] - accuracies_init[i] 
                      for i in range(current_task - 1))
    return transfer_sum / (current_task - 1)

def compute_fisher_information(model, dataloader, criterion):
    """Fisher Information Matrix를 계산합니다."""
    fisher_info = {}
    model.eval()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_info[name] = torch.zeros_like(param)
    
    for data, target in dataloader:
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_info[name] += param.grad.data ** 2
    
    for name in fisher_info:
        fisher_info[name] /= len(dataloader)
    
    return fisher_info

def compute_delta_fisher(fisher_t, fisher_t_minus_1):
    """Fisher Information의 변화량(L1 norm)을 계산합니다."""
    delta = 0.0
    for name in fisher_t:
        if name in fisher_t_minus_1:
            delta += torch.sum(torch.abs(fisher_t[name] - fisher_t_minus_1[name])).item()
    return delta

def update_ema_parameters(ema_params, current_params, alpha=0.999):
    """Exponential Moving Average(EMA) 파라미터를 업데이트합니다."""
    if ema_params is None:
        return {name: param.clone() for name, param in current_params.items()}
    
    updated_ema = {}
    for name, param in current_params.items():
        if name in ema_params:
            updated_ema[name] = alpha * ema_params[name] + (1 - alpha) * param
        else:
            updated_ema[name] = param.clone()
    
    return updated_ema

def compute_ema_drift(ema_params, current_params):
    """현재 파라미터와 EMA 파라미터 간의 Drift를 계산합니다."""
    ema_norm = 0.0
    diff_norm = 0.0
    
    for name, param in current_params.items():
        if name in ema_params:
            param_diff = param - ema_params[name]
            diff_norm += torch.sum(param_diff ** 2).item()
            ema_norm += torch.sum(param ** 2).item()
    
    ema_norm = torch.sqrt(torch.tensor(ema_norm))
    diff_norm = torch.sqrt(torch.tensor(diff_norm))
    
    return (diff_norm / ema_norm).item() if ema_norm > 0 else 0.0

def calculate_plasticity(loss_before, loss_after, eps=1e-8):
    """
    새로운 태스크 학습 후 손실 감소를 기반으로 가소성을 계산합니다.
    결과는 0과 1 사이의 값으로 정규화됩니다.
    """
    delta_loss = loss_before - loss_after
    relative_improvement = delta_loss / (loss_before + eps)
    plasticity_metric = max(0.0, min(1.0, relative_improvement))
    return plasticity_metric

class ContinualLearningMetrics:
    """연속 학습 관련 지표들을 관리하고 계산하는 클래스."""
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracies_max = [0.0] * num_tasks
        self.accuracies_init = [0.0] * num_tasks
        self.accuracies_current = [0.0] * num_tasks
        self.fisher_history = []
        self.ema_params = None
        
    def update_accuracy(self, task_id, accuracy, is_init=False):
        """태스크별 정확도를 업데이트합니다."""
        self.accuracies_current[task_id] = accuracy
        if is_init:
            self.accuracies_init[task_id] = accuracy
        if accuracy > self.accuracies_max[task_id]:
            self.accuracies_max[task_id] = accuracy
    
    def compute_all_metrics(self, model, current_task, dataloader=None, criterion=None):
        """모든 연속 학습 지표를 계산합니다."""
        metrics = {}
        metrics['AvgF'] = compute_avgf(self.accuracies_max, self.accuracies_current, current_task)
        metrics['BWT'] = compute_bwt(self.accuracies_current, self.accuracies_init, current_task)
        
        if dataloader and criterion:
            current_fisher = compute_fisher_information(model, dataloader, criterion)
            self.fisher_history.append(current_fisher)
            if len(self.fisher_history) > 1:
                metrics['DeltaFisher'] = compute_delta_fisher(self.fisher_history[-1], self.fisher_history[-2])
        
        current_params = {name: param.data for name, param in model.named_parameters()}
        self.ema_params = update_ema_parameters(self.ema_params, current_params)
        metrics['EMA_Drift'] = compute_ema_drift(self.ema_params, current_params)
        
        return metrics

###############################################################
# 2. 텍스트 생성 품질 관련 지표
###############################################################

def calculate_groundedness(prediction, context):
    """생성된 응답(prediction)이 주어진 문맥(context)에 얼마나 기반하는지 계산합니다."""
    pred_tokens = set(prediction.lower().split())
    context_tokens = set(context.lower().split())
    if not pred_tokens:
        return 0.0
    common_tokens = pred_tokens.intersection(context_tokens)
    return len(common_tokens) / len(pred_tokens)

def calculate_metrics(predictions, labels, losses, inputs):
    """
    다양한 텍스트 생성 품질 지표(ROUGE, BLEU, METEOR 등)를 계산합니다.
    """
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bleu_scores, meteor_scores, exact_matches, groundedness_scores = [], [], [], []
    cosine_similarities, jaccard_similarities = [], []

    for pred, label, inp in zip(predictions, labels, inputs):
        # ROUGE
        r_score = rouge_scorer_obj.score(label, pred)
        for key in rouge_scores:
            rouge_scores[key].append(r_score[key].fmeasure)

        # BLEU (with smoothing)
        bleu = bleu_score.sentence_bleu([label.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu)

        # METEOR
        meteor_val = meteor_score.single_meteor_score(label.split(), pred.split())
        meteor_scores.append(meteor_val)

        # Exact Match
        exact_matches.append(1 if pred.strip() == label.strip() else 0)

        # Groundedness
        groundedness_scores.append(calculate_groundedness(pred, inp))

        # Cosine & Jaccard Similarity
        pred_tokens = set(pred.lower().split())
        label_tokens = set(label.lower().split())
        
        # Jaccard
        union = pred_tokens.union(label_tokens)
        intersection = pred_tokens.intersection(label_tokens)
        jaccard_similarities.append(len(intersection) / len(union) if union else 0.0)

        # Cosine
        if not pred.strip() or not label.strip():
            cosine_similarities.append(0.0)
        else:
            try:
                vectorizer = CountVectorizer().fit([pred.strip(), label.strip()])
                vectors = vectorizer.transform([pred.strip(), label.strip()])
                cosine_similarities.append(cosine_similarity(vectors)[0, 1])
            except ValueError:
                cosine_similarities.append(0.0)
    
    # 평균 및 집계 계산
    results = {
        "loss": np.mean(losses),
        "rouge1": np.mean(rouge_scores['rouge1']),
        "rouge2": np.mean(rouge_scores['rouge2']),
        "rougeL": np.mean(rouge_scores['rougeL']),
        "bleu": np.mean(bleu_scores),
        "meteor": np.mean(meteor_scores),
        "exact_match": np.mean(exact_matches),
        "groundedness": np.mean(groundedness_scores),
        "cosine_similarity": np.mean(cosine_similarities),
        "jaccard_similarity": np.mean(jaccard_similarities),
        "f1": f1_score([lbl.strip() for lbl in labels], [prd.strip() for prd in predictions], average="weighted", zero_division=0),
        "r2": r2_score([len(lbl) for lbl in labels], [len(prd) for prd in predictions])
    }
    return results
