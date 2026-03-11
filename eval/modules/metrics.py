import math
import torch
import numpy as np
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1) Metrics related to Continual Learning

def compute_avgf(accuracies_max, accuracies_final, current_task):
    """
    Average Forgetting (AvgF)

    - accuracies_max[i]   
      The maximum performance observed so far for Task i (max_t R_{t,i})
    - accuracies_final[i]
      Performance of task i at the current point in time (current_task) (R_{T,i})
    - current_task
      Number of tasks completed so far (1-based, T)

    F_T = (1/(T-1)) * sum_{i=1..T-1} ( max_{t<=T} R_{t,i} - R_{T,i} ) implemented as an accuracy version.
    """
    if current_task <= 1:
        return 0.0
    
    forgetting_sum = 0.0
    # Consider the task index to be 0-based: 0 .. current_task-2
    for i in range(current_task - 1):
        forgetting_sum += (accuracies_max[i] - accuracies_final[i])
    return forgetting_sum / (current_task - 1)


def compute_bwt(accuracies_final, accuracies_init, current_task):
    """
    Backward Transfer (BWT)

    - accuracies_init[i]
      task i's performance 'immediately after initial training' (e.g., R_{i,i})
    - accuracies_final[i]
      Performance of task i at the current point in time (current_task) (e.g., R_{T,i})
    - current_task
      Number of tasks completed so far (1-based, T)

    Lopez-Paz & Ranzato (2017):
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
    Function for estimating Fisher information.

    - Empirical Fisher: E[ (∂ log p(y|x) / ∂θ)^2 ]. Approximated by averaging the squares of the gradients at each batch unit.
    - in each batch loss = CrossEntropyLoss(ignore_index=-100, reduction='mean')
    - When the gradient with respect to it is denoted as g_batch,
      Fisher ≈ (1 / #batches) * sum_batch g_batch^2
    """
    # Fisher tensor init
    fisher_information = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    model.eval()
    device = next(model.parameters()).device

    num_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        labels = batch["labels"]

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
    Calculate the Fisher Information change (L1 norm).
    Sum of the L1 norm of the Fisher tensor difference at two time points t and t-1.
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
    Exponential Moving Average(EMA) Parameter Update.

    ema_new = alpha * ema_old + (1 - alpha) * current
    """
    if ema_params is None:
        # If it is the first step, use the current parameter as the EMA.
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
    Calculate the drift between the current parameter and the EMA parameter.

    drift = ||θ_t - θ_ema||_2 / ||θ_ema||_2

    - The denominator is normalized by the L2 norm of the EMA parameters.
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
    Calculate "plasticity" based on the reduction in pre- and post-learning loss for new tasks.

    plasticity = clip( (loss_before - loss_after) / (loss_before + eps), 0, 1 )

    """
    delta_loss = loss_before - loss_after
    relative_improvement = delta_loss / (loss_before + eps)
    plasticity_metric = max(0.0, min(1.0, relative_improvement))
    return plasticity_metric


class ContinualLearningMetrics:
    """A class for managing and calculating metrics related to continuous learning."""
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracies_max = [0.0] * num_tasks      # Peak performance for each task
        self.accuracies_init = [0.0] * num_tasks     # Performance immediately after task learning R_{i,i}
        self.accuracies_current = [0.0] * num_tasks  # Current performance R_{T,i}
        self.fisher_history = []
        self.ema_params = None
        
    def update_accuracy(self, task_id, accuracy, is_init=False):
        """
        Task-specific accuracy update.

        - task_id: 0-based index
        - is_init=True: When the task's performance is immediately after initial training (R_{i,i})
        """
        self.accuracies_current[task_id] = accuracy
        if is_init:
            self.accuracies_init[task_id] = accuracy
        if accuracy > self.accuracies_max[task_id]:
            self.accuracies_max[task_id] = accuracy
    
    def compute_all_metrics(self, model, current_task, dataloader=None, criterion=None):
        """
        Calculate all continuous learning metrics.

        - current_task: Number of tasks completed so far (1-based)
        - When dataloader and criterion are provided, also update Fisher-based metrics.
        """
        metrics = {}
        metrics['AvgF'] = compute_avgf(
            self.accuracies_max, self.accuracies_current, current_task
        )
        metrics['BWT'] = compute_bwt(
            self.accuracies_current, self.accuracies_init, current_task
        )
        
        # Fisher based metric
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

# 2) Text Generation Quality Metrics

def calculate_groundedness(prediction, context):
    """
    Calculate how grounded the generated response (prediction) is in the given context.

    Simply:
        groundedness = (# of tokens in the prediction that also appear in the context) / (# of tokens in the prediction)
    """
    pred_tokens = set(prediction.lower().split())
    context_tokens = set(context.lower().split())
    if not pred_tokens:
        return 0.0
    common_tokens = pred_tokens.intersection(context_tokens)
    return len(common_tokens) / len(pred_tokens)


def calculate_metrics(predictions, labels, losses, inputs):
    """
    Calculate various text generation quality metrics (ROUGE, BLEU, METEOR, etc.).

    Args:
        predictions: List[str] - Generated text
        labels:      List[str] - Ground truth text
        losses:      List[float] - (Optional) Per-example loss (Only the average is used)
        inputs:      List[str] - Prompt/context text (For groundedness)
    """
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    exact_matches = []
    groundedness_scores = []
    cosine_similarities = []
    jaccard_similarities = []

    # Using BLEU and METEOR in HuggingFace evaluate Library
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
                # When a CountVectorizer error occurs due to empty vocabulary, etc.
                cosine_similarities.append(0.0)

    # BLEU / METEOR is computed only once for the entire sample.
    references_for_eval = [[lbl] for lbl in labels]
    bleu_results = bleu_metric.compute(
        predictions=predictions, references=references_for_eval
    )
    meteor_results = meteor_metric.compute(
        predictions=predictions, references=references_for_eval
    )
    
    # If the loss list is empty, treat it as 0.0.
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
