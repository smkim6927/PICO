# eval/modules/metrics.py

from collections import Counter

import evaluate
import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEXT_METRICS = [
    "bleu",
    "rouge1",
    "rouge2",
    "rougeL",
    "meteor",
    "cosine_similarity",
    "token_precision",
    "token_recall",
    "token_f1",
]


_BLEU = None
_METEOR = None
_ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def _mean_finite(values):
    values = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]
    return float(np.mean(values)) if values else float("nan")


def _validate_R_matrix(R_matrix, current_task):
    R = np.asarray(R_matrix, dtype=float)

    if R.ndim != 2:
        raise ValueError("R_matrix must be a 2D array.")

    if current_task < 0:
        raise ValueError("current_task must be non-negative.")

    if R.shape[0] <= current_task:
        raise ValueError(
            f"R_matrix requires at least {current_task + 1} rows "
            f"(base + {current_task} task stages), got {R.shape[0]}."
        )

    if R.shape[1] < current_task:
        raise ValueError(
            f"R_matrix requires at least {current_task} task columns, "
            f"got {R.shape[1]}."
        )

    return R


def compute_fwt(R_matrix, current_task):
    """
    FWT = 1/(T-1) sum_{i=2}^{T} (R_{i-1,i} - R_{0,i})

    R[0]       : base-model evaluation
    R[i]       : evaluation after learning task i
    current_task = T
    """
    if current_task <= 1:
        return 0.0

    R = _validate_R_matrix(R_matrix, current_task)

    values = []

    for task_idx in range(1, current_task):
        before_learning = R[task_idx, task_idx]
        base_value = R[0, task_idx]

        if np.isfinite(before_learning) and np.isfinite(base_value):
            values.append(before_learning - base_value)

    return _mean_finite(values)


def compute_bwt(R_matrix, current_task):
    """
    BWT = 1/(T-1) sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
    """
    if current_task <= 1:
        return 0.0

    R = _validate_R_matrix(R_matrix, current_task)

    values = []

    for task_idx in range(current_task - 1):
        learned_step = task_idx + 1

        final_value = R[current_task, task_idx]
        after_learning = R[learned_step, task_idx]

        if np.isfinite(final_value) and np.isfinite(after_learning):
            values.append(final_value - after_learning)

    return _mean_finite(values)


def compute_avgf(R_matrix, current_task):
    """
    AvgF = 1/(T-1) sum_{i=1}^{T-1}
           (max_{t=i,...,T-1} R_{t,i} - R_{T,i})
    """
    if current_task <= 1:
        return 0.0

    R = _validate_R_matrix(R_matrix, current_task)

    values = []

    for task_idx in range(current_task - 1):
        learned_step = task_idx + 1
        final_value = R[current_task, task_idx]

        if not np.isfinite(final_value):
            continue

        past_values = R[
            learned_step:current_task,
            task_idx,
        ]

        past_values = past_values[np.isfinite(past_values)]

        if past_values.size == 0:
            continue

        peak = float(np.max(past_values))
        values.append(peak - final_value)

    return _mean_finite(values)


def compute_wcr(R_matrix, current_task):
    """
    WCR = max_i (1 - R_{T,i} / R_i^peak)

    R_i^peak = max_{t=i,...,T} R_{t,i}

    Higher-is-better metrics only.
    Lower WCR is better.
    """
    if current_task <= 0:
        return 0.0

    R = _validate_R_matrix(R_matrix, current_task)

    values = []

    for task_idx in range(current_task):
        learned_step = task_idx + 1
        final_value = R[current_task, task_idx]

        if not np.isfinite(final_value):
            continue

        trajectory = R[
            learned_step:current_task + 1,
            task_idx,
        ]

        trajectory = trajectory[np.isfinite(trajectory)]

        if trajectory.size == 0:
            continue

        peak = float(np.max(trajectory))

        if peak <= 0.0:
            continue

        values.append(
            1.0 - (final_value / peak)
        )

    return float(max(values)) if values else float("nan")


def compute_cl_metrics(R_matrix, current_task):
    return {
        "FWT": compute_fwt(R_matrix, current_task),
        "BWT": compute_bwt(R_matrix, current_task),
        "AvgF": compute_avgf(R_matrix, current_task),
        "WCR": compute_wcr(R_matrix, current_task),
    }


def _token_overlap_metrics(prediction, reference):
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0

    if not pred_tokens:
        return 0.0, 0.0, 0.0

    if not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    overlap = sum(
        (pred_counter & ref_counter).values()
    )

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1


def _cosine_similarity(prediction, reference):
    if not prediction.strip() or not reference.strip():
        return 0.0

    try:
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(
            [prediction, reference]
        )
        return float(
            cosine_similarity(vectors[0], vectors[1])[0, 0]
        )
    except ValueError:
        return 0.0


def calculate_metrics(predictions, labels):
    global _BLEU, _METEOR

    if len(predictions) != len(labels):
        raise ValueError(
            "predictions and labels must have the same length."
        )

    if len(predictions) == 0:
        return {
            metric: 0.0
            for metric in TEXT_METRICS
        }

    predictions = [
        p if isinstance(p, str) else str(p)
        for p in predictions
    ]
    labels = [
        l if isinstance(l, str) else str(l)
        for l in labels
    ]

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    cosine_scores = []

    token_precision_scores = []
    token_recall_scores = []
    token_f1_scores = []

    for prediction, label in zip(predictions, labels):
        rouge = _ROUGE.score(
            label,
            prediction,
        )

        rouge1_scores.append(
            rouge["rouge1"].fmeasure
        )
        rouge2_scores.append(
            rouge["rouge2"].fmeasure
        )
        rougeL_scores.append(
            rouge["rougeL"].fmeasure
        )

        cosine_scores.append(
            _cosine_similarity(
                prediction,
                label,
            )
        )

        precision, recall, f1 = _token_overlap_metrics(
            prediction,
            label,
        )

        token_precision_scores.append(precision)
        token_recall_scores.append(recall)
        token_f1_scores.append(f1)

    if _BLEU is None:
        _BLEU = evaluate.load("bleu")

    if _METEOR is None:
        _METEOR = evaluate.load("meteor")

    bleu_result = _BLEU.compute(
        predictions=predictions,
        references=[
            [label]
            for label in labels
        ],
    )

    meteor_result = _METEOR.compute(
        predictions=predictions,
        references=labels,
    )

    return {
        "bleu": float(
            bleu_result.get("bleu", 0.0)
        ),
        "rouge1": float(
            np.mean(rouge1_scores)
        ),
        "rouge2": float(
            np.mean(rouge2_scores)
        ),
        "rougeL": float(
            np.mean(rougeL_scores)
        ),
        "meteor": float(
            meteor_result.get("meteor", 0.0)
        ),
        "cosine_similarity": float(
            np.mean(cosine_scores)
        ),
        "token_precision": float(
            np.mean(token_precision_scores)
        ),
        "token_recall": float(
            np.mean(token_recall_scores)
        ),
        "token_f1": float(
            np.mean(token_f1_scores)
        ),
    }
