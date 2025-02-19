import numpy as np
from sklearn.metrics import f1_score, r2_score
from rouge_score import rouge_scorer
from nltk.translate import bleu_score, meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_plasticity(self, loss_before, loss_after, eps=1e-8):
        delta_loss = loss_before - loss_after
        relative_improvement = delta_loss / (loss_before + eps)
        plasticity_metric = max(0.0, min(1.0, relative_improvement))
        return plasticity_metric
        
def calculate_groundedness(prediction, context):
    """prediction과 context 간에 겹치는 토큰 비율을 계산하는 예시 메트릭."""
    pred_tokens = set(prediction.lower().split())
    context_tokens = set(context.lower().split())
    common_tokens = pred_tokens.intersection(context_tokens)
    return len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0


def calculate_metrics(predictions, labels, losses, inputs):
    """
    predictions: list of str (디코딩된 예측값)
    labels: list of str (디코딩된 정답)
    losses: list of float
    inputs: list of str (모델에 들어간 prompt 등)
    
    LlamaIndex 관련 평가는 전부 제거했습니다.
    대신 전통적인 텍스트 평가 지표(ROUGE, BLEU, METEOR, F1, R^2)와
    추가적인 예시(Cosine Similarity, Jaccard Similarity, Groundedness)를 제공합니다.
    """

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    bleu_scores = []
    meteor_scores = []
    exact_matches = []
    groundedness_scores = []
    cosine_similarities = []
    jaccard_similarities = []

    for pred, label, inp in zip(predictions, labels, inputs):
        # ------------------------
        # 1) ROUGE
        # ------------------------
        r_score = rouge.score(label, pred)
        rouge_scores['rouge1'].append(r_score['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(r_score['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(r_score['rougeL'].fmeasure)

        # ------------------------
        # 2) BLEU (with smoothing)
        # ------------------------
        
        smoothing_function = SmoothingFunction().method1
        bleu = bleu_score.sentence_bleu([label.split()], pred.split(), smoothing_function=smoothing_function)
        bleu_scores.append(bleu)

        # ------------------------
        # 3) METEOR (list[str] 형태로!)
        # ------------------------
        meteor_val = meteor_score.single_meteor_score(label.split(), pred.split())
        meteor_scores.append(meteor_val)

        # ------------------------
        # 4) Exact Match
        # ------------------------
        exact_matches.append(1 if pred.strip() == label.strip() else 0)

        # ------------------------
        # 5) Groundedness (예시)
        # ------------------------
        groundedness_scores.append(calculate_groundedness(pred, inp))

        # ------------------------
        # 6) Cosine Similarity (BoW 기반), with empty check
        # ------------------------
        pred_clean = pred.strip()
        label_clean = label.strip()
        if not pred_clean or not label_clean:
            # 둘 중 하나가 비어 있으면 유사도 0.0
            cos_val = 0.0
        else:
            try:
                vectorizer = CountVectorizer(stop_words=None)
                vectors = vectorizer.fit_transform([pred_clean, label_clean])
                arr = vectors.toarray()  # shape: (2, vocab_size)
                cos_mat = cosine_similarity(arr)  # shape: (2, 2)
                cos_val = cos_mat[0, 1]
            except ValueError:
                # empty vocabulary 등으로 에러 발생 시 0.0 처리
                cos_val = 0.0
        cosine_similarities.append(cos_val)

        # ------------------------
        # 7) Jaccard Similarity
        # ------------------------
        pred_tokens = set(pred.lower().split())
        label_tokens = set(label.lower().split())
        union = pred_tokens.union(label_tokens)
        intersection = pred_tokens.intersection(label_tokens)
        if len(union) == 0:
            jaccard = 0.0
        else:
            jaccard = len(intersection) / len(union)
        jaccard_similarities.append(jaccard)

    # ------------------------
    # 평균(또는 집계) 계산
    # ------------------------
    avg_rouge1 = np.mean(rouge_scores['rouge1'])
    avg_rouge2 = np.mean(rouge_scores['rouge2'])
    avg_rougeL = np.mean(rouge_scores['rougeL'])
    avg_bleu = np.mean(bleu_scores)
    avg_meteor = np.mean(meteor_scores)
    exact_match_rate = np.mean(exact_matches)
    avg_groundedness = np.mean(groundedness_scores)
    avg_loss = np.mean(losses)

    # F1 (문자열 전체를 라벨로 간주하여 "weighted" F1. 실제 토큰 단위와 다를 수 있음)
    f1 = f1_score(
        [lbl.strip() for lbl in labels],
        [prd.strip() for prd in predictions],
        average="weighted",
        zero_division=0,
    )

    # R^2 (텍스트 길이로 단순 수치화한 뒤 비교)
    true_lengths = [len(lbl) for lbl in labels]
    predicted_lengths = [len(prd) for prd in predictions]
    r2 = r2_score(true_lengths, predicted_lengths)

    # 추가 메트릭 평균
    avg_cosine_sim = np.mean(cosine_similarities)
    avg_jaccard_sim = np.mean(jaccard_similarities)

    return {
        "loss": avg_loss,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "bleu": avg_bleu,
        "meteor": avg_meteor,
        "f1": f1,
        "r2": r2,
        "exact_match": exact_match_rate,
        "groundedness": avg_groundedness,
        "cosine_similarity": avg_cosine_sim,
        "jaccard_similarity": avg_jaccard_sim,
    }
