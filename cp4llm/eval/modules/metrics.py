# eval/modules/metrics.py

"""Multilingual text metrics for English, Spanish, and Korean.

Design goals
------------
1. Keep the public API compatible with EvalRunner::
       calculate_metrics(predictions, labels, meteor_mode=...)
2. Use one deterministic Unicode-aware lexical policy for EN/ES/KO.
3. Compute corpus BLEU with SacreBLEU using pre-tokenized text, exponential
   smoothing, and effective order for short-answer robustness.
4. Compute corpus chrF2 directly on normalized Unicode text. chrF is especially
   useful for Korean because it does not depend on whitespace/morphological
   segmentation.
5. Keep all returned metrics on the 0..1 scale used by the existing CL code.

Dependency
----------
    pip install sacrebleu rouge-score nltk numpy

Important BLEU note
-------------------
A returned BLEU value of 0.03 means about 3 BLEU points on SacreBLEU's usual
0..100 reporting scale. Low BLEU can be legitimate for short/free-form answers;
interpret it together with chrF rather than trying to inflate BLEU.
"""

from __future__ import annotations

import math
import re
import unicodedata
from importlib import metadata as importlib_metadata
from collections import Counter
from typing import Any, List, Sequence

import numpy as np
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError as exc:  # fail early with an actionable message
    raise ImportError(
        "metrics.py requires sacrebleu. Install it with `pip install sacrebleu`."
    ) from exc


# Lexical tokens for ROUGE/METEOR/cosine. Unicode letters/numbers are retained;
# punctuation is excluded. Apostrophes inside a token are kept for English and
# Spanish contractions/possessives where applicable.
_UNICODE_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)

# BLEU tokenizer: keep Unicode word/number runs and keep every remaining
# non-whitespace symbol/punctuation mark as its own token. This gives EN/ES/KO
# one shared tokenizer and avoids checkpoint/model-tokenizer dependence.
_UNICODE_BLEU_RE = re.compile(
    r"[^\W_]+(?:['’][^\W_]+)*|[^\s]",
    flags=re.UNICODE,
)

# Fallback for symbol-only lexical inputs.
_NONSPACE_CHAR_RE = re.compile(r"\S", flags=re.UNICODE)


class _UnicodeWordTokenizer:
    """Tokenizer adapter for rouge_score.RougeScorer."""

    def tokenize(self, text: str) -> List[str]:
        return tokenize_unicode_words(text)


class _IdentityStemmer:
    """Language-neutral no-op stemmer for lexical METEOR."""

    def stem(self, word: str) -> str:
        return word


class _NoWordNet:
    """Disable English WordNet synonym expansion in multilingual METEOR."""

    @staticmethod
    def synsets(_word: str) -> list:
        return []


_IDENTITY_STEMMER = _IdentityStemmer()
_NO_WORDNET = _NoWordNet()

# SacreBLEU sees already-tokenized strings, so its own tokenizer must be off.
# effective_order=True is important for corpora consisting of very short answers;
# smooth_method="exp" is SacreBLEU's standard smoothing policy.
_BLEU = BLEU(
    tokenize="none",
    smooth_method="exp",
    effective_order=True,
    lowercase=False,
)

# chrF2: character n-grams only (word_order=0). This deliberately avoids making
# Korean depend on a morphological analyzer while remaining equally applicable
# to English and Spanish. Scores from SacreBLEU are 0..100 and normalized below.
_CHRF = CHRF(
    char_order=6,
    word_order=0,
    beta=2,
    lowercase=False,
    whitespace=False,
    eps_smoothing=False,
)


def _package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def get_metric_policy() -> dict[str, Any]:
    """Return a JSON-serializable description of the metric configuration.

    SacreBLEU's built-in signature cannot describe our external Unicode BLEU
    tokenizer because SacreBLEU itself is intentionally run with ``tokenize=none``.
    Persist this policy beside CL results so paper numbers remain auditable.
    """
    return {
        "scale": "0..1",
        "normalization": "Unicode NFKC",
        "lexical_tokenizer": {
            "name": "unicode_words_v1",
            "casefold": True,
            "pattern": _UNICODE_WORD_RE.pattern,
            "symbol_only_fallback": True,
        },
        "bleu": {
            "implementation": "sacrebleu.metrics.BLEU",
            "external_tokenizer": "unicode_word_or_symbol_v1",
            "external_tokenizer_pattern": _UNICODE_BLEU_RE.pattern,
            "case_sensitive": True,
            "sacrebleu_tokenize": "none",
            "smooth_method": "exp",
            "effective_order": True,
            "max_ngram_order": 4,
        },
        "chrf": {
            "implementation": "sacrebleu.metrics.CHRF",
            "char_order": 6,
            "word_order": 0,
            "beta": 2,
            "case_sensitive": True,
            "whitespace": False,
            "eps_smoothing": False,
        },
        "rouge": {
            "implementation": "rouge_score.RougeScorer",
            "variants": ["rouge1", "rouge2", "rougeL"],
            "use_stemmer": False,
            "tokenizer": "unicode_words_v1",
        },
        "meteor": {
            "recommended_mode": "multilingual_lexical",
            "multilingual_lexical_stemmer": "identity",
            "multilingual_lexical_wordnet": "disabled",
        },
        "cosine_similarity": "bag_of_unicode_lexical_token_counts",
        "package_versions": {
            "sacrebleu": _package_version("sacrebleu"),
            "rouge-score": _package_version("rouge-score"),
            "nltk": _package_version("nltk"),
            "numpy": _package_version("numpy"),
        },
    }


def _to_text(value: Any) -> str:
    """Convert a value to text without converting None to literal 'None'."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def normalize_unicode(text: Any, *, casefold: bool = False) -> str:
    """NFKC-normalize text; optionally apply Unicode-aware case folding."""
    value = unicodedata.normalize("NFKC", _to_text(text))
    return value.casefold() if casefold else value


def tokenize_unicode_words(text: Any) -> List[str]:
    """Language-neutral lexical tokenizer for English, Spanish, and Korean.

    The tokenizer is intentionally *not* a Korean morphological analyzer. Korean
    eojeol variation is therefore complemented by chrF, which operates directly
    on character n-grams.
    """
    normalized = normalize_unicode(text, casefold=True)
    lexical_tokens = _UNICODE_WORD_RE.findall(normalized)
    if lexical_tokens:
        return lexical_tokens
    return _NONSPACE_CHAR_RE.findall(normalized)


def tokenize_bleu_shared(text: Any) -> List[str]:
    """Shared BLEU tokenizer for EN/ES/KO.

    Case is preserved to stay close to standard case-sensitive BLEU. NFKC is
    applied so compatibility variants do not create accidental mismatches.
    """
    normalized = normalize_unicode(text, casefold=False)
    return _UNICODE_BLEU_RE.findall(normalized)


def _strict_mean(values: Sequence[Any], *, name: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute {name}: no values were provided.")

    converted: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Cannot compute {name}: non-numeric value {value!r}."
            ) from exc
        if not math.isfinite(numeric):
            raise RuntimeError(
                f"Cannot compute {name}: non-finite value {numeric!r}."
            )
        converted.append(numeric)
    return float(np.mean(converted))

def _cosine_similarity_from_tokens(
    tokens_a: Sequence[str], tokens_b: Sequence[str]
) -> float:
    if not tokens_a or not tokens_b:
        return 0.0

    count_a = Counter(tokens_a)
    count_b = Counter(tokens_b)
    common = count_a.keys() & count_b.keys()

    dot = sum(count_a[token] * count_b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in count_a.values()))
    norm_b = math.sqrt(sum(value * value for value in count_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _compute_multilingual_lexical_meteor(
    predictions: Sequence[str], labels: Sequence[str]
) -> float:
    """Lexical METEOR without English-specific stemming/WordNet expansion."""
    scores: List[float] = []
    for prediction, label in zip(predictions, labels):
        pred_tokens = tokenize_unicode_words(prediction)
        label_tokens = tokenize_unicode_words(label)

        if not pred_tokens or not label_tokens:
            scores.append(0.0)
            continue

        score = single_meteor_score(
            reference=label_tokens,
            hypothesis=pred_tokens,
            stemmer=_IDENTITY_STEMMER,
            wordnet=_NO_WORDNET,
        )
        scores.append(float(score))

    return _strict_mean(scores, name="meteor")


def _compute_bleu(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """Corpus BLEU for EN/ES/KO, returned on a 0..1 scale.

    We pre-tokenize with a fixed Unicode tokenizer and disable SacreBLEU's own
    tokenizer. This makes BLEU independent of the model/checkpoint tokenizer and
    uses the same policy across all three target languages.
    """
    tokenized_predictions = [
        " ".join(tokenize_bleu_shared(prediction)) for prediction in predictions
    ]
    tokenized_labels = [
        " ".join(tokenize_bleu_shared(label)) for label in labels
    ]

    empty_refs = [i for i, ref in enumerate(tokenized_labels) if not ref.strip()]
    if empty_refs:
        raise ValueError(
            "BLEU received empty/whitespace-only references after normalization; "
            f"indices={empty_refs[:8]}."
        )

    if not any(pred.strip() for pred in tokenized_predictions):
        return 0.0

    score_100 = float(
        _BLEU.corpus_score(tokenized_predictions, [tokenized_labels]).score
    )
    score = score_100 / 100.0
    if not math.isfinite(score):
        raise RuntimeError(f"BLEU is non-finite: {score!r}.")
    return score


def _compute_chrf(predictions: Sequence[str], labels: Sequence[str]) -> float:
    """Corpus chrF2 (character n-gram F-score), returned on a 0..1 scale."""
    normalized_predictions = [
        normalize_unicode(prediction, casefold=False) for prediction in predictions
    ]
    normalized_labels = [
        normalize_unicode(label, casefold=False) for label in labels
    ]

    empty_refs = [i for i, ref in enumerate(normalized_labels) if not ref.strip()]
    if empty_refs:
        raise ValueError(
            "chrF received empty/whitespace-only references; "
            f"indices={empty_refs[:8]}."
        )

    if not any(pred.strip() for pred in normalized_predictions):
        return 0.0

    score_100 = float(
        _CHRF.corpus_score(normalized_predictions, [normalized_labels]).score
    )
    score = score_100 / 100.0
    if not math.isfinite(score):
        raise RuntimeError(f"chrF is non-finite: {score!r}.")
    return score


def _compute_standard_meteor(
    predictions: Sequence[str], labels: Sequence[str]
) -> float:
    """NLTK METEOR with default language resources.

    This mode is retained only for backward compatibility. For EN/ES/KO joint
    evaluation, ``multilingual_lexical`` is the recommended mode because the
    standard stemmer/WordNet behavior is English-centric.
    """
    scores: List[float] = []
    for prediction, label in zip(predictions, labels):
        pred_tokens = tokenize_unicode_words(prediction)
        ref_tokens = tokenize_unicode_words(label)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        # Using NLTK's default resources may require WordNet data in the runtime.
        scores.append(float(single_meteor_score(ref_tokens, pred_tokens)))
    return _strict_mean(scores, name="meteor")


def calculate_metrics(
    predictions: Sequence[Any],
    labels: Sequence[Any],
    *,
    meteor_mode: str = "multilingual_lexical",
) -> dict[str, float]:
    """Compute text metrics for English, Spanish, and Korean.

    Returns ``rouge1``, ``rouge2``, ``rougeL``, ``bleu``, ``chrf``, ``meteor``,
    and ``cosine_similarity``. Every metric is normalized to the 0..1 scale.

    Recommended interpretation:
      * BLEU: exact token n-gram correspondence; conservative for free generation.
      * chrF: character n-gram correspondence; especially useful for Korean.
      * ROUGE: lexical overlap/sequence recall-F1 view.
      * METEOR: lexical alignment without English-only synonym/stemming in the
        recommended ``multilingual_lexical`` mode.
      * cosine: bag-of-lexical-token similarity.
    """
    if predictions is None or labels is None:
        raise ValueError("predictions and labels must not be None.")
    if len(predictions) != len(labels):
        raise ValueError(
            "Length mismatch: "
            f"len(predictions)={len(predictions)}, len(labels)={len(labels)}."
        )
    if len(predictions) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    predictions_str = [_to_text(value) for value in predictions]
    labels_str = [_to_text(value) for value in labels]

    empty_label_indices = [
        i for i, label in enumerate(labels_str) if not normalize_unicode(label).strip()
    ]
    if empty_label_indices:
        raise ValueError(
            "References must not be empty/whitespace-only; "
            f"indices={empty_label_indices[:8]}."
        )

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=_UnicodeWordTokenizer(),
    )
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    cosine_scores: List[float] = []

    for prediction, label in zip(predictions_str, labels_str):
        pair_scores = rouge.score(label, prediction)
        for key in rouge_scores:
            rouge_scores[key].append(float(pair_scores[key].fmeasure))

        cosine_scores.append(
            _cosine_similarity_from_tokens(
                tokenize_unicode_words(prediction),
                tokenize_unicode_words(label),
            )
        )

    bleu = _compute_bleu(predictions_str, labels_str)
    chrf = _compute_chrf(predictions_str, labels_str)

    normalized_meteor_mode = str(meteor_mode).strip().lower()
    if normalized_meteor_mode == "multilingual_lexical":
        meteor = _compute_multilingual_lexical_meteor(predictions_str, labels_str)
    elif normalized_meteor_mode == "standard":
        meteor = _compute_standard_meteor(predictions_str, labels_str)
    else:
        raise ValueError(
            "meteor_mode must be 'multilingual_lexical' or 'standard', "
            f"got {meteor_mode!r}."
        )

    result = {
        "rouge1": _strict_mean(rouge_scores["rouge1"], name="rouge1"),
        "rouge2": _strict_mean(rouge_scores["rouge2"], name="rouge2"),
        "rougeL": _strict_mean(rouge_scores["rougeL"], name="rougeL"),
        "bleu": float(bleu),
        "chrf": float(chrf),
        "meteor": float(meteor),
        "cosine_similarity": _strict_mean(cosine_scores, name="cosine_similarity"),
    }

    for name, value in result.items():
        if not math.isfinite(value):
            raise RuntimeError(f"Metric {name!r} is non-finite: {value!r}.")
        if value < -1e-12 or value > 1.0 + 1e-12:
            raise RuntimeError(
                f"Metric {name!r} escaped the expected 0..1 range: {value!r}."
            )

    return result
