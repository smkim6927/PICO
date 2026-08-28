"""
CL Transfer Metric Computation from cl_summary.json
====================================================

Loads a `cl_summary.json` (with structure:
  - R_matrix: dict {metric_name: list-of-stage-dicts}
  - b_initial: dict {domain: {metric_name: baseline_value}}
  - domains, first_learned_step, ...)

Builds a (T+1) x T R-matrix for each metric over LEARNED domains,
computes FWT, BWT, AvgF, CR, WCR using the paper's formulas, and
saves results to JSON.

R-matrix convention (paper §3.X)
--------------------------------
  Row 0     = R̄_i  (pre-CPT baseline from b_initial; NOT random-init)
  Row 1..T  = scores after training stage 1..T
  Columns   = learned domains, in the order they were trained
              (derived from first_learned_step, sorted by stage index)

Formulas (matching paper definitions exactly)
---------------------------------------------
  FWT  = (1/(T-1)) Σ_{i=2..T} (R[i-1, i] - R̄_i)
  BWT  = (1/(T-1)) Σ_{i=1..T-1} (R[T, i] - R[i, i])
  AvgF = (1/(T-1)) Σ_{i=1..T-1} (R^peak_i - R[T, i])
           where R^peak_i = max_{t ∈ [i, T-1]} R[t, i]
  CR(τ)= (1/T) Σ_{i=1..T} 1[ R[T, i] < τ · R^*_i ]
  WCR  = max_i (1 - R[T, i] / R^*_i)
           where R^*_i = max_{t ∈ [i, T]} R[t, i]

Note: For lower-is-better metrics (e.g., 'loss', 'ppl'), the sign of
each metric is flipped before computation by negating the matrix
(this preserves "higher is better" assumption of all formulas).

References
----------
  Lopez-Paz & Ranzato 2017 (NeurIPS): FWT, BWT
  Chaudhry et al. 2018 (ECCV): AvgF, intransigence
  De Lange et al. 2023 (ICLR): worst-case continual metrics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. Number formatting
# =============================================================================

def trunc4(x: float) -> float:
    """Truncate (not round) toward zero, 4 decimals. NaN-safe."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return math.trunc(x * 10000) / 10000


def fmt4(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    s = f"{trunc4(x):.4f}"
    return "0.0000" if s == "-0.0000" else s


# =============================================================================
# 2. Metric computation — EXACTLY matching paper formulas
# =============================================================================

def compute_fwt(R: np.ndarray) -> float:
    """FWT = (1/(T-1)) Σ_{i=2..T} (R[i-1, i] - R̄_i)"""
    T = R.shape[1]
    if T < 2:
        return float("nan")
    s = 0.0
    for i in range(2, T + 1):
        j = i - 1                              # column 0-indexed
        stage_row = i - 1                      # row of stage i-1
        s += (R[stage_row, j] - R[0, j])
    return s / (T - 1)


def compute_bwt(R: np.ndarray) -> float:
    """BWT = (1/(T-1)) Σ_{i=1..T-1} (R[T, i] - R[i, i])"""
    T = R.shape[1]
    if T < 2:
        return float("nan")
    s = 0.0
    for i in range(1, T):
        j = i - 1
        s += (R[T, j] - R[i, j])
    return s / (T - 1)


def compute_avgf(R: np.ndarray) -> float:
    """
    AvgF = (1/(T-1)) Σ_{i=1..T-1} (R^peak_i - R[T, i])
      where R^peak_i = max_{t ∈ [i, T-1]} R[t, i]   (final excluded)
    """
    T = R.shape[1]
    if T < 2:
        return float("nan")
    s = 0.0
    for i in range(1, T):
        j = i - 1
        peak = float(np.max(R[i:T, j]))       # rows i..T-1 inclusive
        s += (peak - R[T, j])
    return s / (T - 1)


def compute_peak_star(R: np.ndarray) -> np.ndarray:
    """R^*_i = max_{t ∈ [i, T]} R[t, i]   (final included)"""
    T = R.shape[1]
    peaks = np.zeros(T)
    for i in range(1, T + 1):
        j = i - 1
        peaks[j] = float(np.max(R[i:T+1, j]))
    return peaks


def compute_cr(R: np.ndarray, tau: float) -> float:
    """CR(τ) = (1/T) Σ_{i=1..T} 1[ R[T, i] < τ · R^*_i ]"""
    T = R.shape[1]
    peaks = compute_peak_star(R)
    count = 0
    for j in range(T):
        if peaks[j] > 0 and R[T, j] < tau * peaks[j]:
            count += 1
    return count / T


def compute_wcr(R: np.ndarray) -> float:
    """WCR = max_i (1 - R[T, i] / R^*_i)"""
    T = R.shape[1]
    peaks = compute_peak_star(R)
    worst = 0.0
    for j in range(T):
        if peaks[j] > 0:
            ratio = 1.0 - (R[T, j] / peaks[j])
            worst = max(worst, ratio)
    return worst


def compute_all_metrics(R: np.ndarray, tau: float) -> Dict[str, float]:
    """Return all five metrics for an R-matrix."""
    return {
        "FWT":  float(compute_fwt(R)),
        "BWT":  float(compute_bwt(R)),
        "AvgF": float(compute_avgf(R)),
        "CR":   float(compute_cr(R, tau=tau)),
        "WCR":  float(compute_wcr(R)),
    }


# =============================================================================
# 3. Build (T+1) x T R-matrix from cl_summary structure
# =============================================================================

def build_R_matrix(
    summary: dict,
    metric_name: str,
    lower_is_better: bool,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (T+1) x T R-matrix for one metric.

    Returns
    -------
    R : np.ndarray of shape (T+1, T)
        Row 0 = pre-CPT baseline (from b_initial), rows 1..T = after stage 1..T.
        Columns = learned domains in stage order.
    learned_domains : list of str
        Column ordering of R, matching first_learned_step.
    """
    first_learned = summary["first_learned_step"]                  # {domain: stage_index_starting_from_1}
    # Order learned domains by their first_learned_step value
    learned_domains = [
        dom for dom, _ in sorted(first_learned.items(), key=lambda kv: kv[1])
    ]
    T = len(learned_domains)
    if T < 2:
        raise ValueError(f"Need at least 2 learned domains for CL metrics, got {T}.")

    if metric_name not in summary["R_matrix"]:
        raise KeyError(
            f"Metric '{metric_name}' not in R_matrix. "
            f"Available: {list(summary['R_matrix'].keys())}"
        )
    if metric_name not in summary["b_initial"][learned_domains[0]]:
        raise KeyError(
            f"Metric '{metric_name}' not in b_initial. "
            f"Available: {list(summary['b_initial'][learned_domains[0]].keys())}"
        )

    R = np.full((T + 1, T), np.nan, dtype=float)

    # Row 0: pre-CPT baseline from b_initial
    for j, dom in enumerate(learned_domains):
        R[0, j] = float(summary["b_initial"][dom][metric_name])

    # Rows 1..T: after each training stage
    # NOTE: R_matrix list may include an entry at index 0 that mirrors b_initial.
    # We use stage_idx 1..T from the matrix, mapping to row 1..T of R.
    stage_dicts = summary["R_matrix"][metric_name]
    if len(stage_dicts) < T + 1:
        raise ValueError(
            f"R_matrix['{metric_name}'] has {len(stage_dicts)} stage entries; "
            f"need at least {T+1} (= pre-CPT + T learned stages)."
        )
    for stage in range(1, T + 1):
        stage_dict = stage_dicts[stage]
        for j, dom in enumerate(learned_domains):
            R[stage, j] = float(stage_dict[dom])

    # Sign flip for lower-is-better metrics so that all formulas assume "higher is better"
    if lower_is_better:
        R = -R

    return R, learned_domains


# =============================================================================
# 4. Process one summary file → save metrics JSON
# =============================================================================

def process_summary(
    input_file: Path,
    output_dir: Path,
    metrics_to_compute: List[str],
    tau: float,
) -> None:
    if not input_file.is_file():
        print(f"🚨 입력 파일이 없습니다: {input_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    lower_set = set(summary.get("lower_is_better", []))
    file_stem = input_file.stem

    # 자동: metrics_to_compute에 'all'이 있거나 비어있으면 R_matrix의 모든 키 사용
    if not metrics_to_compute or "all" in metrics_to_compute:
        metrics_to_compute = list(summary["R_matrix"].keys())

    results: Dict[str, Dict[str, Optional[float]]] = {}
    learned_domains_seen: List[str] = []

    for m_name in metrics_to_compute:
        try:
            lib = m_name in lower_set
            R, learned_domains = build_R_matrix(summary, m_name, lower_is_better=lib)
            if not learned_domains_seen:
                learned_domains_seen = learned_domains

            # NaN row 가 있으면 skip (예: accuracy 처럼 일부 도메인만 정의된 metric)
            if np.isnan(R).any():
                print(f"⚠️ '{m_name}': R matrix contains NaN → skipped")
                results[m_name] = {k: None for k in ["FWT", "BWT", "AvgF", "CR", "WCR"]}
                continue

            raw = compute_all_metrics(R, tau=tau)

            # lower-is-better 였으면 부호 의미가 뒤집힘.
            # Sign-flip 후 모든 수식은 "higher better" 가정으로 계산됨.
            # 따라서 결과 부호는 그대로 두되, downstream 해석은 항상 same convention.
            # (FWT>0: positive transfer, BWT<0: forgetting, AvgF>0: forgetting, etc.)
            formatted = {k: float(fmt4(v)) for k, v in raw.items()}
            results[m_name] = formatted

        except Exception as e:
            print(f"❌ '{m_name}': {e}")
            results[m_name] = {k: None for k in ["FWT", "BWT", "AvgF", "CR", "WCR"]}

    # ── 출력 JSON ──
    output = {
        "source_file": str(input_file),
        "tau_for_CR": tau,
        "learned_domains_in_order": learned_domains_seen,
        "T": len(learned_domains_seen),
        "lower_is_better_metrics": sorted(list(lower_set)),
        "notes": (
            "For lower-is-better metrics, the matrix is internally negated so "
            "that all formulas assume 'higher is better'. Reported FWT/BWT/AvgF "
            "signs follow the higher-is-better convention regardless."
        ),
        "metrics": results,
    }

    out_path = output_dir / f"{file_stem}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    # ── 콘솔 요약 ──
    print(f"\n✅ 저장 완료: {out_path}")
    print(f"   T = {len(learned_domains_seen)}, "
          f"learned domains = {learned_domains_seen}\n")
    header = f"{'metric':<22} {'FWT':>10} {'BWT':>10} {'AvgF':>10} {'CR':>8} {'WCR':>10}"
    print(header)
    print("-" * len(header))
    for m_name, vals in results.items():
        if vals.get("FWT") is None:
            print(f"{m_name:<22} {'  (skipped)':>50}")
            continue
        print(
            f"{m_name:<22} "
            f"{fmt4(vals['FWT']):>10} {fmt4(vals['BWT']):>10} "
            f"{fmt4(vals['AvgF']):>10} {fmt4(vals['CR']):>8} {fmt4(vals['WCR']):>10}"
        )


# =============================================================================
# 5. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="cl_summary.json → CL transfer metrics (FWT/BWT/AvgF/CR/WCR)"
    )
    parser.add_argument(
        "--input_file", type=str,
        default="./results/cl_eval/pico_/all/llama3.1_8b_777/cl_summary.json",
        help="cl_summary.json path",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./results/cl_eval/pico_/all/llama3.1_8b_777/cr+wcr",
        help="Output folder for metrics JSON",
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=["bleu", "rougeL", "rouge1", "rouge2", "meteor",
                 "token_f1", "cosine_similarity", "jaccard_similarity",
                 "groundedness", "loss", "ppl"],
        help="Metrics to compute (use 'all' for every metric in R_matrix)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.90,
        help="τ threshold for CR (default 0.90 = 10%% collapse)",
    )
    args = parser.parse_args()

    process_summary(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        metrics_to_compute=args.metrics,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
