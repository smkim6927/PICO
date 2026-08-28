#!/usr/bin/env bash
# scripts/env.sh
# Shared configuration. Sourced by every other script.
# Override any value from the caller's environment, for example:
#   SEED=911 bash scripts/train.sh pico
#
# Do NOT set a condition here for one method only. The point of this file is
# that every arm of the comparison reads the same numbers.

# ── model scale ──────────────────────────────────────────────────────────
# MODEL_SIZE selects a preset. MODEL_NAME/BATCH_SIZE/etc. set explicitly in
# the environment always win over the preset.
# Per-device batch 4 with accumulation 8 at every scale, per paper H.2.8.
#   1b (default)  Llama-3.2-1B-Instruct   DDP
#   3b            Llama-3.2-3B-Instruct   DDP
#   8b            Llama-3.1-8B-Instruct   balanced device map, 1 epoch
export MODEL_SIZE="${MODEL_SIZE:-1b}"
case "$MODEL_SIZE" in
  1b)
    _DEF_MODEL="meta-llama/Llama-3.2-1B-Instruct"
    _DEF_EPOCHS=5; _DEF_DEVICE_MAP="" ;;
  3b)
    _DEF_MODEL="meta-llama/Llama-3.2-3B-Instruct"
    _DEF_EPOCHS=5; _DEF_DEVICE_MAP="" ;;
  8b)
    _DEF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    _DEF_EPOCHS=1; _DEF_DEVICE_MAP="balanced" ;;
  *)
    echo "error: unknown MODEL_SIZE '$MODEL_SIZE' (use 1b, 3b, or 8b)" >&2
    exit 1 ;;
esac

# ── shared training conditions ───────────────────────────────────────────
export MODEL_NAME="${MODEL_NAME:-$_DEF_MODEL}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export NUM_EPOCHS="${NUM_EPOCHS:-$_DEF_EPOCHS}"
# Single-process model parallelism for scales that do not fit on one GPU.
# Non-empty DEVICE_MAP switches the launcher from accelerate to plain python.
export DEVICE_MAP="${DEVICE_MAP:-$_DEF_DEVICE_MAP}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export MAX_LENGTH="${MAX_LENGTH:-256}"
export CHUNK_SIZE="${CHUNK_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
# Seeds run per method, in order. Override with SEEDS="777" for a single run.
export SEEDS="${SEEDS:-777 911 4041}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# ── PICO ─────────────────────────────────────────────────────────────────
# f = 1 at every scale. The camera-ready reports f = 1 as the primary
# configuration for the main comparison, the ablations, and the scaling runs.
export PICO_SIGMA="${PICO_SIGMA:-0.001}"
export PICO_F="${PICO_F:-1}"
export PICO_K="${PICO_K:-1}"

# ── paths ────────────────────────────────────────────────────────────────
export RUN_ROOT="${RUN_ROOT:-runs/${MODEL_SIZE}}"
export RESULT_ROOT="${RESULT_ROOT:-results/${MODEL_SIZE}}"

# ── dataset ──────────────────────────────────────────────────────────────
export HF_REPO_ID="${HF_REPO_ID:-aigogongburani/PICO_Cross-Lingual_CPT_Corpora}"
# Pin a commit sha once the repo is stable. Empty means main.
export HF_REVISION="${HF_REVISION:-}"

# KO-Medical is not redistributable and is never fetched from the Hub.
# It combines material derived from AI Hub datasets 71487 (v1.2) and
# 110 (v1.1). Obtain them yourself and point this at the assembled file.
#   export KOR_MEDICAL_PATH=/data/aihub71487/new-medical-kor-dataset.txt
export KOR_MEDICAL_PATH="${KOR_MEDICAL_PATH:-}"

# ── evaluation ───────────────────────────────────────────────────────────
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-512}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
export EVAL_SHOT_TYPE="${EVAL_SHOT_TYPE:-zero-shot}"
# Requested split. domain_map falls back to train when a dataset has no test split.
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export EVAL_PER_DOMAIN_METRICS="${EVAL_PER_DOMAIN_METRICS:-chrf,bleu,rougeL,meteor,cosine_similarity,token_precision,token_recall,token_f1}"
export EVAL_INFERENCE_DTYPE="${EVAL_INFERENCE_DTYPE:-bf16}"
# Curriculum domains plus GSM8K as the evaluation-only OOD probe (paper Sec. 4).
export EVAL_DOMAINS="${EVAL_DOMAINS:-kor_medical,eng_medical,kor_legal,eng_legal,math}"

# ── method list ──────────────────────────────────────────────────────────
ALL_METHODS=(pico adam sophia sgd ewc replay rewarm mer lora adamw)

# ── helpers ──────────────────────────────────────────────────────────────
die() { echo "error: $*" >&2; exit 1; }

is_known_method() {
  local m="$1"
  for k in "${ALL_METHODS[@]}"; do [ "$k" = "$m" ] && return 0; done
  return 1
}

# The four stages, in paper order. Used to build the eval curriculum.
STAGE_IDS=(kor_medical eng_medical kor_legal eng_legal)

# Stages actually available. KO-Medical only when KOR_MEDICAL_PATH is set.
available_stages() {
  local out=()
  for s in "${STAGE_IDS[@]}"; do
    if [ "$s" = "kor_medical" ] && [ -z "$KOR_MEDICAL_PATH" ]; then
      continue
    fi
    out+=("$s")
  done
  echo "${out[@]}"
}

check_fsdp_config() {
  # FSDP flattens parameters, which silently disables PICO's spectral monitor.
  local cfg
  cfg="$(python -c 'import os,accelerate.utils as u; print(u.default_config_file if hasattr(u,"default_config_file") else "")' 2>/dev/null || true)"
  if [ -n "$cfg" ] && [ -f "$cfg" ] && grep -qi 'FSDP' "$cfg"; then
    die "accelerate config at $cfg uses FSDP. Use DDP (MULTI_GPU) instead."
  fi
}
