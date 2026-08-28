#!/usr/bin/env bash
# scripts/train.sh — run continual pre-training for one or more methods.
#
#   bash scripts/train.sh pico                 # pico x seeds 777,911,4041
#   bash scripts/train.sh adam sgd ewc         # each method runs all seeds
#   bash scripts/train.sh all
#   SEEDS="777" bash scripts/train.sh pico     # single seed
#   MODEL_SIZE=3b bash scripts/train.sh pico   # Llama-3.2-3B-Instruct, DDP
#   MODEL_SIZE=8b bash scripts/train.sh pico   # Llama-3.1-8B-Instruct,
#                                              # single process, balanced map
#
# Order is method-outer, seed-inner: all seeds of one method finish before
# the next method starts.
#
# Every method receives the identical shared conditions from scripts/env.sh.
# Method-specific flags are added only where the method requires them.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

[ "$#" -ge 1 ] || die "usage: bash scripts/train.sh <method|all> [method ...]"

if [ "$1" = "all" ]; then
  METHODS=("${ALL_METHODS[@]}")
else
  METHODS=("$@")
fi

for m in "${METHODS[@]}"; do
  is_known_method "$m" || die "unknown method '$m'. known: ${ALL_METHODS[*]}"
done

check_fsdp_config

if [ -z "$KOR_MEDICAL_PATH" ]; then
  echo "WARNING: KOR_MEDICAL_PATH is not set."
  echo "         The kor_medical stage will be skipped, so the run covers"
  echo "         only three of the four stages used in the paper."
  echo "         Its two AI Hub sources (71487 v1.2 and 110 v1.1) permit"
  echo "         training use but not redistribution, so they cannot be"
  echo "         fetched automatically."
  echo
fi

COMMON=(
  --model_name "$MODEL_NAME"
  --batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --num_epochs "$NUM_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --max_length "$MAX_LENGTH"
  --chunk_size "$CHUNK_SIZE"
  --num_workers "$NUM_WORKERS"
  --weight_decay "$WEIGHT_DECAY"
  --hf_repo_id "$HF_REPO_ID"
)
[ -n "$HF_REVISION" ] && COMMON+=(--hf_revision "$HF_REVISION")

for METHOD in "${METHODS[@]}"; do
  EXTRA=()
  case "$METHOD" in
    pico)
      EXTRA=(--sigma "$PICO_SIGMA"
             --spectral_update_freq "$PICO_F"
             --power_iterations "$PICO_K") ;;
    ewc)    EXTRA=(--lambda_ewc 10.0 --fisher_num_batches 200) ;;
    replay) EXTRA=(--replay_capacity 5000 --replay_ratio 0.5
                   --replay_sample_per_stage 2000) ;;
    mer)    EXTRA=(--replay_capacity 5000 --replay_ratio 0.5
                   --replay_sample_per_stage 2000
                   --mer_k 500 --mer_eps 0.1) ;;
    rewarm) EXTRA=(--warmup_ratio 0.05 --min_lr_ratio 0.10) ;;
    lora)   EXTRA=(--lora_r 16 --lora_dropout 0.05) ;;
  esac

  for SEED in $SEEDS; do
    OUT="${RUN_ROOT}/${METHOD}/seed${SEED}"
    mkdir -p "$OUT"

    echo "========================================================================"
    echo " TRAIN  method=$METHOD  size=$MODEL_SIZE  seed=$SEED  out=$OUT"
    echo "========================================================================"

    if [ -n "$DEVICE_MAP" ]; then
      # Model parallel across the visible GPUs in one process. accelerate
      # launch would spawn one process per GPU, which conflicts with the map.
      python cp4llm/train_cpt.py \
        --method "$METHOD" \
        --output_dir "$OUT" \
        --seed "$SEED" \
        --device_map "$DEVICE_MAP" \
        "${COMMON[@]}" "${EXTRA[@]}" \
        2>&1 | tee "${OUT}/train.log"
    else
      accelerate launch cp4llm/train_cpt.py \
        --method "$METHOD" \
        --output_dir "$OUT" \
        --seed "$SEED" \
        "${COMMON[@]}" "${EXTRA[@]}" \
        2>&1 | tee "${OUT}/train.log"
    fi
  done
done

echo
echo "training complete: ${METHODS[*]}"
