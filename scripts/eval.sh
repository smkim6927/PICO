#!/usr/bin/env bash
# scripts/eval.sh — continual-learning evaluation for one or more trained methods.
#
#   bash scripts/eval.sh pico                  # pico x seeds 777,911,4041
#   bash scripts/eval.sh adam sgd ewc
#   bash scripts/eval.sh all
#   SEEDS="777" bash scripts/eval.sh pico      # single seed
#
# Order matches train.sh: method-outer, seed-inner.
#
# The curriculum string required by cp4llm/eval/run_cl_eval.py has the form
#   domain:checkpoint_path:epoch,domain:checkpoint_path:epoch,...
# It is built here from the per-stage checkpoints that training wrote, so the
# evaluation stream always matches the stream that was actually trained.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

[ "$#" -ge 1 ] || die "usage: bash scripts/eval.sh <method|all> [method ...]"

if [ "$1" = "all" ]; then
  METHODS=("${ALL_METHODS[@]}")
else
  METHODS=("$@")
fi

for m in "${METHODS[@]}"; do
  is_known_method "$m" || die "unknown method '$m'. known: ${ALL_METHODS[*]}"
done

build_curriculum() {
  # $1 = run directory. Emits "domain:abs_path:epoch,..." in paper stage order.
  # Paths are absolute because run_cl_eval.py executes from cp4llm/eval.
  local run_dir="$1"
  local items=()
  for stage in "${STAGE_IDS[@]}"; do
    local ckpt="${run_dir}/${stage}_epoch_${NUM_EPOCHS}"
    if [ -d "$ckpt" ]; then
      local abs
      abs="$(cd "$ckpt" && pwd)"
      items+=("${stage}:${abs}:${NUM_EPOCHS}")
    fi
  done
  local IFS=,
  echo "${items[*]}"
}

for METHOD in "${METHODS[@]}"; do
  for SEED in $SEEDS; do
    RUN_DIR="${RUN_ROOT}/${METHOD}/seed${SEED}"
    if [ ! -d "$RUN_DIR" ]; then
      echo "skip $METHOD seed=$SEED: no run at $RUN_DIR"
      continue
    fi

    CURRICULUM="$(build_curriculum "$RUN_DIR")"
    if [ -z "$CURRICULUM" ]; then
      echo "skip $METHOD seed=$SEED: no checkpoints in $RUN_DIR"
      continue
    fi

    N_STAGES=$(awk -F, '{print NF}' <<< "$CURRICULUM")
    OUT="${RESULT_ROOT}/cl_eval/${METHOD}/seed${SEED}"
    mkdir -p "$OUT"
    OUT_ABS="$(cd "$OUT" && pwd)"

    echo "========================================================================"
    echo " EVAL   method=$METHOD  seed=$SEED  stages=$N_STAGES"
    echo " curriculum: $CURRICULUM"
    echo "========================================================================"

    if [ "$N_STAGES" -lt 4 ]; then
      echo "WARNING: only $N_STAGES stages present. Metrics are not comparable"
      echo "         to the four-stage results reported in the paper."
    fi

    ( cd cp4llm/eval && python run_cl_eval.py \
        --seed "$SEED" \
        --base_ckpt "$MODEL_NAME" \
        --curriculum "$CURRICULUM" \
        --output_dir "$OUT_ABS" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --max_length "$EVAL_MAX_LENGTH" \
        --gen_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --per_domain_metrics "$EVAL_PER_DOMAIN_METRICS" \
        --shot_type "$EVAL_SHOT_TYPE" \
        --eval_split "$EVAL_SPLIT" \
        --eval_domains "$EVAL_DOMAINS" \
        --inference_dtype "$EVAL_INFERENCE_DTYPE" \
        --dump_raw_json \
    ) 2>&1 | tee "${OUT}/eval.log"

    SUMMARY="${OUT}/cl_summary.json"
    if [ -f "$SUMMARY" ]; then
      python cp4llm/eval/cl_metrics.py \
        --input_file "$SUMMARY" \
        --output_dir "${OUT}/metrics" \
        --metrics all
    else
      echo "note: $SUMMARY not found, skipping metric recomputation"
    fi
  done
done

echo
echo "evaluation complete: ${METHODS[*]}"
