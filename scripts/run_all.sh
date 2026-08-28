#!/usr/bin/env bash
# scripts/run_all.sh — train then evaluate, in one command.
#
#   bash scripts/run_all.sh pico               # pico x seeds 777,911,4041, train+eval
#   bash scripts/run_all.sh pico adam replay   # method-outer, seed-inner
#   bash scripts/run_all.sh all
#   bash scripts/run_all.sh --eval-only pico
#   SEEDS="777" bash scripts/run_all.sh pico
#
# Options:
#   --eval-only   skip training, evaluate existing checkpoints
#   --train-only  skip evaluation

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

DO_TRAIN=1
DO_EVAL=1
METHODS=()

for arg in "$@"; do
  case "$arg" in
    --eval-only)  DO_TRAIN=0 ;;
    --train-only) DO_EVAL=0 ;;
    -*)           die "unknown option: $arg" ;;
    *)            METHODS+=("$arg") ;;
  esac
done

[ "${#METHODS[@]}" -ge 1 ] || die \
  "usage: bash scripts/run_all.sh [--eval-only|--train-only] <method|all> [...]"

if [ "${METHODS[0]}" = "all" ]; then
  METHODS=("${ALL_METHODS[@]}")
fi

echo "methods : ${METHODS[*]}"
echo "seeds   : $SEEDS"
echo "stages  : $(available_stages)"
echo

if [ "$DO_TRAIN" -eq 1 ]; then
  bash scripts/train.sh "${METHODS[@]}"
fi

if [ "$DO_EVAL" -eq 1 ]; then
  bash scripts/eval.sh "${METHODS[@]}"
fi

echo
echo "pipeline complete."
echo "  checkpoints : ${RUN_ROOT}/<method>/seed<seed>/"
echo "  metrics     : ${RESULT_ROOT}/cl_eval/<method>/seed<seed>/metrics/"
