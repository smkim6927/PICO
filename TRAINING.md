# Training and Evaluation

Everything runs through three shell scripts. They read one shared configuration
file so that PICO and every baseline train under identical conditions.

```
scripts/env.sh      shared conditions, paths, dataset settings
scripts/train.sh    continual pre-training for one or more methods
scripts/eval.sh     continual-learning evaluation, builds the curriculum itself
scripts/run_all.sh  train then evaluate, one command
```

## Setup

```bash
pip install -r requirements.txt
accelerate config          # choose MULTI_GPU (DDP). Do NOT choose FSDP.
export HF_TOKEN=hf_xxx     # the corpus repo is private
```

FSDP is rejected at startup. FSDP flattens parameters, so PICO's `p.dim() == 2`
spectral guard matches nothing and SpecFlag is silently disabled. A run that
quietly turns off half the method is worse than a run that refuses to start.

## Corpora

Three of the four stages download automatically from
`aigogongburani/PICO_Cross-Lingual_CPT_Corpora`.

The Korean Medical stage does not. Per the paper's data appendix, it combines
material derived from two AI Hub datasets: 71487, the Medical and Legal
Professional Book Corpus (v1.2), and 110, the Professional Domain Corpus
(v1.1). AI Hub terms permit training use but prohibit redistribution, so
neither source is mirrored. Obtain them yourself and point the pipeline at
your assembled copy:

```bash
export KOR_MEDICAL_PATH=/data/aihub71487/new-medical-kor-dataset.txt
```

Without it the pipeline runs a three-stage stream and prints a warning. A
three-stage run covers a different stream from the four-stage protocol in
the paper and is not directly comparable to it.

Work using the Korean Medical stage must acknowledge both AI Hub sources as
products of the Korean Ministry of Science and ICT intelligent information
industry infrastructure program, administered by the National Information
Society Agency. The README carries BibTeX entries for both.

## Running

```bash
bash scripts/run_all.sh pico              # PICO x seeds 777, 911, 4041
bash scripts/run_all.sh pico adam replay  # several methods in sequence
bash scripts/run_all.sh all               # every method
bash scripts/run_all.sh --eval-only pico  # evaluate existing checkpoints
SEEDS="777" bash scripts/run_all.sh pico  # single seed
```

Seeds run method-outer, seed-inner: all three seeds of one method finish,
including their evaluations when using `run_all.sh`, before the next method
starts. The default seed set is `777 911 4041`.

Override any shared condition from the environment. It applies to every method
in that invocation, which is the point:

```bash
SEEDS="911" bash scripts/run_all.sh all
NUM_EPOCHS=1 BATCH_SIZE=16 bash scripts/train.sh pico adam
CUDA_VISIBLE_DEVICES=0,1,3 bash scripts/train.sh pico
```

## Model scales

`MODEL_SIZE` selects the backbone and matching defaults; any explicitly set
variable still wins.

```bash
bash scripts/run_all.sh pico                    # 1b: Llama-3.2-1B-Instruct, DDP
MODEL_SIZE=3b bash scripts/run_all.sh pico      # 3b: Llama-3.2-3B-Instruct, DDP
MODEL_SIZE=8b bash scripts/run_all.sh pico      # 8b: Llama-3.1-8B-Instruct
```

| Size | Backbone | Parallelism | batch | epochs |
|---|---|---|---|---|
| `1b` | Llama-3.2-1B-Instruct | DDP (accelerate launch) | 4 | 5 |
| `3b` | Llama-3.2-3B-Instruct | DDP (accelerate launch) | 4 | 5 |
| `8b` | Llama-3.1-8B-Instruct | single process, `device_map=balanced` | 4 | 1 |

The 8B scale does not fit full fine-tuning on one 40GB GPU, and FSDP is
rejected because parameter flattening silently disables SpecFlag. The 8B path
therefore uses single-process model parallelism via `--device_map balanced`:
layers spread across the visible GPUs, parameters keep their 2D shapes, and
the spectral monitor stays active. This matches the corrected 8B protocol
from the FSDP-artifact investigation. Set the GPUs with
`CUDA_VISIBLE_DEVICES` as usual; `scripts/train.sh` switches from
`accelerate launch` to plain `python` automatically when a device map is set.
Outputs are separated per scale under `runs/<size>/` and `results/<size>/`.

PICO uses `f = 1` at every scale. The camera-ready reports `f = 1` as the
primary configuration throughout, so `PICO_F` defaults to 1 and the scaling
runs use the same value.

## Methods

| `--method` | Optimizer | What makes it different |
|---|---|---|
| `pico` | PICO | GateU, SpecFlag, group pause, gated Gaussian noise |
| `adam` | Adam | plain sequential baseline (H.2.1) |
| `sophia` | SophiaG | GNB diagonal-Hessian preconditioning, k=10 (H.2.2) |
| `sgd` | SGD | momentum fixed at 0 in code, no flag (H.2.1) |
| `ewc` | AdamW | Fisher penalty, lambda=10, 200 Fisher batches (H.2.3) |
| `replay` | Adam | FIFO buffer C=5000, N=2000, r=0.5 (H.2.4) |
| `rewarm` | Adam | per-stage warmup and cosine schedule (H.2.6) |
| `mer` | Adam | replay plus Reptile interpolation every k updates (H.2.5) |
| `lora` | Adam | rank 16, alpha fixed at 16 in code, frozen base (H.2.7) |
| `adamw` | AdamW | extra reference, not in the paper's baseline table |

Each method is a subclass in `cp4llm/train/methods.py` that overrides only its
hooks. The training loop, data pipeline, seeding, padding, and checkpointing all
live in `cp4llm/train/common.py` and are shared. Progress is reported to
stdout and `train.log`; there is no external experiment tracker. A method cannot
change a shared condition without editing the shared file, which makes
accidental divergence hard.

## Unified conditions

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Precision | bfloat16, `mixed_precision="bf16"` |
| Batch size per device | 4 (paper H.2.8) |
| Gradient accumulation | 8 |
| Effective batch size per device | 32 |
| Epochs per stage | 5 |
| Learning rate | 2e-5, constant |
| `max_length` / `chunk_size` | 256 / 64 |
| Padding | left |
| Gradient clipping | none |
| LR schedule, warmup | none, except `rewarm` where it is the method |
| Optimizer state | preserved across stages |
| Gradient checkpointing | on, `use_cache=False` |
| Seeds | 777, 911, 4041 per method, deterministic algorithms on |
| Parallelism | DDP only |

## What changed from the original per-method scripts

These were real divergences between arms, not stylistic differences.

1. **FSDP removed everywhere.** Several scripts imported
   `FullyShardedDataParallelPlugin` or referenced an undefined
   `transformer_auto_wrap_policy`. `ewc+adam.py`, `llama-replay.py`, and
   `rewarm+adam.py` raised `NameError` on that reference. `rewarm+adam.py` also
   had an unclosed parenthesis and could not be imported at all.
2. **Global TF32 removed.** `llama-replay.py` set
   `torch.backends.cuda.matmul.allow_tf32 = True` at import, changing matmul
   precision for that arm only while every other script disabled it.
3. **Gradient clipping removed** from LoRA and MER. PICO uses none.
4. **Batch size, accumulation, epochs, LR, seed, and workers unified to the
   paper's shared configuration (H.2.8).** Per-device batch 4, accumulation
   8, effective 32 per device. The archived scripts ranged over batch 24 to
   36 and accumulation 8 to 16 and were superseded by the audited
   configuration.
5. **LoRA no longer imports `optimizer.IVE`.** It used `HPO_v3_` while its own
   logging recorded `"optimizer": "AdamW"`. It now uses AdamW.
6. **MER uses the shared data pipeline.** Its own packing loop produced a
   different token stream from `TextDatasetwchunk`, so it was not comparable
   to the arm it was meant to be paired against.
7. **Checkpoints are keyed by stage id**, not by source filename. Filename
   keying broke once corpora came from the Hub cache.
8. **W&B removed entirely.** Training logs go to stdout and `train.log`.
   The evaluator's `wandb_mode` is forced to `disabled` and its import is
   guarded, so the package is not required.
9. **Paper alignment (H.2).** SGD is momentum-free per H.2.1. LoRA uses
   Adam on adapter parameters with rank 16 and alpha 16 per H.2.7, confirmed
   against the archived run configuration. Sophia follows H.2.2 with the
   official SophiaG implementation and the Gauss-Newton-Bartlett estimator.
10. **SHA and fingerprint enforcement removed from evaluation.** The revision
   gate, tokenizer and model fingerprint hashing, cross-checkpoint fingerprint
   comparison, and the frozen-content sha256 are all gone. What remains is a
   cheap per-domain example-count consistency check and the frozen-manifest
   construction itself, which already fixes the prompt and reference set.
   The manifest cache keying inside `domain_map.py` still uses internal hashes
   for cache file naming; that is a cache mechanism, not a validation gate.

## Outputs

```
runs/<method>/seed<seed>/
    kor_medical_epoch_5/     per-stage checkpoints
    eng_medical_epoch_5/
    kor_legal_epoch_5/
    eng_legal_epoch_5/
    train.log

results/cl_eval/<method>/seed<seed>/
    cl_summary.json          stage-by-domain performance matrix
    metrics/                 FWT, BWT, AvgF, CR, WCR
    eval.log

One directory per method and seed. Aggregate across the three seeds when
reporting.
```

Evaluation covers the four curriculum domains plus `math` (GSM8K) as the
paper's evaluation-only out-of-domain probe; adjust with `EVAL_DOMAINS`.

`scripts/eval.sh` builds the `--curriculum` string from the checkpoints that
training actually wrote, with absolute paths, so the evaluated stream always
matches the trained stream. The requested split is `test`; `domain_map` falls
back to `train` for datasets that publish no test split, which is the intended
behavior. If a stage is missing, it is dropped from the curriculum and the script
warns that the metrics are not comparable to the paper.
