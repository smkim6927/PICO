#!/usr/bin/env python3
"""
cp4llm/train_cpt.py
===================
One entry point for PICO and every baseline.

    accelerate launch cp4llm/train_cpt.py --method pico  --output_dir runs/pico
    accelerate launch cp4llm/train_cpt.py --method adam  --output_dir runs/adam

Everything that is not the method itself is fixed by cp4llm/train/common.py.
Overriding a shared flag on one method and not another breaks the comparison,
so the shell scripts in scripts/ set them once for all methods.
"""

import os

os.environ.setdefault("PYTHONHASHSEED", "777")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.common import HF_DATASET_REPO, STAGE_SPEC, TRAIN_DEFAULTS  # noqa: E402
from train.methods import METHODS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-lingual continual pre-training: PICO and baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--method", type=str, required=True, choices=sorted(METHODS),
                   help="training method")
    p.add_argument("--output_dir", type=str, required=True)

    # ── shared conditions, identical for every method ────────────────────
    g = p.add_argument_group("shared conditions (keep identical across methods)")
    g.add_argument("--model_name", type=str, default=TRAIN_DEFAULTS["model_name"])
    g.add_argument("--batch_size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    g.add_argument("--gradient_accumulation_steps", type=int,
                   default=TRAIN_DEFAULTS["gradient_accumulation_steps"])
    g.add_argument("--num_epochs", type=int, default=TRAIN_DEFAULTS["num_epochs"])
    g.add_argument("--learning_rate", type=float,
                   default=TRAIN_DEFAULTS["learning_rate"])
    g.add_argument("--max_length", type=int, default=TRAIN_DEFAULTS["max_length"])
    g.add_argument("--chunk_size", type=int, default=TRAIN_DEFAULTS["chunk_size"])
    g.add_argument("--num_workers", type=int, default=TRAIN_DEFAULTS["num_workers"])
    g.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--device_map", type=str, default=None,
                   help="e.g. 'balanced'. Single-process model parallelism for "
                        "models that do not fit on one GPU (8B). Keeps 2D "
                        "parameter shapes so SpecFlag stays active, unlike FSDP.")

    # ── corpora ──────────────────────────────────────────────────────────
    d = p.add_argument_group("corpora")
    d.add_argument("--hf_repo_id", type=str, default=HF_DATASET_REPO)
    d.add_argument("--hf_revision", type=str, default=None,
                   help="pin a commit sha for reproducibility")
    d.add_argument("--hf_cache_dir", type=str, default=None)
    d.add_argument("--only_stages", type=str, default=None,
                   help="comma-separated subset, e.g. eng_medical,kor_legal")

    # ── logging ──────────────────────────────────────────────────────────
    w = p.add_argument_group("logging")
    w.add_argument("--no_debug", dest="debug", action="store_false")
    p.set_defaults(debug=True)

    # ── PICO ─────────────────────────────────────────────────────────────
    m = p.add_argument_group("pico")
    m.add_argument("--beta_utility", type=float, default=0.999)
    m.add_argument("--sigma", type=float, default=0.001, help="sigma_0")
    m.add_argument("--spectral_update_freq", type=int, default=1, help="f")
    m.add_argument("--power_iterations", type=int, default=1, help="K")

    # SGD has no method flags. Momentum is fixed at 0 in code (paper H.2.1).

    # ── Sophia ───────────────────────────────────────────────────────────
    so = p.add_argument_group("sophia")
    so.add_argument("--sophia_beta1", type=float, default=0.965)
    so.add_argument("--sophia_beta2", type=float, default=0.99)
    so.add_argument("--sophia_rho", type=float, default=0.05)
    so.add_argument("--sophia_k", type=int, default=10,
                    help="Hessian update interval in optimizer updates")

    # ── EWC ──────────────────────────────────────────────────────────────
    e = p.add_argument_group("ewc")
    e.add_argument("--lambda_ewc", type=float, default=10.0)
    e.add_argument("--fisher_num_batches", type=int, default=200)
    e.add_argument("--online_ewc", action="store_true")
    e.add_argument("--ewc_gamma", type=float, default=0.95)

    # ── Replay and MER ───────────────────────────────────────────────────
    r = p.add_argument_group("replay / mer")
    r.add_argument("--replay_capacity", type=int, default=5000)
    r.add_argument("--replay_ratio", type=float, default=0.5)
    r.add_argument("--replay_sample_per_stage", type=int, default=2000)
    r.add_argument("--mer_k", type=int, default=500)
    r.add_argument("--mer_eps", type=float, default=0.1)
    r.add_argument("--no_mer_reset_anchor_per_stage",
                   dest="mer_reset_anchor_per_stage", action="store_false")
    p.set_defaults(mer_reset_anchor_per_stage=True)

    # ── Re-warm ──────────────────────────────────────────────────────────
    rw = p.add_argument_group("rewarm")
    rw.add_argument("--warmup_ratio", type=float, default=0.05)
    rw.add_argument("--min_lr_ratio", type=float, default=0.10)

    # ── LoRA ─────────────────────────────────────────────────────────────
    l = p.add_argument_group("lora")
    l.add_argument("--lora_r", type=int, default=16)
    # lora_alpha is fixed at 16 in code (paper H.2.7) and has no flag.
    l.add_argument("--lora_dropout", type=float, default=0.05)
    l.add_argument("--lora_target_modules", type=str, default=None,
                   help="comma-separated; default is the Llama attention and MLP set")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.only_stages:
        args.only_stages = [s.strip() for s in args.only_stages.split(",") if s.strip()]
        known = {spec["id"] for spec in STAGE_SPEC}
        unknown = [s for s in args.only_stages if s not in known]
        if unknown:
            raise SystemExit(f"unknown stage ids: {unknown}. known: {sorted(known)}")
    if args.lora_target_modules:
        args.lora_target_modules = [
            s.strip() for s in args.lora_target_modules.split(",") if s.strip()
        ]
    args.stage_paths = None

    trainer = METHODS[args.method](args)
    trainer.load_model_and_tokenizer()
    trainer.train()


if __name__ == "__main__":
    main()
