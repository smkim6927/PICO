# eval/run_cl_eval.py

import argparse
from typing import List, Tuple

# 위에서 만든 모듈 import
from eval.cl_evaluator import CLEvaluatorCL, set_seed

def parse_curriculum_arg(arg: str) -> List[Tuple[str, str, int]]:
    """
    형식: "domain:path:epoch,domain:path:epoch,..."
    """
    items = []
    if not arg:
        return items
    pairs = [x.strip() for x in arg.split(",") if x.strip()]
    for p in pairs:
        parts = p.split(":")
        if len(parts) != 3:
            raise ValueError(f"Malformed curriculum item: '{p}'. Expected 'domain:path:epoch'")
        domain = parts[0].strip()
        path = parts[1].strip()
        epoch = int(parts[2].strip())
        items.append((domain, path, epoch))
    return items


def main():
    parser = argparse.ArgumentParser(description="CL Evaluation (RAW-space CL metrics)")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output_dir", type=str, default="./results/cl_eval")
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--curriculum", type=str, required=True)
    parser.add_argument("--eval_domains", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, default="CL_Eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--dump_raw_json", action="store_true")
    parser.add_argument("--preprocessed_eval_root", type=str, default=None)
    parser.add_argument("--per_domain_metrics", type=str, default="ppl,rougeL,bleu")

    args = parser.parse_args()

    set_seed(args.seed)

    curriculum = parse_curriculum_arg(args.curriculum)
    eval_domains = [d.strip() for d in args.eval_domains.split(",")] if args.eval_domains else None
    per_domain_metrics = [m.strip() for m in args.per_domain_metrics.split(",") if m.strip()]

    evaluator = CLEvaluatorCL(
        output_dir=args.output_dir,
        curriculum=curriculum,
        base_ckpt_path=args.base_ckpt,
        eval_domains=eval_domains,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gen_max_new_tokens=args.gen_max_new_tokens,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        log_raw_json=args.dump_raw_json,
        preprocessed_eval_root=args.preprocessed_eval_root,
        per_domain_metrics=per_domain_metrics,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
