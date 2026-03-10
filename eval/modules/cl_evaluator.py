import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np
import torch
import wandb
from tqdm import tqdm
import logging
import warnings

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM

from modules.eval_stability import EvalRunner

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEXT_METRICS = [
    "bleu",
    "rouge1", "rouge2", "rougeL",
    "meteor",
    "exact_match",
    "cosine_similarity",
    "jaccard_similarity",
    "groundedness",
    "token_precision", "token_recall", "token_f1",
    "token_precision_micro", "token_recall_micro", "token_f1_micro",
    "accuracy",
]

ALL_METRICS = ["ppl", "loss"] + TEXT_METRICS
LOWER_IS_BETTER = {"ppl", "loss"}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_special_tokens(tokenizer, model):
    add_dict = {}
    if tokenizer.eos_token is None:
        add_dict["eos_token"] = "</s>"
    if tokenizer.pad_token is None:
        add_dict["pad_token"] = "<|pad|>"

    if add_dict:
        tokenizer.add_special_tokens(add_dict)
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model, "generation_config") and model.generation_config is not None:
        gc = model.generation_config
        gc.pad_token_id = tokenizer.pad_token_id
        gc.eos_token_id = tokenizer.eos_token_id
        gc.do_sample = False

    return tokenizer, model


def load_model_tokenizer(ckpt_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        local_files_only=False,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        local_files_only=False,
        padding_side="left",
    )
    tokenizer, model = ensure_special_tokens(tokenizer, model)
    return model, tokenizer


def mean_finite(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

# CL Evaluator Class (RAW space)
class CLEvaluatorCL:
    """
    - No metric undergoes sign inversion/utility conversion.
    - BWT/FWT/AvgF is calculated as the "raw value difference".
    """

    def __init__(
        self,
        output_dir: str,
        curriculum: List[Tuple[str, str, int]],
        base_ckpt_path: str,
        eval_domains: Optional[List[str]] = None,
        batch_size: int = 8,
        max_length: int = 512,
        gen_max_new_tokens: int = 128,
        project: str = "CL_Eval",
        run_name: Optional[str] = None,
        log_raw_json: bool = True,
        preprocessed_eval_root: Optional[str] = None,
        per_domain_metrics: Optional[List[str]] = None,
        shot_type: str = "zero-shot",
        seed: int = 777,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.curriculum = curriculum
        self.base_ckpt_path = base_ckpt_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.gen_max_new_tokens = gen_max_new_tokens
        self.preprocessed_eval_root = preprocessed_eval_root

        # A metric subset to summarize detailed CL metrics by domain
        self.per_domain_metrics = per_domain_metrics or ["ppl", "rougeL", "bleu", "accuracy"]

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_type="full_state_dict",
            backward_prefetch="BACKWARD_POST",
            forward_prefetch=False,
        )
        self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        self.shot_type = shot_type
        self.seed = seed
        self.project = project
        self.run_name = run_name or f"CL_run_{os.path.basename(base_ckpt_path)}"
        self.log_raw_json = log_raw_json

        trained_domains = [d for d, _, _ in curriculum]
        all_eval_domains = trained_domains + (eval_domains or [])
        self.unique_domains = list(dict.fromkeys(all_eval_domains))

        # R[m][t][domain]: about metric m, t step eval(step) domain performance
        self.R: Dict[str, List[Dict[str, float]]] = {m: [] for m in ALL_METRICS}

        # step
        self.step_info: Dict[int, Tuple[str, int, str]] = {}
        self.first_learned_step: Dict[str, int] = {}

        for step_idx, (domain, ckpt_path, epoch) in enumerate(self.curriculum, start=1):
            self.step_info[step_idx] = (domain, epoch, ckpt_path)
            if domain not in self.first_learned_step:
                self.first_learned_step[domain] = step_idx

        # base(step0)에서의 domain 성능 (FWT baseline)
        self.b_initial: Dict[str, Dict[str, float]] = {}

        if self.accelerator.is_main_process:
            
            self.wandb_run = wandb.init(project=self.project, name=self.run_name, reinit=True)
            self.wandb_run.define_metric("cl_step")
            self.wandb_run.define_metric("eval/*", step_metric="cl_step")
            
            wandb.config.update({
                "base_ckpt": self.base_ckpt_path,
                "curriculum": [(d, epoch, os.path.basename(p)) for d, p, epoch in curriculum],
                "domains": self.unique_domains,
                "metrics": ALL_METRICS,
                "lower_is_better": sorted(list(LOWER_IS_BETTER)),
                "raw_space_metrics": True,
                "shot_type": self.shot_type,
                "seed": self.seed,
            })
        else:
            self.wandb_run = None

    # ---------- evaluation ----------
    def _evaluate_checkpoint_on_all(self, ckpt_path: str, step_idx: int, step_tag: str):
        model, tokenizer = load_model_tokenizer(ckpt_path)
        model.eval()
        model = self.accelerator.prepare(model)

        eval_runner = EvalRunner(
            model=model,
            tokenizer=tokenizer,
            accelerator=self.accelerator,
            batch_size=self.batch_size,
            max_length=self.max_length,
            gen_max_new_tokens=self.gen_max_new_tokens,
            eval_split="train",
            preprocessed_data_root=self.preprocessed_eval_root,
            wandb_run=self.wandb_run,
            cl_step_idx=step_idx,
            step_tag=step_tag,
            shot_type=self.shot_type,
            seed=self.seed,
        )

        results: Dict[str, Dict[str, float]] = {}

        domain_pbar = tqdm(
            self.unique_domains,
            desc=f"[Eval] {step_tag} (step={step_idx})",
            disable=not self.accelerator.is_main_process,
            leave=False,
            dynamic_ncols=True,
        )

        for domain in domain_pbar:
            metrics = eval_runner.evaluate(domain=domain)

            # (1) domain별 메트릭 로깅 (commit=False, step_avg에서 commit)
            if self.accelerator.is_main_process and self.wandb_run is not None:
                log_payload = {
                    f"eval/{domain}/{k}": safe_float(v)
                    for k, v in metrics.items()
                    if k in ALL_METRICS
                }
                wandb.log(log_payload, step=step_idx, commit=False)

            # (2) R에 저장할 값
            results[domain] = {m: safe_float(metrics.get(m, np.nan)) for m in ALL_METRICS}

            if self.accelerator.is_main_process:
                show = {}
                for check_m in ["ppl", "rougeL", "bleu", "accuracy"]:
                    if check_m in metrics and np.isfinite(metrics[check_m]):
                        if check_m == "ppl":
                            show[check_m] = f"{metrics[check_m]:.2f}"
                        else:
                            show[check_m] = f"{metrics[check_m]:.4f}"
                if show:
                    domain_pbar.set_postfix(show)

        # (3) step 평균 로깅 (commit=True)
        if self.accelerator.is_main_process and self.wandb_run is not None:
            step_avg_payload = {}
            for m in ALL_METRICS:
                vals = [
                    results[d][m] for d in self.unique_domains
                    if d in results and np.isfinite(results[d].get(m, np.nan))
                ]
                step_avg_payload[f"step_avg/{m}"] = mean_finite(vals)
            wandb.log(step_avg_payload, step=step_idx, commit=True)

        del eval_runner, model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

        return results

    # R-matrix append
    def _append_to_R(self, step_result: Dict[str, Dict[str, float]]):
        for m in ALL_METRICS:
            self.R[m].append({
                d: safe_float(step_result.get(d, {}).get(m, np.nan))
                for d in self.unique_domains
            })

    # RAW-space CL metrics at arbitrary step
    def _compute_cl_metrics_at_step_raw(self, upto_step: int):
        """
        upto_step: row index(=Eval step) of R. step0(base) add.
        learned_domains: first_learned_step <= upto_step 인 도메인만 학습된 task로 간주.
        """
        learned_domains = [d for d, s in self.first_learned_step.items() if s <= upto_step]
        k = len(learned_domains)

        ACC_avg, BWT_avg, FWT_avg, AVGF_avg = {}, {}, {}, {}
        BWT_domain = {m: {} for m in ALL_METRICS}
        FWT_domain = {m: {} for m in ALL_METRICS}
        AVGF_domain = {m: {} for m in ALL_METRICS}

        if k == 0:
            for m in ALL_METRICS:
                ACC_avg[m] = np.nan
                BWT_avg[m] = np.nan
                FWT_avg[m] = np.nan
                AVGF_avg[m] = np.nan
            return {
                "learned_domains": learned_domains,
                "ACC_avg": ACC_avg,
                "BWT_avg": BWT_avg,
                "FWT_avg": FWT_avg,
                "AvgF_avg": AVGF_avg,
                "BWT_per_domain": BWT_domain,
                "FWT_per_domain": FWT_domain,
                "AvgF_per_domain": AVGF_domain,
            }

        for m in ALL_METRICS:
            # ACC_avg: 현재 step에서 learned domains 평균 (raw)
            acc_vals = [
                self.R[m][upto_step].get(d, np.nan)
                for d in learned_domains
                if np.isfinite(self.R[m][upto_step].get(d, np.nan))
            ]
            ACC_avg[m] = mean_finite(acc_vals)

            # BWT_raw: Average_{past task} ( R_{upto_step,i} - R_{t_i,i} )
            bwt_vals = []
            for d in learned_domains:
                t_i = self.first_learned_step[d]
                if t_i < upto_step:
                    v_final = self.R[m][upto_step].get(d, np.nan)
                    v_after = self.R[m][t_i].get(d, np.nan)
                    if np.isfinite(v_final) and np.isfinite(v_after):
                        bwt = float(v_final - v_after)
                        bwt_vals.append(bwt)
                        BWT_domain[m][d] = bwt
            BWT_avg[m] = mean_finite(bwt_vals) if bwt_vals else np.nan

            # AvgF_raw: Average_{past task} ( max_{t in [t_i..upto_step-1]} R_{t,i} - R_{upto_step,i} )
            avgf_vals = []
            for d in learned_domains:
                t_i = self.first_learned_step[d]
                if t_i <= upto_step - 1:
                    v_final = self.R[m][upto_step].get(d, np.nan)
                    if not np.isfinite(v_final):
                        continue
                    past_vals = []
                    for t in range(t_i, upto_step):  # upto_step 제외
                        v_t = self.R[m][t].get(d, np.nan)
                        if np.isfinite(v_t):
                            past_vals.append(v_t)
                    if past_vals:
                        v_past_max = float(np.max(past_vals))
                        avgf = float(v_past_max - v_final)
                        avgf_vals.append(avgf)
                        AVGF_domain[m][d] = avgf
            AVGF_avg[m] = mean_finite(avgf_vals) if avgf_vals else np.nan

            # FWT_raw: ( R_{t_i-1,i} - b_initial[i] )
            fwt_vals = []
            for d in learned_domains:
                t_i = self.first_learned_step[d]
                if (t_i - 1) >= 0 and d in self.b_initial:
                    v_before = self.R[m][t_i - 1].get(d, np.nan)
                    v_init = self.b_initial.get(d, {}).get(m, np.nan)
                    if np.isfinite(v_before) and np.isfinite(v_init):
                        fwt = float(v_before - v_init)
                        fwt_vals.append(fwt)
                        FWT_domain[m][d] = fwt
            FWT_avg[m] = mean_finite(fwt_vals) if fwt_vals else np.nan

        return {
            "learned_domains": learned_domains,
            "ACC_avg": ACC_avg,
            "BWT_avg": BWT_avg,
            "FWT_avg": FWT_avg,
            "AvgF_avg": AVGF_avg,
            "BWT_per_domain": BWT_domain,
            "FWT_per_domain": FWT_domain,
            "AvgF_per_domain": AVGF_domain,
        }

    # step-wise CL logging
    def _log_cl_metrics_stepwise(self, step_idx: int, cl: Optional[Dict[str, Any]] = None):
        if not (self.accelerator.is_main_process and self.wandb_run is not None):
            return

        if cl is None:
            cl = self._compute_cl_metrics_at_step_raw(upto_step=step_idx)

        payload = {
            "cl/num_learned_tasks": len(cl["learned_domains"]),
            "cl/raw_space": 1,
        }

        # Average CL Metrics
        for m in ALL_METRICS:
            payload[f"cl/ACC_avg/{m}"] = cl["ACC_avg"].get(m, np.nan)
            payload[f"cl/BWT_avg/{m}"] = cl["BWT_avg"].get(m, np.nan)
            payload[f"cl/FWT_avg/{m}"] = cl["FWT_avg"].get(m, np.nan)
            payload[f"cl/AvgF_avg/{m}"] = cl["AvgF_avg"].get(m, np.nan)

            # 도메인별 BWT/FWT/AvgF
            for d, val in cl["BWT_per_domain"].get(m, {}).items():
                payload[f"cl/BWT_per_domain/{m}/{d}"] = val
            for d, val in cl["FWT_per_domain"].get(m, {}).items():
                payload[f"cl/FWT_per_domain/{m}/{d}"] = val
            for d, val in cl["AvgF_per_domain"].get(m, {}).items():
                payload[f"cl/AvgF_per_domain/{m}/{d}"] = val

        wandb.log(payload, step=step_idx, commit=True)

    # JSON dump 
    def _maybe_dump_json(self, step_idx: int, data: Dict[str, Any], tag: str):
        if not self.log_raw_json:
            return
        p = os.path.join(self.output_dir, f"raw_step_{step_idx}_{tag}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # run
    def run(self):
        # Step 0: base
        self.accelerator.print("\n========== STEP 0: BASE ==========")
        step0_res = self._evaluate_checkpoint_on_all(self.base_ckpt_path, 0, "base")
        self._append_to_R(step0_res)

        # FWT baseline
        for d in self.unique_domains:
            self.b_initial[d] = step0_res.get(d, {})

        # step0 CL cal. + save and logging
        cl0 = self._compute_cl_metrics_at_step_raw(upto_step=0)

        if self.accelerator.is_main_process:
            self._maybe_dump_json(0, {"eval": step0_res, "cl": cl0}, "base")

        self._log_cl_metrics_stepwise(step_idx=0, cl=cl0)

        # curriculum steps
        cur_pbar = tqdm(
            enumerate(self.curriculum, start=1),
            total=len(self.curriculum),
            desc="[Total Steps]",
            disable=not self.accelerator.is_main_process,
        )

        for step_idx, (domain, ckpt_path, epoch) in cur_pbar:
            tag = f"after_{domain}_epoch_{epoch}"
            self.accelerator.print(f"\n--- Step {step_idx}: {tag} ---")

            step_res = self._evaluate_checkpoint_on_all(ckpt_path, step_idx, tag)
            self._append_to_R(step_res)

            cl_step = self._compute_cl_metrics_at_step_raw(upto_step=step_idx)

            if self.accelerator.is_main_process:
                # SAVE eval + CL
                self._maybe_dump_json(step_idx, {"eval": step_res, "cl": cl_step}, tag)

            # step-wise CL metrics logging
            self._log_cl_metrics_stepwise(step_idx=step_idx, cl=cl_step)

        if self.accelerator.is_main_process:
            self._finalize_and_log()

    #  finalize and logging
    def _finalize_and_log(self):
        final_step = len(self.R[ALL_METRICS[0]]) - 1
        cl_final = self._compute_cl_metrics_at_step_raw(upto_step=final_step)

        summary = {
            "raw_space_metrics": True,
            "lower_is_better": sorted(list(LOWER_IS_BETTER)),
            "domains": self.unique_domains,
            "first_learned_step": self.first_learned_step,
            "num_eval_points": len(self.R[ALL_METRICS[0]]),
            "R_matrix": self.R,
            "b_initial": self.b_initial,
            "final": {
                "learned_domains": cl_final["learned_domains"],
                "ACC_avg": cl_final["ACC_avg"],
                "BWT_avg": cl_final["BWT_avg"],
                "FWT_avg": cl_final["FWT_avg"],
                "AvgF_avg": cl_final["AvgF_avg"],
                "BWT_per_domain": cl_final["BWT_per_domain"],
                "FWT_per_domain": cl_final["FWT_per_domain"],
                "AvgF_per_domain": cl_final["AvgF_per_domain"],
            },
        }

        p = os.path.join(self.output_dir, "cl_summary.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.accelerator.print(f"\n✓ Saved summary: {p}")

        if self.wandb_run is None:
            return

        # (1) wandb.summary update
        for m in ALL_METRICS:
            self.wandb_run.summary[f"final/ACC_avg/{m}"] = cl_final["ACC_avg"].get(m, np.nan)
            self.wandb_run.summary[f"final/BWT_avg/{m}"] = cl_final["BWT_avg"].get(m, np.nan)
            self.wandb_run.summary[f"final/FWT_avg/{m}"] = cl_final["FWT_avg"].get(m, np.nan)
            self.wandb_run.summary[f"final/AvgF_avg/{m}"] = cl_final["AvgF_avg"].get(m, np.nan)

        # (2) Domain-specific final CL metric summary
        for m in self.per_domain_metrics:
            for d, val in cl_final["BWT_per_domain"].get(m, {}).items():
                self.wandb_run.summary[f"final/BWT_per_domain/{d}/{m}"] = float(val)
            for d, val in cl_final["FWT_per_domain"].get(m, {}).items():
                self.wandb_run.summary[f"final/FWT_per_domain/{d}/{m}"] = float(val)
            for d, val in cl_final["AvgF_per_domain"].get(m, {}).items():
                self.wandb_run.summary[f"final/AvgF_per_domain/{d}/{m}"] = float(val)

        # (3) Also record the final scalar in the last step to wandb.log.
        final_payload = {
            "final/num_learned_tasks": len(cl_final["learned_domains"]),
            "final/raw_space": 1,
        }
        for m in ALL_METRICS:
            final_payload[f"final/ACC_avg/{m}"] = cl_final["ACC_avg"].get(m, np.nan)
            final_payload[f"final/BWT_avg/{m}"] = cl_final["BWT_avg"].get(m, np.nan)
            final_payload[f"final/FWT_avg/{m}"] = cl_final["FWT_avg"].get(m, np.nan)
            final_payload[f"final/AvgF_avg/{m}"] = cl_final["AvgF_avg"].get(m, np.nan)

        wandb.log(final_payload, step=final_step, commit=True)
        self.wandb_run.finish()
