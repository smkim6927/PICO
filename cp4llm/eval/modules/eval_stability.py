import math
import re
from typing import Optional, Dict, List, Tuple
import wandb

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from modules.metrics import calculate_metrics
from collections import Counter

import sys
sys.path.append('/home/jovyan/sumin_data/cp4llm')
from utils.domain_map import get_collate_fn, get_processed_dataset


def _extract_choice(text: str) -> Optional[str]:
    """
    Robustly extract only the selected option letter (A–E) from the text.
    - Normalize "E", "E.", "E:" to "E"
    - Handle cases like "The answer is E." using regular expressions
    """
    if text is None:
        return None
    s = text.strip()

    1) First part of the colon (e.g., "E: Subarachnoid hemorrhage" → "E")
    first_part = s.split(":", 1)[0].strip()

    # 2) [A-E] Regular expression to find a single character
    m = re.search(r"\b([A-E])\b", first_part, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) fallback: Use if the first letter is A–E
    if first_part and first_part[0].upper() in "ABCDE":
        return first_part[0].upper()

    return None


class EvalRunner:
    def __init__(
        self,
        model,
        tokenizer,
        accelerator: Accelerator,
        batch_size: int = 8,
        max_length: int = 512,
        gen_max_new_tokens: int = 128,
        eval_split: str = "train",
        preprocessed_data_root: Optional[str] = None,
        wandb_run=None,
        cl_step_idx: Optional[int] = None,
        step_tag: str = "",
        shot_type = "zero-shot",
        seed: int = 777,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.max_length = max_length
        self.gen_max_new_tokens = gen_max_new_tokens
        self.eval_split = eval_split
        self.preprocessed_data_root = preprocessed_data_root

        self.tokenizer.padding_side = "left"

        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        pad_id = int(self.tokenizer.pad_token_id or 0)
        self.collate_fn = get_collate_fn(collate_type="smart", pad_id=pad_id)

        self.wandb_run = wandb_run
        self.cl_step_idx = cl_step_idx
        self.step_tag = step_tag
        self.shot_type = shot_type
        self.seed = seed

    def _load_dataset_for_domain(self, domain: str):
        return get_processed_dataset(
            domain=domain,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mode="evaluation",
            split=self.eval_split,
            shot_type=self.shot_type,
            preprocessed_root=self.preprocessed_data_root,
            seed=self.seed,
        )

    # Token-overlap P/R/F1
    def _encode_tokens(self, text: str) -> List[int]:
        """
        Convert the string into a sequence of subword token IDs based on the 'model tokenizer'.
        - Use only pure tokens by setting add_special_tokens=False.
        """
        if text is None:
            return []
        s = text.strip()
        if not s:
            return []
        return self.tokenizer.encode(s, add_special_tokens=False)

    @staticmethod
    def _overlap_count(pred_toks: List[int], gold_toks: List[int]) -> Tuple[int, int, int]:
        """
        Multi-set (overlap allowed):
          common = Counter(pred) & Counter(gold)
          num_same = sum(common.values())
        """
        if len(pred_toks) == 0 or len(gold_toks) == 0:
            return 0, len(pred_toks), len(gold_toks)
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        return num_same, len(pred_toks), len(gold_toks)

    @staticmethod
    def _prf(num_same: int, pred_len: int, gold_len: int) -> Tuple[float, float, float]:
        """
        precision = num_same / pred_len
        recall    = num_same / gold_len
        f1        = 2PR/(P+R)
        """
        if pred_len == 0 and gold_len == 0:
            return 1.0, 1.0, 1.0
        if pred_len == 0 or gold_len == 0 or num_same == 0:
            return 0.0, 0.0, 0.0

        p = float(num_same) / float(pred_len)
        r = float(num_same) / float(gold_len)
        f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def _compute_token_metrics(self, predictions: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Return:
          - token_precision, token_recall, token_f1 (PRF average per example)
          - token_precision_micro, token_recall_micro, token_f1_micro (PRF based on global count summation)
        """
        if not predictions or not labels or len(predictions) != len(labels):
            return {
                "token_precision": float("nan"),
                "token_recall": float("nan"),
                "token_f1": float("nan"),
                "token_precision_micro": float("nan"),
                "token_recall_micro": float("nan"),
                "token_f1_micro": float("nan"),
            }

        ps: List[float] = []
        rs: List[float] = []
        f1s: List[float] = []

        total_same = 0
        total_pred = 0
        total_gold = 0

        for pred_str, gold_str in zip(predictions, labels):
            pred_toks = self._encode_tokens(pred_str)
            gold_toks = self._encode_tokens(gold_str)

            num_same, pred_len, gold_len = self._overlap_count(pred_toks, gold_toks)
            p, r, f1 = self._prf(num_same, pred_len, gold_len)

            ps.append(p)
            rs.append(r)
            f1s.append(f1)

            total_same += int(num_same)
            total_pred += int(pred_len)
            total_gold += int(gold_len)

        token_precision = float(sum(ps) / len(ps)) if ps else float("nan")
        token_recall = float(sum(rs) / len(rs)) if rs else float("nan")
        token_f1 = float(sum(f1s) / len(f1s)) if f1s else float("nan")

        p_micro, r_micro, f1_micro = self._prf(total_same, total_pred, total_gold)

        return {
            "token_precision": token_precision,
            "token_recall": token_recall,
            "token_f1": token_f1,
            "token_precision_micro": float(p_micro),
            "token_recall_micro": float(r_micro),
            "token_f1_micro": float(f1_micro),
        }

    def _compute_accuracy(self, predictions: List[str], labels: List[str]) -> float:
        """
        Accuracy for multiple-choice question-answering based on options (A–E).
        - Parsing one character from each option in both label/pred using _extract_choice for comparison.
        """
        if (not predictions) or (not labels) or (len(predictions) != len(labels)):
            return float("nan")

        correct = 0
        total = 0

        for pred, label in zip(predictions, labels):
            gold = _extract_choice(label)
            pred_c = _extract_choice(pred)

            If you can't pick either choice, skip it.
            if gold is None or pred_c is None:
                continue

            total += 1
            if gold == pred_c:
                correct += 1

        if total == 0:
            return float("nan")
        return float(correct) / float(total)

    # Main evaluate
    def evaluate(self, domain: str) -> Dict[str, float]:
        dataset = self._load_dataset_for_domain(domain)

        # Handling Empty Datasets
        if len(dataset) == 0:
            return {
                "ppl": float("nan"),
                "loss": float("nan"),
                "rouge1": float("nan"),
                "rouge2": float("nan"),
                "rougeL": float("nan"),
                "bleu": float("nan"),
                "meteor": float("nan"),
                "exact_match": float("nan"),
                "groundedness": float("nan"),
                "cosine_similarity": float("nan"),
                "jaccard_similarity": float("nan"),
                "token_precision": float("nan"),
                "token_recall": float("nan"),
                "token_f1": float("nan"),
                "token_precision_micro": float("nan"),
                "token_recall_micro": float("nan"),
                "token_f1_micro": float("nan"),
                "accuracy": float("nan"),
            }

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        dataloader = self.accelerator.prepare(dataloader)

        self.model.eval()

        total_nll = 0.0
        total_tokens = 0

        predictions: List[str] = []
        labels_text: List[str] = []
        inputs_text: List[str] = []

        with torch.no_grad():
            it = tqdm(
                dataloader,
                desc=f"[Eval:{domain}]",
                disable=not self.accelerator.is_main_process,
                leave=False,
                dynamic_ncols=True,
            )
            batch_idx = 0

            for batch in it:
                device = next(self.model.parameters()).device

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True) if "labels" in batch else None

                # 1) loss / ppl Cumulative 
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs.logits

                loss = self.ce_loss(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

                loss_g = self.accelerator.gather_for_metrics(loss.detach())
                tok_cnt = (labels != -100).sum()
                tok_g = self.accelerator.gather_for_metrics(tok_cnt.detach())

                batch_total_nll = loss_g.sum().item()
                batch_total_tokens = tok_g.sum().item()

                total_nll += batch_total_nll
                total_tokens += batch_total_tokens

                # Batch unit ppl/loss logging
                if (
                    self.accelerator.is_main_process
                    and self.wandb_run is not None
                    and batch_total_tokens > 0
                ):
                    batch_avg_nll = batch_total_nll / float(batch_total_tokens)
                    batch_ppl = math.exp(batch_avg_nll)
                    
                    cl_step = int(self.cl_step_idx if self.cl_step_idx is not None else -1)
                    eval_batch_step = cl_step * 1_000_000 + int(batch_idx)

                    self.wandb_run.log(
                        {
                            "cl_step": cl_step,
                            "eval_batch_step": eval_batch_step,
                            f"eval_batch/{domain}/loss": float(batch_avg_nll),
                            f"eval_batch/{domain}/ppl": float(batch_ppl),
                            "eval_batch_idx": int(batch_idx),
                            "eval_domain": domain,
                            "checkpoint_tag": self.step_tag,
                        }
                    )


                # 2) Reconstructing left-padding for the generate() call
                gen_model = self.model
                if not hasattr(gen_model, "generate") and hasattr(gen_model, "base_model"):
                    gen_model = gen_model.base_model
                if not hasattr(gen_model, "generate"):
                    raise AttributeError(
                        f"Model of type {type(self.model)} has no 'generate' method "
                        f"and no inner 'base_model.generate'."
                    )

                seq_list = []
                for ids, mask in zip(input_ids, attention_mask):
                    valid_ids = ids[mask == 1]
                    seq_list.append(valid_ids)

                pad_id = int(self.tokenizer.pad_token_id or 0)
                max_len = max(seq.size(0) for seq in seq_list)
                bsz = len(seq_list)

                new_input_ids = input_ids.new_full((bsz, max_len), fill_value=pad_id)
                new_attention_mask = attention_mask.new_zeros((bsz, max_len))

                for i, seq in enumerate(seq_list):
                    L = seq.size(0)
                    new_input_ids[i, -L:] = seq
                    new_attention_mask[i, -L:] = 1

                gen_out = gen_model.generate(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    max_new_tokens=self.gen_max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # 3) gather & decoding
                pred_ids_g = self.accelerator.gather_for_metrics(gen_out)
                label_ids_g = self.accelerator.gather_for_metrics(labels)
                input_ids_g = self.accelerator.gather_for_metrics(new_input_ids)
                attn_g = self.accelerator.gather_for_metrics(new_attention_mask)

                if self.accelerator.is_main_process:
                    pred_ids_g = pred_ids_g.cpu()
                    label_ids_g = label_ids_g.cpu()
                    input_ids_g = input_ids_g.cpu()
                    attn_g = attn_g.cpu()

                    for pi, li, ii, am in zip(pred_ids_g, label_ids_g, input_ids_g, attn_g):
                        valid = li[li != -100]
                        label_str = (
                            self.tokenizer.decode(valid, skip_special_tokens=True).strip()
                            if valid.numel() > 0 else ""
                        )
                        ii_valid = ii[am == 1]
                        input_str = self.tokenizer.decode(ii_valid, skip_special_tokens=True).strip()
                        pred_str = self.tokenizer.decode(pi, skip_special_tokens=True).strip()

                        labels_text.append(label_str)
                        inputs_text.append(input_str)
                        predictions.append(pred_str)

                batch_idx += 1

        # 4) Final metric calculation (only in the main process)
        metrics = None

        if self.accelerator.is_main_process:
            num_examples = len(predictions)

            if num_examples == 0 or total_tokens == 0:
                metrics = {
                    "ppl": float("nan"),
                    "loss": float("nan"),
                    "rouge1": float("nan"),
                    "rouge2": float("nan"),
                    "rougeL": float("nan"),
                    "bleu": float("nan"),
                    "meteor": float("nan"),
                    "exact_match": float("nan"),
                    "groundedness": float("nan"),
                    "cosine_similarity": float("nan"),
                    "jaccard_similarity": float("nan"),
                    "token_precision": float("nan"),
                    "token_recall": float("nan"),
                    "token_f1": float("nan"),
                    "token_precision_micro": float("nan"),
                    "token_recall_micro": float("nan"),
                    "token_f1_micro": float("nan"),
                    "accuracy": float("nan"),
                }
            else:
                dummy_losses = [0.0] * num_examples
                text_metrics = calculate_metrics(
                    predictions=predictions,
                    labels=labels_text,
                    losses=dummy_losses,
                    inputs=inputs_text,
                )

                token_metrics = self._compute_token_metrics(
                    predictions=predictions,
                    labels=labels_text,
                )

                avg_nll = total_nll / float(total_tokens)
                ppl = math.exp(avg_nll)

                metrics = dict(text_metrics)
                metrics.update(token_metrics)
                metrics["ppl"] = float(ppl)
                metrics["loss"] = float(avg_nll)

                # Accuracy for multiple-choice QA (especially eng_medical)
                acc = self._compute_accuracy(predictions, labels_text)
                metrics["accuracy"] = acc

        metrics_list = [metrics]
        broadcast_object_list(metrics_list)
        return metrics_list[0]
