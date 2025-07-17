import os, json, gc, random
import torch, numpy as np, wandb
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader,DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from eval_stability import Eval
from utils.metrics import ContinualLearningMetrics, calculate_metrics
from utils.domain_map import domain_info
from utils.dataset_loader import TextDatasetwchunk

# Seed 설정 함수
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def collate(batch):
    def to_tensor(x, dtype=torch.long):
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype)
        return torch.tensor(x, dtype=dtype)

    ids  = [to_tensor(b["input_ids"]) for b in batch]
    mask = [to_tensor(b["attention_mask"]) for b in batch]
    lbls = [to_tensor(b["labels"]) for b in batch]

    return dict(
        input_ids = pad_sequence(ids ,  batch_first=True, padding_value=0),
        attention_mask = pad_sequence(mask, batch_first=True, padding_value=0),
        labels = pad_sequence(lbls, batch_first=True, padding_value=-100))


class Trainer:
    def __init__(self, model_name, dataset_list, output_dir,
                 num_epochs=1, learning_rate=2e-5, seed=777,
                 max_length=512, chunk_size=128, batch_size=2):
        """
        Trainer 초기화
        :param model_name: 사전학습된 모델 이름
        :param dataset_list: 학습할 데이터셋 파일 경로 리스트
        :param output_dir: 모델 저장 디렉토리
        :param num_epochs: 학습 에폭 수
        :param learning_rate: 학습률
        :param seed: 랜덤 시드
        :param max_length: 입력 시퀀스 최대 길이
        :param chunk_size: 텍스트 청크 크기
        :param batch_size: 배치 크기
        """
        set_seed(seed)
        self.model_name = model_name
        self.dataset_list = dataset_list
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.max_length = max_length
        self.chunk_size = chunk_size

        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.domain_order = ['kor_medical', 'eng_medical', 'kor_legal', 'eng_legal']
        self.cl_metrics = ContinualLearningMetrics(num_tasks=len(self.domain_order))
        self.wandb_active = False

    def _build_loader(self, txt_path):
        ds = TextDatasetwchunk(txt_path, self.tokenizer,
                               max_length=self.max_length,
                               chunk_size=self.chunk_size)

        # world_size>1 일 때만 DistributedSampler 사용
        if self.accelerator.num_processes > 1:
            sampler = DistributedSampler(
                        ds,
                        num_replicas=self.accelerator.num_processes,
                        rank=self.accelerator.process_index,
                        shuffle=True,
                        drop_last=True)
            shuffle_flag = False   # Sampler가 셔플해 주므로
        else:
            sampler = None
            shuffle_flag = True

        return DataLoader(ds,
                          batch_size=self.batch_size,
                          sampler=sampler,
                          shuffle=shuffle_flag,
                          num_workers=4,
                          pin_memory=True,
                          persistent_workers=True,
                          collate_fn=collate)
    @staticmethod
    def _plasticity(before: float, after: float, eps=1e-8):
        return max(0., min(1., (before-after)/(before+eps)))
    
    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def _train_one(self, loader, ds_name):
        """단일 데이터셋 학습"""
        
        if not self.wandb_active and self.accelerator.is_main_process:
            # Process dataset name
            if 'new-medical-kor-dataset' in ds_name:
                wandb_dataset_name = 'kor-medical'
            elif 'guidline_medical' in ds_name:
                wandb_dataset_name = 'eng-medical'
            elif 'new-legal-kor-dataset' in ds_name:
                wandb_dataset_name = 'kor-legal'
            elif 'eng-new-legal-dataset' in ds_name:
                wandb_dataset_name = 'eng-legal'
            else: 
                wandb_dataset_name = ds_name

            wandb.init(project="kd-recyclable",
                       name=f"{wandb_dataset_name}_{self.model_name.replace('/','_')}",
                       config={"lr":self.lr,"model":self.model_name}, group="training")
            self.wandb_active = True

        elif not self.wandb_active:
            wandb.init(mode="disabled"); self.wandb_active=True
            self.wandb_initialized = True

        print(f"Training on dataset: {ds_name}")
        opt = Adam(self.model.parameters(), lr=self.lr)
        self.model, opt, loader = self.accelerator.prepare(self.model, opt, loader)
              
        gstep = 0
        for ep in range(self.num_epochs):
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(ep)
                
            self.model.train()

            bar = tqdm(loader,
                       disable=not self.accelerator.is_local_main_process,
                       desc=f"{ds_name} | ep {ep+1}")

            for batch in bar:
                gstep += 1
                with torch.no_grad():
                    loss_b = self.model(**batch).loss.item()

                out = self.model(**batch)
                self.accelerator.backward(out.loss)
                opt.step(); opt.zero_grad()

                loss_a = out.loss.item()
                plast  = self._plasticity(loss_b, loss_a)

                if self.accelerator.is_main_process:
                    wandb.log({"step":gstep,
                               f"{ds_name}/loss":loss_a,
                               f"{ds_name}/plastic":plast})
                bar.set_postfix(loss=f"{loss_a:.3f}", plast=f"{plast:.3f}")

            # checkpoint per-epoch
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                ck = os.path.join(self.output_dir, f"{ds_name}_ep{ep+1}")
                os.makedirs(ck, exist_ok=True)
                self.accelerator.unwrap_model(self.model).save_pretrained(ck)
                self.tokenizer.save_pretrained(ck)
                print("✓ saved", ck)

        if self.accelerator.is_main_process:
            wandb.finish(); self.wandb_active=False
        
    def _eval_cl_metrics(self, loader, domain_name, task_id):
        if not self.accelerator.is_main_process:
            return

        # 1) Continual-learning 지표 계산
        cl_dict = self.cl_metrics.compute_all_metrics(
                      model      = self.accelerator.unwrap_model(self.model),
                      current_task=task_id+1,
                      dataloader = loader,
                      criterion  = torch.nn.CrossEntropyLoss())

        # 2) wandb run (eval)
        wandb.init(project="kd-recyclable-eval",
                   name=f"eval_{domain_name}_t{task_id}", reinit=True)
        wandb.log(cl_dict); wandb.finish()

        # 3) JSON 저장
        ckpt = os.path.join(self.output_dir, f"checkpoint_after_{domain_name}")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "metrics.json"), "w") as fp:
            json.dump(cl_dict | {"task_id":task_id, "domain":domain_name}, fp, indent=2)
        # model save
        self.accelerator.unwrap_model(self.model).save_pretrained(ckpt)
        self.tokenizer.save_pretrained(ckpt)
        print("(｡・‧̫・｡)o🪄 eval-saved", ckpt)

    def train_across_datasets(self):
        """모든 데이터셋에 대해 순차적으로 학습하고 각 단계별로 평가 수행"""
        for task_id, txt in enumerate(self.dataset_list):
            ds_name = os.path.basename(txt).split('.')[0]
            loader  = self._build_loader(txt)

            print(f"\n▶ Task {task_id+1}/{len(self.dataset_list)} : {ds_name}")
            self._train_one(loader, ds_name)

            # ----- 평가 + 저장 -----
            self.accelerator.wait_for_everyone()
            self._eval_cl_metrics(loader, ds_name, task_id)

if __name__ == "__main__":
    trainer = Trainer(
        model_name="openai-community/gpt2-large",
        dataset_list=[
            "/home/infonet/sumin/cp4gm/utils/data_storage/new-medical-kor-dataset.txt",
            "/home/infonet/sumin/cp4gm/utils/data_storage/guidline_medical.txt",
            "/home/infonet/sumin/cp4gm/utils/data_storage/new-legal-kor-dataset.txt",
            "/home/infonet/sumin/cp4gm/utils/data_storage/eng-new-legal-dataset.txt"
        ],
        output_dir="/home/infonet/sumin/saved_model/recyclable/gpt2/",
        batch_size=1,
        num_epochs=5,
        learning_rate=2e-5,
    )
    trainer.load_model_and_tokenizer()
    trainer.train_across_datasets()
