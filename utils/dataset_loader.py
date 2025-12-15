from datasets import load_dataset
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import os
from ftplib import FTP
from tqdm import tqdm
import io, codecs, chardet

def robust_readlines(path, min_len=3):
    """
    ① 1 KB 샘플로 인코딩 추정 → ② 실패 시 latin-1 fallback
    ③ 최종적으로 errors='replace' 로 무조건 디코딩
    """
    with open(path, 'rb') as bf:
        raw = bf.read()                       # 한 번만 read, 메모리 충분
    enc_guess = chardet.detect(raw[:2048])['encoding'] or 'utf-8'
    txt = raw.decode(enc_guess, errors='replace')
    lines = [l.strip() for l in txt.splitlines() if len(l.strip())>=min_len]
    return lines


class GuidelinesCopora:
    def __init__(self, tokenizer=None, cache_dir="/home/jovyan/sumin_data/custom_datasets/"):
        """
        Load and merge the 'epfl-llm/guidelines' dataset.
        :param tokenizer: Pretrained tokenizer.
        :param cache_dir: Directory to cache datasets.
        """
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.texts = []  # Store combined texts

    def load_and_combine(self, save_txt_path=None):
        # Load the dataset
        dataset = load_dataset(
            "epfl-llm/guidelines",
            cache_dir=self.cache_dir
        )

        # Combine texts from the dataset
        for data in tqdm(dataset['train'], desc="Processing guidelines"):
            if "clean_text" in data:
                text = data['clean_text']
                if self.tokenizer:
                    text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
                self.texts.append(text)

        print(f"Total texts combined: {len(self.texts)}")

        # Save to a text file if save_txt_path is provided
        if save_txt_path:
            self.save_to_txt(save_txt_path)

        return self.texts

    def save_to_txt(self, file_path):
        """
        Save the combined texts to a .txt file.
        :param file_path: Path to save the .txt file.
        """
        print(f"Saving combined texts to {file_path}...")
        with open(file_path, "w", encoding="utf-8") as f:
            for text in self.texts:
                f.write(text + "\n")
        print(f"Texts successfully saved to {file_path}")



class LawCopora:
    def __init__(self, domain="law", tokenizer=None, cache_dir="/home/jovyan/sumin_data/custom_datasets/"):
        """
        Load and merge datasets from the specified domain.
        :param domain: Dataset domain name.
        :param tokenizer: Pretrained tokenizer.
        :param cache_dir: Directory to cache datasets.
        """
        self.domain = domain
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.texts = []  # Store combined texts

    def load_and_combine(self, save_txt_path=None):
        DOMAIN_TO_DATASETS = {
            "law": ["cc_casebooks", "exam_outlines"]
                    }

        # Combine texts from all datasets
        for name in tqdm(DOMAIN_TO_DATASETS[self.domain], desc="Loading datasets"):
            dataset = load_dataset(
                "pile-of-law/pile-of-law", name,
                cache_dir=self.cache_dir,
                data_dir=self.cache_dir,
                trust_remote_code=True
            )
            for data in tqdm(dataset['train'], desc=f"Processing {name}"):
                if "text" in data:
                    self.texts.append(self.tokenizer.bos_token + data['text'] + self.tokenizer.eos_token)

        print(f"Total texts combined: {len(self.texts)}")

        # Save to a text file if save_txt_path is provided
        if save_txt_path:
            self.save_to_txt(save_txt_path)

        return self.texts

    def save_to_txt(self, file_path):
        """
        Save the combined texts to a .txt file.
        :param file_path: Path to save the .txt file.
        """
        print(f"Saving combined texts to {file_path}...")
        with open(file_path, "w", encoding="utf-8") as f:
            for text in self.texts:
                f.write(text + "\n")
        print(f"Texts successfully saved to {file_path}")


class TextDatasetwchunk(Dataset):
    def __init__(
        self,
        txt_file=None,
        tokenizer=None,
        max_length=256,
        chunk_size=64,
        pseudo_labels=None  # 새 매개변수(초기 pseudo-label이 있다면 전달)
    ):
        """
        Initialize the TextDataset.
        
        Args:
            txt_file (str): Path to the .txt file containing text data.
            tokenizer: Pretrained tokenizer (HuggingFace 등).
            max_length (int): Maximum token length for each data point.
            chunk_size (int): Tokenization chunk size.
            pseudo_labels (List[Any] or None): 사전에 준비한 pseudo-label(길이 = 텍스트 개수)
                                              e.g. [cluster_id_0, cluster_id_1, ...]
                                              만약 없으면 None
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.texts = robust_readlines(txt_file)

        # pseudo-label 저장용
        # 길이 == len(self.texts) 가 되도록 유지해야 함
        self._pseudo_labels = pseudo_labels if pseudo_labels is not None else []

    def load_from_txt(self, file_path):
        """
        Load texts from a .txt file.
        """
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(self.texts)} texts.")

        # 텍스트를 새로 읽었으니 pseudo_labels 리스트도 초기화 또는 길이 체크
        if len(self._pseudo_labels) != len(self.texts):
            # 필요하다면 길이를 맞추거나 초기화
            self._pseudo_labels = [None] * len(self.texts)

    def attach_pseudo_labels(self, pseudo_labels_list):
        """
        외부(Trainer 등)에서 생성한 pseudo-label 리스트를 붙인다.
        pseudo_labels_list: List[Any], 길이는 self.__len__()와 동일해야 함.
        """
        if len(pseudo_labels_list) != len(self.texts):
            print(f"Warning: pseudo_labels_list 길이({len(pseudo_labels_list)}) != texts 길이({len(self.texts)})")
            # 상황에 맞게 처리 (여기선 잘린 부분만 반영)
            min_len = min(len(pseudo_labels_list), len(self.texts))
            self._pseudo_labels = pseudo_labels_list[:min_len] + [None]*(len(self.texts)-min_len)
        else:
            self._pseudo_labels = pseudo_labels_list

        print(f"Attached {len(self._pseudo_labels)} pseudo-labels to dataset.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        # Tokenization
        inputs = self.tokenizer(
            chunks,
            return_tensors="pt",
            max_length=self.chunk_size,
            truncation=True,
            padding="max_length"
        )
        # (batch_size * chunk_size,) -> flatten -> max_length까지 자름
        input_ids = inputs.input_ids.view(-1)[: self.max_length]
        attention_mask = inputs.attention_mask.view(-1)[: self.max_length]

        # 기본적으로 causal LM에서는 labels = input_ids 그대로
        labels = input_ids.clone()

        # pseudo_label이 있는 경우 가져오기
        pseudo_label = None
        if 0 <= idx < len(self._pseudo_labels):
            pseudo_label = self._pseudo_labels[idx]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        # pseudo_label이 존재하면 배치에 추가
        if pseudo_label is not None:
            # pseudo_label을 텐서로 변환(문장 단위 클러스터면 scalar, 토큰 단위면 shape 맞춰야 함)
            # 여기선 scalar 예시:
            if not torch.is_tensor(pseudo_label):
                pseudo_label = torch.tensor(pseudo_label, dtype=torch.long)
            output["pseudo_label"] = pseudo_label

        return output

class TextDataset(Dataset):
    def __init__(
        self,
        txt_file=None,
        tokenizer=None,
        max_length=8192,
        pseudo_labels=None  # 새 매개변수(초기 pseudo-label이 있다면 전달)
    ):
        """
        Initialize the TextDataset.
        
        Args:
            txt_file (str): Path to the .txt file containing text data.
            tokenizer: Pretrained tokenizer (HuggingFace 등).
            max_length (int): Maximum token length for each data point.
            pseudo_labels (List[Any] or None): 사전에 준비한 pseudo-label(길이 = 텍스트 개수)
                                              e.g. [cluster_id_0, cluster_id_1, ...]
                                              만약 없으면 None
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []

        # pseudo-label 저장용
        # 길이 == len(self.texts) 가 되도록 유지해야 함
        self._pseudo_labels = pseudo_labels if pseudo_labels is not None else []

        if txt_file:
            self.load_from_txt(txt_file)

    def load_from_txt(self, file_path):
        """
        Load texts from a .txt file.
        """
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(self.texts)} texts.")

        # 텍스트를 새로 읽었으니 pseudo_labels 리스트도 초기화 또는 길이 체크
        if len(self._pseudo_labels) != len(self.texts):
            # 필요하다면 길이를 맞추거나 초기화
            self._pseudo_labels = [None] * len(self.texts)

    def attach_pseudo_labels(self, pseudo_labels_list):
        """
        외부(Trainer 등)에서 생성한 pseudo-label 리스트를 붙인다.
        pseudo_labels_list: List[Any], 길이는 self.__len__()와 동일해야 함.
        """
        if len(pseudo_labels_list) != len(self.texts):
            print(f"Warning: pseudo_labels_list 길이({len(pseudo_labels_list)}) != texts 길이({len(self.texts)})")
            min_len = min(len(pseudo_labels_list), len(self.texts))
            self._pseudo_labels = pseudo_labels_list[:min_len] + [None]*(len(self.texts)-min_len)
        else:
            self._pseudo_labels = pseudo_labels_list

        print(f"Attached {len(self._pseudo_labels)} pseudo-labels to dataset.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 텍스트 전체를 바로 tokenizer에 전달
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        # 토크나이저 결과는 (batch_size=1, max_length) 형태이므로 squeeze 처리
        input_ids = inputs.input_ids.squeeze(0)    # shape: (max_length,)
        attention_mask = inputs.attention_mask.squeeze(0)  # shape: (max_length,)

        # causal LM에서 일반적으로 labels = input_ids를 그대로 사용
        labels = input_ids.clone()

        # pseudo_label이 있는 경우 가져오기
        pseudo_label = None
        if 0 <= idx < len(self._pseudo_labels):
            pseudo_label = self._pseudo_labels[idx]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # pseudo_label이 존재하면 배치에 추가
        if pseudo_label is not None:
            # pseudo_label을 텐서로 변환(문장 단위 클러스터면 scalar, 토큰 단위면 shape 맞춰야 함)
            if not torch.is_tensor(pseudo_label):
                pseudo_label = torch.tensor(pseudo_label, dtype=torch.long)
            output["pseudo_label"] = pseudo_label

        return output
