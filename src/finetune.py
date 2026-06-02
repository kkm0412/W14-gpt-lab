# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿.

Top-down으로 보면 이 파일은 다음 4단계를 담당합니다.

1. 원본 NSMC TSV를 과제용 dict 리스트로 바꾸기
   - 입력 컬럼: id, document, label
   - document: 영화 리뷰 문장
   - label: 0은 부정, 1은 긍정
2. 리뷰 문장을 token id tensor로 바꾸는 Dataset 만들기
   - tokenizer.encode(text)로 정수 ID 리스트 생성
   - GPT가 한 번에 볼 수 있는 길이(max_length)에 맞춰 자르거나 padding
3. 사전 학습된 GPT backbone 위에 분류 head 붙이기
   - GPT의 LM head는 "다음 토큰 예측"용
   - 감성 분류는 hidden state를 뽑아 Linear classifier로 0/1 logits 예측
4. train/evaluate 루프에서 loss와 accuracy 계산하기
   - loss: 정답 label에 비해 logits가 얼마나 틀렸는지
   - accuracy: argmax(logits)가 label과 같은 비율
"""

import csv
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _clean_text(text: str | None) -> str:
    """리뷰 안의 여러 공백/개행을 하나의 공백으로 정리합니다."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    """NSMC TSV 파일을 {"text": ..., "label": ...} 리스트로 읽습니다."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = _clean_text(row.get("document"))
            label = row.get("label")

            # 빈 리뷰나 0/1이 아닌 label은 분류 학습에 쓸 수 없으므로 건너뜁니다.
            if not text or label not in {"0", "1"}:
                continue

            rows.append({"text": text, "label": int(label)})
    return rows


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """감성 분류용 dict 리스트를 JSONL 파일로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.
    """

    train_rows = _read_nsmc_tsv(train_tsv_path)
    test_rows = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    # seed를 고정하면 매번 같은 train/validation split이 만들어져 비교가 쉬워집니다.
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    if len(train_rows) <= 1 or val_ratio <= 0:
        val_size = 0
    else:
        # 너무 작은 데이터에서도 validation이 생기게 하되, train이 전부 사라지지는 않게 합니다.
        val_size = max(1, int(len(train_rows) * val_ratio))
        val_size = min(val_size, len(train_rows) - 1)

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = test_rows

    if output_dir is not None:
        output_dir = Path(output_dir)
        _write_jsonl(output_dir / "nsmc_sentiment_train.jsonl", train_data)
        _write_jsonl(output_dir / "nsmc_sentiment_val.jsonl", val_data)
        _write_jsonl(output_dir / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다.

    이 Dataset은 DataLoader가 batch를 만들 수 있도록 샘플 하나를 tensor 형태로 꺼내 줍니다.
    입력 dict 하나의 모양은 {"text": "리뷰", "label": 0 또는 1}입니다.
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다.
        """
        item = self.data[idx]
        token_ids = self.tokenizer.encode(item["text"], add_bos_eos=True)

        token_ids = token_ids[: self.max_length]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label = int(item["label"])
        return input_ids, label


class GPTForSequenceClassification(nn.Module):

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = 0
        # TODO: dropout과 classifier를 정의하세요.
        # classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        emb_dim = gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.

        """

        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.finalnorm(x)

        non_pad_lengths = (input_ids != self.pad_id).sum(dim=1).clamp(min=1)
        last_token_positions = non_pad_lengths - 1
        batch_positions = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = x[batch_positions, last_token_positions]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if labels is None:
            return logits

        labels = labels.long()
        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다.

    구현 흐름:
        1. model.train()으로 dropout/학습 모드를 켭니다.
        2. train_loader에서 input_ids, labels batch를 꺼냅니다.
        3. batch tensor를 device로 옮깁니다.
        4. model(input_ids, labels)로 loss와 logits를 얻습니다.
        5. optimizer.zero_grad() -> loss.backward() -> optimizer.step() 순서로 갱신합니다.
        6. loss 합계와 정답 개수를 누적합니다.

    accuracy 계산:
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum()
    """
    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device) if torch.is_tensor(labels) else torch.tensor(labels, device=device)
        labels = labels.long()

        optimizer.zero_grad(set_to_none=True)
        loss, logits = model(input_ids, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")

    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """
    TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다.
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device) if torch.is_tensor(labels) else torch.tensor(labels, device=device)
            labels = labels.long()

            loss, logits = model(input_ids, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        return float("nan"), float("nan")

    return total_loss / total_examples, total_correct / total_examples
