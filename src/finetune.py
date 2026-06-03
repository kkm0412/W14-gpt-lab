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
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = _clean_text(row.get("document"))
            label = row.get("label")

            if not text or label not in {"0", "1"}:
                continue

            rows.append({"text": text, "label": int(label)})
    return rows


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
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

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    if len(train_rows) <= 1 or val_ratio <= 0:
        val_size = 0
    else:
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

    accuracy 계산:
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).sum()
    """
    model.to(device)
    model.train()
    if not any(param.requires_grad for param in model.gpt.parameters()):
        model.gpt.eval()

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
