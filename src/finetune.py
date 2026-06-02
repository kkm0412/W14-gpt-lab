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

    Top-down 목표:
        "파일에 있는 행"을 "모델이 학습할 샘플 목록"으로 바꾸는 전처리 함수입니다.

    입력 데이터 모양:
        id<TAB>document<TAB>label
        6270596<TAB>굳 ㅋ<TAB>1
        9274899<TAB>GDNTOPCLASSINTHECLUB<TAB>0

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]

    구현 흐름:
        1. train_tsv_path를 UTF-8 텍스트로 엽니다.
        2. header(id, document, label)를 기준으로 각 행을 읽습니다.
        3. document가 비어 있는 행은 학습 신호가 없으므로 제거합니다.
        4. label은 문자열 "0"/"1"에서 int 0/1로 바꿉니다.
        5. train 파일 일부를 validation으로 분리합니다.
        6. test_tsv_path가 있으면 같은 규칙으로 test_data를 만듭니다.

    학습 포인트:
        여기서는 토크나이징을 하지 않습니다. 이 함수의 책임은 "원본 파일 정리"까지이고,
        token id 변환은 ReviewSentimentDataset이 담당합니다.
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
        # data는 make_sentiment_dataset이 만든 dict 리스트입니다.
        # tokenizer는 문자열을 token id 리스트로 바꾸는 역할만 맡습니다.
        # max_length는 GPT 입력 길이를 고정하기 위한 기준입니다.
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # pad_id를 외부에서 주지 않으면 tokenizer의 <pad> ID를 사용합니다.
        # padding은 batch 안의 모든 샘플 길이를 같게 만들기 위해 필요합니다.
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        # Dataset의 길이 = 사용할 리뷰 샘플 개수입니다.
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다.

        구현 흐름:
            1. self.data[idx]에서 text와 label을 꺼냅니다.
            2. tokenizer.encode(text, add_bos_eos=True 또는 False)로 token id 리스트를 만듭니다.
            3. 길이가 max_length보다 길면 앞에서 max_length개만 남깁니다.
            4. 길이가 max_length보다 짧으면 pad_id를 뒤에 붙입니다.
            5. input_ids는 torch.long tensor, label은 int로 반환합니다.

        왜 길이를 고정하나:
            Transformer는 batch 단위로 (batch_size, seq_len) 모양의 tensor를 받습니다.
            한 batch 안에서 seq_len이 서로 다르면 바로 쌓을 수 없어서 padding이 필요합니다.
        """
        item = self.data[idx]
        token_ids = self.tokenizer.encode(item["text"], add_bos_eos=True)

        # GPT 입력 길이는 고정되어야 하므로 길면 자르고, 짧으면 pad_id로 채웁니다.
        token_ids = token_ids[: self.max_length]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label = int(item["label"])
        return input_ids, label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.

    Top-down 구조:
        input_ids
          -> GPT token/position embedding
          -> Transformer blocks
          -> 각 위치별 hidden state: (batch, seq_len, emb_dim)
          -> 문장 대표 벡터 하나 선택
          -> classifier
          -> logits: (batch, num_labels)

    여기서 logits[0]은 보통 "부정 점수", logits[1]은 "긍정 점수"로 해석합니다.
    """

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
        #
        # 학습 포인트:
        # - GPTModel의 lm_head는 vocab_size개의 "다음 토큰 후보 점수"를 냅니다.
        # - 감성 분류는 vocab_size가 아니라 num_labels(여기서는 2)개의 점수만 필요합니다.
        # - 그래서 backbone은 공유하되, 마지막 head는 classification용으로 따로 둡니다.
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

        구현 흐름:
            1. input_ids를 GPT backbone에 통과시켜 hidden state를 얻습니다.
            2. 문장 전체를 대표할 벡터 하나를 고릅니다.
               - 쉬운 방식: 마지막 토큰 위치 hidden state 사용
               - padding을 더 엄밀히 처리하려면 마지막 non-pad 위치를 찾을 수 있습니다.
            3. dropout을 적용합니다.
            4. classifier로 (batch, num_labels) logits를 만듭니다.
            5. labels가 있으면 CrossEntropyLoss(logits, labels)를 계산합니다.

        중요한 구분:
            label 0/1은 "분류 정답 번호"입니다.
            token id 0/1은 tokenizer vocabulary 안의 "<pad>/<unk>" 같은 토큰 번호일 수 있습니다.
            둘 다 숫자지만 의미가 완전히 다릅니다.
        """
        # GPTModel.forward()는 LM head까지 지난 vocab logits를 반환합니다.
        # 감성 분류에는 vocab logits가 아니라 Transformer hidden state가 필요하므로
        # backbone의 내부 모듈을 같은 순서로 직접 호출합니다.
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.finalnorm(x)

        # padding이 뒤에 붙어 있으므로 각 샘플의 마지막 non-pad 위치를 문장 대표 위치로 씁니다.
        # BOS/EOS를 붙인 Dataset에서는 보통 EOS 위치의 hidden state가 선택됩니다.
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

    구현 흐름:
        1. model.eval()로 평가 모드를 켭니다.
        2. torch.no_grad() 안에서 forward만 수행합니다.
        3. optimizer.step()은 절대 호출하지 않습니다.
        4. train과 같은 방식으로 평균 loss와 accuracy를 누적 계산합니다.

    학습 포인트:
        train은 "가중치를 바꾸는 루프"이고 evaluate는 "현재 가중치를 재는 루프"입니다.
        두 함수가 비슷해 보여도 gradient와 optimizer 사용 여부가 핵심 차이입니다.
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
