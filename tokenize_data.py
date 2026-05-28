# -*- coding: utf-8 -*-
"""
NSMC LM 텍스트를 BPE token id 텐서로 미리 변환합니다.

실행 예:
    .venv/bin/python tokenize_data.py
    .venv/bin/python tokenize_data.py --vocab-size 3000
    .venv/bin/python tokenize_data.py --train-char-limit 200000
"""

import argparse
from pathlib import Path

import torch

from src.bpe import BPETokenizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def read_tokenizer_training_text(path: Path, train_char_limit: int | None) -> str:
    text = path.read_text(encoding="utf-8")
    if train_char_limit is None:
        return text
    return text[:train_char_limit]


def encode_lm_file(tokenizer: BPETokenizer, input_path: Path, output_path: Path) -> int:
    token_ids: list[int] = []
    eos_id = tokenizer.get_eos_id()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            token_ids.extend(tokenizer.encode(text))
            token_ids.append(eos_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(token_ids, dtype=torch.long), output_path)
    return len(token_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/load BPE tokenizer and cache NSMC LM files as token id tensors."
    )
    parser.add_argument("--train-text", type=Path, default=DATA_DIR / "nsmc_lm_train.txt")
    parser.add_argument("--val-text", type=Path, default=DATA_DIR / "nsmc_lm_val.txt")
    parser.add_argument("--tokenizer-out", type=Path, default=DATA_DIR / "vocab_bpe.json")
    parser.add_argument("--train-ids-out", type=Path, default=DATA_DIR / "nsmc_lm_train_ids.pt")
    parser.add_argument("--val-ids-out", type=Path, default=DATA_DIR / "nsmc_lm_val_ids.pt")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument(
        "--train-char-limit",
        type=int,
        default=200_000,
        help="BPE vocab 학습에 사용할 train 텍스트 글자 수. 0이면 전체 train 파일을 사용합니다.",
    )
    parser.add_argument(
        "--reuse-tokenizer",
        action="store_true",
        help="tokenizer-out 파일이 이미 있으면 새로 학습하지 않고 불러옵니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_char_limit = None if args.train_char_limit == 0 else args.train_char_limit
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)

    if args.reuse_tokenizer and args.tokenizer_out.exists():
        tokenizer.load(args.tokenizer_out)
        print(f"loaded tokenizer: {args.tokenizer_out}")
    else:
        train_text = read_tokenizer_training_text(args.train_text, train_char_limit)
        tokenizer.train(train_text)
        args.tokenizer_out.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(args.tokenizer_out)
        print(f"saved tokenizer: {args.tokenizer_out}")
        print(f"vocab size: {len(tokenizer.id_to_token):,}")

    train_count = encode_lm_file(tokenizer, args.train_text, args.train_ids_out)
    val_count = encode_lm_file(tokenizer, args.val_text, args.val_ids_out)

    print(f"saved train ids: {args.train_ids_out} ({train_count:,} tokens)")
    print(f"saved val ids: {args.val_ids_out} ({val_count:,} tokens)")


if __name__ == "__main__":
    main()
