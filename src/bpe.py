# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
from collections import defaultdict
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for token, idx in SPECIAL_IDS.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        for i in range(NUM_BYTES):
            self.id_to_token[i + BYTE_OFFSET] = bytes([i])
            self.token_to_id[bytes([i])] = i + BYTE_OFFSET

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """ 
        # corpus를 utf-8로 encode, byte seqeunce 생성
        byte_sequence = corpus.encode("utf-8") # bytes 타입의 byte 값들이 나열된 객체
        # byte sequnece를 순회하며 각 byte에 4를 더해 vocabulary 안의 token ID로 변환해 byte token id sequence 생성
        token_ids = [byte + BYTE_OFFSET for byte in byte_sequence]

        new_id = len(self.token_to_id)
        while ((new_id < self.vocab_size) and len(token_ids) >= 2):
            # 가장 자주 등장하는 이웃 token pair 찾기
            frequency = defaultdict(int)
            for i in range(len(token_ids) - 1):
                frequency[(token_ids[i], token_ids[i + 1])] += 1

            most_frequent_pair = max(frequency, key = lambda pair: frequency[pair])

            # 가장 자주 등장한 이웃의 빈도수가 2보다 작으면 merge 진행 X
            if frequency[most_frequent_pair] < 2:
                break
            # new token ID sequence를 만들고, sequence의 해당 pair를 새 ID로 치환
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == most_frequent_pair:
                    new_token_ids.append(new_id)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1

            # token_ids update
            token_ids = new_token_ids
            # new_token을 tuple로 저장
            new_token = most_frequent_pair
            self.merges.append(most_frequent_pair)
            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id

            new_id += 1

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        saved_id_to_token = {}
        for token_id, token in self.id_to_token.items():
            if isinstance(token, tuple):
                saved_id_to_token[token_id] = {"type": "tuple", "value": list(token)}
            elif isinstance(token, bytes):
                saved_id_to_token[token_id] = {"type": "bytes", "value": list(token)}
            else:
                saved_id_to_token[token_id] = {"type": "special", "value": token}

        saved_merges = [list(pair) for pair in self.merges]

        saved_vocab_size = self.vocab_size

        data = {
            "vocab_size": saved_vocab_size,
            "id_to_token": saved_id_to_token,
            "merges": saved_merges
            }
        
        with open(path, 'w', encoding = "utf-8") as f:
            json.dump(data, f, ensure_ascii = False, indent = 2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, 'r', encoding = "utf-8") as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.id_to_token = {}
        for token_id, token_dict in data["id_to_token"].items():
            if token_dict["type"] == "bytes":
                self.id_to_token[int(token_id)] = bytes(token_dict["value"])
            elif token_dict["type"] == "tuple":
                self.id_to_token[int(token_id)] = tuple(token_dict["value"])
            else:
                self.id_to_token[int(token_id)] = token_dict["value"]
        
        self.token_to_id = {}
        for token_id, token in self.id_to_token.items():
            self.token_to_id[token] = token_id

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
