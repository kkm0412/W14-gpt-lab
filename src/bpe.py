# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path


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

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        self.id_to_token[0] = PAD_TOKEN
        self.id_to_token[1] = UNK_TOKEN
        self.id_to_token[2] = BOS_TOKEN
        self.id_to_token[3] = EOS_TOKEN

        self.token_to_id[PAD_TOKEN] = 0
        self.token_to_id[UNK_TOKEN] = 1
        self.token_to_id[BOS_TOKEN] = 2
        self.token_to_id[EOS_TOKEN] = 3

        for i in range(NUM_BYTES):
            tok_id = BYTE_OFFSET + i
            self.id_to_token[tok_id] = i
            self.token_to_id[i] = tok_id

        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

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
        self._init_special_tokens()

        tokens = [self.token_to_id[byte] for byte in corpus.encode("utf-8")]
        
        while len(self.id_to_token) < self.vocab_size:
            pair_count = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair not in pair_count:
                    pair_count[pair] = 1
                else:
                    pair_count[pair] += 1
            best_pair = max(pair_count, key=lambda pair: pair_count[pair])
            
            if best_pair not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[best_pair] = new_id
                self.id_to_token[new_id] = best_pair
                self.merges.append(best_pair)
            
            new_token = []
            i = 0
            while i < len(tokens):
                if tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_token.append(self.token_to_id[best_pair])
                    i += 2
                else:
                    new_token.append(tokens[i])
                    i += 1
            tokens = new_token              


        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        path = Path(path)

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": [],
            "merges": [list(pair) for pair in self.merges],
        }

        for token_id, token in sorted(self.id_to_token.items()):
            if isinstance(token, str):
                item = {
                    "id": token_id,
                    "type": "special",
                    "value": token,
                }
            elif isinstance(token, int):
                item = {
                    "id": token_id,
                    "type": "byte",
                    "value": token,
                }
            elif isinstance(token, tuple):
                item = {
                    "id": token_id,
                    "type": "merge",
                    "value": list(token),
                }
            else:
                raise TypeError(f"저장할 수 없는 token 타입입니다: {type(token)}")

            data["id_to_token"].append(item)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = [tuple(pair) for pair in data["merges"]]

        for item in data["id_to_token"]:
            token_id = int(item["id"])
            token_type = item["type"]
            value = item["value"]

            if token_type == "special":
                token = value
            elif token_type == "byte":
                token = int(value)
            elif token_type == "merge":
                token = tuple(value)
            else:
                raise ValueError(f"알 수 없는 token type입니다: {token_type}")

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        id_list = [self.token_to_id[byte] for byte in text.encode("utf-8")]
        
        while (True):
            copied_list = []
            i = 0
            is_changed = False
            while i < len(id_list):
                if i < len(id_list) -1:
                    pair = (id_list[i], id_list[i+1])
                    if pair in self.merges:
                        copied_list.append(self.token_to_id[pair])
                        i += 2
                        is_changed = True
                        continue
                
                copied_list.append(id_list[i])
                i += 1

            id_list = copied_list
            if is_changed != True:
                break
        if add_bos_eos:
            id_list = [self.get_bos_id()] + id_list + [self.get_eos_id()]
        return id_list
        # raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def expand(self, token_id: int) -> list[int]:
        token = self.id_to_token[token_id]
        if isinstance(token, int):
            return [token]
        if isinstance(token, tuple):
            left, right = token
            return self.expand(left) + self.expand(right)
        return []



    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        
        byte_values = []
        for token in ids:
            if skip_special and token in SPECIAL_IDS.values():
                continue
            byte_values.extend(self.expand(token))
        return bytes(byte_values).decode("utf-8")
        # raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
