# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
from collections import Counter

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
        # 특수 토큰을 고정 ID 0~3에 등록한다.
        self.id_to_token = {idx:token for idx, token in enumerate(SPECIAL_TOKENS)}
        print(self.id_to_token)
        
        self.token_to_id.update(SPECIAL_IDS)
        print(self.token_to_id)
        
        # 바이트를 등록한다.
        for idx in range(NUM_BYTES):
            self.id_to_token[BYTE_OFFSET+idx] = bytes([idx])
        print("\nid_to_token:", list(self.id_to_token.keys())[4:7])
        print("id_to_token:", list(self.id_to_token.values())[4:7])
        
        self.token_to_id.update({bytes([idx]):(BYTE_OFFSET+idx) for idx in range(NUM_BYTES)})
        print("\ntoken_to_id:", list(self.token_to_id.keys())[0:7])
        print("token_to_id:", list(self.token_to_id.values())[0:7])
        
        return
        raise NotImplementedError("_init_special_tokens를 구현하세요.")

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
    
    def replace_pair(self, byte_id_sequence, pair, new_id):
        """
        id list를 순회하면서 pair을 찾아 new_id로 치환한다
        """
        i = 0
        
        while i < len(byte_id_sequence) - 1:
            curr = byte_id_sequence[i]
            next = byte_id_sequence[i+1]
            
            if (curr, next) == pair:
                byte_id_sequence[i] = new_id
                byte_id_sequence.pop(i+1)

            else:
                i += 1
        return
    
    def find_freq_pair(self, byte_id_sequence):
        """자주 등장하는 인접 쌍 찾기
        인접한 두 쌍:count의 dict을 만들고, 가장 count가 높은 pair을 반환한다. 만약 2번 이상 나타나는
        쌍이 없으면 None을 반환한다.

        Args:
            byte_id_sequence (list): id bytes list

        Returns:
            tuple[tuple[int, int], int] | None: ((token_a, token_b), count)
        """
        pair_count = Counter(zip(byte_id_sequence, byte_id_sequence[1:]))
        
        if pair_count.most_common(1)[0][1] >= 2:
            most_common_pair = pair_count.most_common(1)[0][0]
            return most_common_pair
        
        return None

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # corpus를 utf-8로 변환하고 byte sequence를 만든다.
        byte_id_sequence = list(corpus.encode("utf-8"))
        
        while(self.find_freq_pair(byte_id_sequence) and len(self.merges) <= self.vocab_size-260):
            curr_pair = self.find_freq_pair(byte_id_sequence)
            self.merges.append(curr_pair)
            
            new_id = 259 + len(self.merges)
            new_token = self.id_to_token[curr_pair[0]] + self.id_to_token[curr_pair[1]]
            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id
            
            self.replace_pair(byte_id_sequence, curr_pair, new_id)
        
        return
        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        # bytes는 JSON 불가 → type 태그 + 정수 리스트로 변환
        serialized = {}
        for token_id, token in self.id_to_token.items():
            if isinstance(token, bytes):
                serialized[str(token_id)] = {"type": "bytes", "value": list(token)}
            else:  # 특수 토큰(str)
                serialized[str(token_id)] = {"type": "str", "value": token}

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": serialized,
            "merges": [list(pair) for pair in self.merges],  # tuple → list
        }

        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str | Path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.merges = [tuple(pair) for pair in data["merges"]]  # list → tuple 복원

        # JSON 키는 항상 str → int로 복원, value는 type 태그 보고 복원
        self.id_to_token = {}
        for key, entry in data["id_to_token"].items():
            token = bytes(entry["value"]) if entry["type"] == "bytes" else entry["value"]
            self.id_to_token[int(key)] = token

        # id_to_token 역방향으로 token_to_id 재구성
        self.token_to_id = {token: token_id for token_id, token in self.id_to_token.items()}

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # 1. text를 utf-8 byte로 펼치고, 각 byte 값에 BYTE_OFFSET(4)을 더해 ID 리스트로 만든다.
        byte_id_list = [BYTE_OFFSET+byte for byte in text.encode("utf-8")]
        
        # 2. self.merges를 merge rule학습 순서대로 순회한다 (enumerate로 인덱스도 같이):
        for pair in range(self.merges):
            new_id = self.token_to_id[pair]
            self.replace_pair(byte_id_list, pair, new_id)
            
        # 3. add_bos_eos=True이면 merge가 다 끝난 뒤 맨 앞에 bos, 맨 뒤에 eos ID를 붙인다.
        if add_bos_eos == True:
            byte_id_list = BOS_TOKEN + byte_id_list + EOS_TOKEN
            
        # 4. 완성된 ID 리스트를 반환한다.
        return byte_id_list
        
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # 1. 모든 byte를 모을 빈 버퍼(byte 리스트 or bytearray)를 준비한다.
        byte_buffer = bytearray()
        
        # 2. ids를 하나씩 순회한다:
        for id in ids:
            if skip_special and id < BYTE_OFFSET:
                continue
            byte_buffer += self.id_to_token[id]
  
        # 3. 루프가 끝난 뒤, 모인 전체 bytes를 한 번에 .decode("utf-8")로 문자열 복원.
        result = byte_buffer.decode("utf-8")
        return result
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer._init_special_tokens()
