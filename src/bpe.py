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

        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

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

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer._init_special_tokens()
