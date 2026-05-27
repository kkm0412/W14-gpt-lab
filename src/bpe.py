# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

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
        self._init_special_tokens()                             # 0~3 특수  토큰 4~259 byte token 로 기본 vocab 만들어줌
        byteSe = corpus.encode("utf-8")                         # ex) "A" -> [65] "가"->[234,176,128] 처럼 문자열 corpus를 UTF-8 sequence로 바꾼다
        #  b'254,123,111' -> [258, 127, 115]
        byteIdSe = []
        for i in range(len(byteSe)): # 0,1,2                    # ex) byte 65 -> token id 69 각 byte 값들을 token id 로 바꿔서 byteIdSe 리스트에 삽입
            byteIdSe.append(byteSe[i] + BYTE_OFFSET)           
       # 리스트 컴프리헨션 byteIdSE = [byte + NUM_BYTES for byte in byteSe]
       # A 한번만 a byte 한 byte3   a b c d   
        while (len(self.id_to_token) < self.vocab_size):        # 현재 vocab_size 3000인 크기에 도달할때까지 현재 vocab 크기에 merge token을 추가한다 | len(id_to_token) = 현재 vocab에 등록된 token 종류 수
            pair_counts = {}                                    # 인접 piar count 용
            
            for i in range (len(byteIdSe) -1):                  # 현재 token과 다음 token을 묶어서 pair로 만든다 이미 본 count를 증가시키고 처음 본 pair면 1로 시작 즉 pair의 등장횟수를 count로 저장
                pair = (byteIdSe[i], byteIdSe[i+1])
                if pair in pair_counts:
                    pair_counts[pair] += 1
                else:
                    pair_counts[pair] = 1 
            if not pair_counts:                                 # 인접 pair없으면 merge 할 수 없기에 반복문 멈춤
                break
            best_pair = max(pair_counts, key = pair_counts.get) # pair_counts 안의 pair들 중 count 값이 가장 큰 pair를 골라라     {(155,76):2,(156,77):3}

            new_id = len(self.id_to_token)                      # 새 merge token 의 ID를 만든다

            self.id_to_token[new_id] = best_pair                # 새 token id가 어떤 pair를 의미하는지 저장
            self.token_to_id[best_pair] = new_id                # 반대 방향도 저장 나중에 encode할 때 어떤 pair가 merge 대상인지 찾을 수 있다.
            self.merges.append(best_pair)                       # merge 규칙을 학습된 순서대로 저장한다 encode() 는 나중에 self.merges를 앞에서부터 순서대로 적용한다

            new_sequence = []                                   # 이제 현재 token sequence안에서 best_pair가 나오는 부분을 새 token id로 실제 치환한다. 치환 결과를 담을 새 리스트를 만든다
            j = 0

            while j < len(byteIdSe):                            # byteIdSe의 처음부터 끝까지 처리 현재 token과 다음 token이 best pair인지 확인한다
                if j < len(byteIdSe) - 1 and (byteIdSe[j], byteIdSe[j + 1]) == best_pair: # 현재 pair가 best_pair라면 두 token을 새 token 하나로 바꾼다 이미 두 token을 처리했으므로 j를 2칸 이동 
                    new_sequence.append(new_id)
                    j += 2
                else:                                                                     # 현재 위치가 best_pai가 아니라면 기존 token을 그대로 넣는다 하나의 token을 처리했으므로 j를 1칸 이동
                    new_sequence.append(byteIdSe[j])
                    j += 1

            byteIdSe = new_sequence

            
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
