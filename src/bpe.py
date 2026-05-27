# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
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

            if pair_counts[best_pair] < 2:
                break

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


        path = Path(path)                                      # 문자열 경로가 들어와도 Path 객체로 통일해서 path.open(...)처럼 파일 경로 기능을 쓰기 위함

        vocab_data = {}                                        # self.id_to_token을 JSON이 저장할 수 있는 형태로 바꿔 담을 저장용 vocab dict

        for token_id, token in self.id_to_token.items():       # 현재 vocab에 등록된 token id와 token 값을 하나씩 꺼내 JSON용 형태로 변환
            token_id_str = str(token_id)                       # JSON의 key는 문자열로 저장되므로 id를 str로 바꿔 저장하고 load 때 다시 int로 복원

            if isinstance(token, bytes):                       # byte token인 경우 ex) b"A"; JSON은 bytes를 직접 저장하지 못함
                vocab_data[token_id_str] = {
                    "type": "bytes",                          # load에서 bytes(...)로 복원할 수 있게 원래 타입 정보를 함께 저장
                    "value": list(token),                      # ex) b"A" -> [65]; byte 값을 JSON이 저장 가능한 숫자 리스트로 변환
                }
            elif isinstance(token, tuple):                     # merge token인 경우 ex) (69, 70); JSON은 tuple 타입을 직접 보존하지 못함
                vocab_data[token_id_str] = {
                    "type": "tuple",                          # load에서 tuple(...)로 복원할 수 있게 merge token이었다는 타입 정보를 저장
                    "value": list(token),                      # ex) (69, 70) -> [69, 70]; JSON이 저장 가능한 리스트로 변환
                }
            else:                                              # bytes/tuple이 아니면 <pad>, <unk>, <bos>, <eos> 같은 special token 문자열
                vocab_data[token_id_str] = {
                    "type": "str",                             # load에서 문자열 그대로 복원하면 된다는 타입 정보
                    "value": token,                            # 문자열은 JSON에 그대로 저장 가능하므로 변환하지 않음
                }

        merges_data = []                                       # self.merges의 tuple pair들을 JSON 저장용 list pair로 바꿔 담을 리스트
        for pair in self.merges:                               # 학습된 merge rule을 순서대로 하나씩 꺼냄; encode에서 이 순서가 중요함
            merges_data.append(list(pair))                     # ex) (69, 70) -> [69, 70]; JSON 저장 후 load에서 다시 tuple로 복원

        data = {
            "vocab_size": self.vocab_size,                     # tokenizer가 어떤 vocab 크기 설정으로 만들어졌는지 저장
            "id_to_token": vocab_data,                         # decode 때 token id가 무엇을 의미하는지 복원하기 위한 vocab 정보
            "merges": merges_data,                             # encode 때 어떤 pair를 어떤 순서로 합칠지 복원하기 위한 merge rule 정보
        }

        with path.open("w", encoding="utf-8") as f:            # 저장할 JSON 파일을 쓰기 모드로 열고, 한글이 깨지지 않게 UTF-8 인코딩 사용
            json.dump(data, f, ensure_ascii=False)             # 메모리의 data dict를 실제 JSON 파일로 저장; ensure_ascii=False는 한글을 읽기 좋게 저장
            
    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r", encoding="utf-8") as f:           # save()가 만든 JSON 파일을 읽기 모드로 열고 UTF-8로 해석
            data = json.load(f)                                # JSON 파일 내용을 Python dict로 읽어와서 vocab/merge 정보를 꺼낼 준비

        self.vocab_size = data["vocab_size"]                   # 저장 당시 tokenizer의 vocab_size 설정값을 다시 복원
        self.merges = [tuple(pair) for pair in data["merges"]] # JSON에는 list로 저장된 merge pair들을 encode가 쓰기 좋은 tuple pair 리스트로 복원
        self.id_to_token = {}                                  # 파일에 저장된 vocab으로 새로 채우기 위해 기존 id_to_token을 초기화

        for token_id, token_dict in data["id_to_token"].items(): # 저장된 vocab 항목을 하나씩 꺼냄; token_id는 JSON key라 문자열 상태
            token_id_int = int(token_id)                       # save 때 str로 저장한 token id를 실제 tokenizer에서 쓰는 int id로 복원

            if token_dict["type"] == "bytes":                 # save 때 bytes token이라고 표시한 항목이면 ex) [65] -> b"A"로 복원
                self.id_to_token[token_id_int] = bytes(token_dict["value"])
            elif token_dict["type"] == "tuple":               # save 때 merge token이라고 표시한 항목이면 ex) [69, 70] -> (69, 70)으로 복원
                self.id_to_token[token_id_int] = tuple(token_dict["value"])
            else:                                             # special token 문자열이면 ex) "<pad>" 그대로 복원
                self.id_to_token[token_id_int] = token_dict["value"]

        self.token_to_id = {}                                  # id_to_token을 복원했으므로 반대 방향 dict도 다시 만들기 위해 초기화
        for token_id, token in self.id_to_token.items():       # 복원된 vocab을 돌면서 token -> id 방향 매핑을 재구성
            self.token_to_id[token] = token_id                 # encode에서 token이나 merge pair로 id를 찾을 수 있게 반대 방향 저장

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        byteSe = text.encode("utf-8")                         # 입력 문자열을 UTF-8 byte sequence로 변환 ex) "A" -> [65], "가" -> [234, 176, 128]

        byteIdSe = []                                         # encode 결과를 만들기 전, 먼저 byte 값들을 기본 byte token id로 바꿔 담을 리스트
        for i in range(len(byteSe)):                          # UTF-8 byte sequence의 각 byte 값을 하나씩 확인
            byteIdSe.append(byteSe[i] + BYTE_OFFSET)          # special token 0~3을 피하기 위해 byte 값에 4를 더해 token id로 변환 ex) 65 -> 69

        for merge_pair in self.merges:                        # train/load에서 얻은 merge rule을 학습된 순서대로 하나씩 적용
            new_id = self.token_to_id[merge_pair]             # 현재 merge_pair가 vocab에서 어떤 token id인지 찾음 ex) (69, 70) -> 260

            new_sequence = []                                 # 현재 merge rule을 적용한 결과를 새로 담을 리스트; 원본을 직접 고치면 인덱스가 꼬일 수 있음
            j = 0                                             # 현재 byteIdSe에서 처리 중인 위치

            while j < len(byteIdSe):                          # 현재 token sequence를 처음부터 끝까지 훑으며 merge_pair가 있는지 확인
                if j < len(byteIdSe) - 1 and (byteIdSe[j], byteIdSe[j + 1]) == merge_pair: # 현재 token과 다음 token이 merge 대상이면
                    new_sequence.append(new_id)               # 두 token을 merge token 하나로 치환해서 결과 리스트에 추가
                    j += 2                                    # token 두 개를 이미 처리했으므로 두 칸 이동
                else:
                    new_sequence.append(byteIdSe[j])          # merge 대상이 아니면 현재 token을 그대로 결과 리스트에 추가
                    j += 1                                    # token 하나만 처리했으므로 한 칸 이동
            byteIdSe = new_sequence                           # 이번 merge rule 적용 결과를 다음 merge rule의 입력 sequence로 사용

        if add_bos_eos:                                       # 호출할 때 add_bos_eos=True이면 문장 시작/끝 표시 token을 붙임
            byteIdSe = [self.get_bos_id()] + byteIdSe + [self.get_eos_id()] # 앞에는 <bos>, 뒤에는 <eos> token id를 추가

        return byteIdSe                                       # 최종 token id 리스트를 반환; GPT 모델 입력으로 사용할 수 있는 숫자 sequence
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        def token_to_bytes(token_id: int) -> list[int]:        # token id 하나를 원래 byte 값 리스트로 풀어주는 내부 함수
            token = self.id_to_token.get(token_id)             # id_to_token에서 현재 token id가 무엇을 의미하는지 찾음

            if token is None:                                  # vocab에 없는 id가 들어오면 복원할 정보가 없으므로 빈 리스트 반환
                return []

            if isinstance(token, bytes):                       # 기본 byte token인 경우 ex) token id 69 -> b"A"
                return list(token)                             # b"A" -> [65]처럼 실제 byte 값 리스트로 변환

            if isinstance(token, tuple):                       # merge token인 경우 ex) 260 -> (69, 70)
                byte_values = []                               # merge token 내부 token들을 byte까지 풀어서 담을 리스트
                for inner_id in token:                         # merge token은 두 token id를 합친 것이므로 내부 id를 하나씩 확인
                    byte_values.extend(token_to_bytes(inner_id)) # 내부 token도 merge일 수 있으므로 재귀적으로 끝까지 byte token까지 펼침
                return byte_values                             # merge token 전체를 원래 byte 값 리스트로 복원한 결과 반환

            if skip_special:                                   # special token 문자열이고 skip_special=True이면 결과 문자열에서 제외
                return []

            return list(token.encode("utf-8"))                 # skip_special=False이면 "<bos>" 같은 문자열도 보이도록 UTF-8 byte로 변환

        byte_values = []                                       # 모든 token id를 풀어서 얻은 byte 값들을 순서대로 모을 리스트

        for token_id in ids:                                   # 입력으로 들어온 token id sequence를 왼쪽부터 하나씩 처리
            byte_values.extend(token_to_bytes(token_id))       # 각 token id를 byte 값들로 풀고 전체 byte 리스트에 이어 붙임

        return bytes(byte_values).decode("utf-8")              # 한글은 여러 byte로 이루어지므로 byte를 전부 모은 뒤 마지막에 한 번만 UTF-8 decode
