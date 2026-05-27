# `bpe.py` 변경 보고서

비교 기준:
- 원본: commit `035cb838db810bbedfe9c199902020a2b8a21aae`의 `src/bpe.py`
- 현재: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:1)

목적:
- 현재 코드가 원본 대비 어디가 달라졌는지 이해한다.
- 특히 `train()`의 작동 방식이 어떻게 바뀌었는지, 왜 그 변경이 필요했는지 이해한다.

## 한눈에 보는 핵심 요약

현재 변경의 중심은 `train()`이다.  
원본 `train()`은 byte-level BPE를 학습하는 과정에서 `id_to_token`이 사용하는 ID 체계와 다른 값을 시퀀스에 넣고 있었고, 새 merge 토큰도 `bytes`로 바로 만들어 저장했다.  
현재 `train()`은:

1. 학습 시작 시 내부 상태를 다시 초기화한다.
2. raw byte 값이 아니라 `BYTE_OFFSET`이 적용된 token ID 시퀀스로 학습한다.
3. 새 merge 토큰을 "합쳐진 bytes"가 아니라 `(left_id, right_id)` 튜플로 저장한다.

이 세 변화 때문에 `save/load`, `encode`, `decode`도 연쇄적으로 수정되었다.

## 변경 항목 목록

### 1. `_init_special_tokens()`

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:40)

원본 대비 변경점:
- 디버그 `print(...)`가 제거되었다.
- `self.token_to_id = {}` 재초기화가 추가되었다.

의미:
- 원본은 `_init_special_tokens()`를 여러 번 부르면 기존 `token_to_id` 위에 다시 `update()`만 했다.
- 현재는 재학습 시 이전 merge 매핑이 남지 않도록 `token_to_id`를 먼저 비운다.

왜 필요했나:
- 현재 `train()`은 시작할 때 `_init_special_tokens()`를 다시 호출한다.
- 이때 이전 학습 결과가 남아 있으면 새 학습과 old mapping이 섞일 수 있으므로, 초기화가 필요하다.

영향:
- 동작을 바꾼다기보다 "반복 호출 시 상태가 섞이지 않게" 만드는 안전장치다.

---

### 2. `find_freq_pair()`

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:94)

원본 대비 변경점:
- 빈 입력일 때 `None`을 바로 반환하는 guard가 추가되었다.
- 반환 타입 설명이 실제 구현에 맞게 바뀌었다.

원본 동작:
```python
pair_count = Counter(zip(byte_id_sequence, byte_id_sequence[1:]))
if pair_count.most_common(1)[0][1] >= 2:
    ...
```

문제:
- 시퀀스 길이가 0 또는 1이면 `most_common(1)[0]` 접근 자체가 실패할 수 있다.

현재 동작:
```python
if not pair_count:
    return None
```

의미:
- 예외를 없애고, "학습할 pair가 없음"을 정상 흐름으로 처리한다.

---

### 3. `train()` 핵심 변경

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:116)

이 함수가 이번 변경의 중심이다.

## `train()` 변경 전/후 개요

### 원본 `train()`의 핵심 흐름

```python
byte_id_sequence = list(corpus.encode("utf-8"))

while(self.find_freq_pair(byte_id_sequence) and len(self.merges) <= self.vocab_size-260):
    curr_pair = self.find_freq_pair(byte_id_sequence)
    self.merges.append(curr_pair)

    new_id = 259 + len(self.merges)
    new_token = self.id_to_token[curr_pair[0]] + self.id_to_token[curr_pair[1]]
    self.id_to_token[new_id] = new_token
    self.token_to_id[new_token] = new_id

    self.replace_pair(byte_id_sequence, curr_pair, new_id)
```

### 현재 `train()`의 핵심 흐름

```python
self.merges = []
self._init_special_tokens()

byte_id_sequence = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]

while(self.find_freq_pair(byte_id_sequence) and len(self.merges) < self.vocab_size-260):
    curr_pair = self.find_freq_pair(byte_id_sequence)

    self.merges.append(curr_pair)
    new_id = 259 + len(self.merges)
    self.id_to_token[new_id] = curr_pair
    self.token_to_id[curr_pair] = new_id
    self.replace_pair(byte_id_sequence, curr_pair, new_id)
```

## `train()` 세부 변경 분석

### 3-1. 학습 시작 시 상태 초기화 추가

추가된 코드:
```python
self.merges = []
self._init_special_tokens()
```

의미:
- 학습 전에 merge 규칙과 기본 vocab을 다시 만든다.

원본과 차이:
- 원본은 `train()` 안에서 `_init_special_tokens()`를 호출하지 않았다.
- 따라서 `train()` 호출 전에 사용자가 직접 `_init_special_tokens()`를 했는지에 따라 성공/실패가 갈릴 수 있었다.

왜 필요했나:
- `train()`이 독립적으로 동작해야 하기 때문이다.
- 테스트에서도 `tok.train(...)`만 호출하고 vocab이 준비되기를 기대한다.

추가로 중요한 점:
- `self.merges = []`는 재학습 시 이전 merge 규칙이 누적되지 않게 한다.
- 이 부분이 없으면 두 번째 `train()` 호출에서 이전 merge가 그대로 남는다.

---

### 3-2. 학습 시퀀스의 표현이 바뀜

원본:
```python
byte_id_sequence = list(corpus.encode("utf-8"))
```

현재:
```python
byte_id_sequence = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]
```

이 변화가 가장 중요하다.

원본 문제:
- `corpus.encode("utf-8")` 결과는 0~255 범위의 raw byte 값이다.
- 그런데 `_init_special_tokens()`가 등록한 byte token의 ID는 `4~259` 범위다.

예시:
- 공백 byte는 `32`
- 하지만 vocab 안의 공백 token ID는 `BYTE_OFFSET + 32 = 36`

원본에서 생기는 일:
- 학습 시퀀스에는 `32`가 들어간다.
- 이후 `self.id_to_token[curr_pair[0]]` 같은 lookup을 하면 `self.id_to_token[32]`를 찾게 된다.
- 실제 등록된 키는 `36`이라 `KeyError: 32`가 발생할 수 있다.

현재 방식의 의미:
- 학습 시퀀스와 vocab lookup이 같은 ID 체계를 쓰게 된다.
- 즉, 학습용 시퀀스의 값과 `id_to_token`의 키가 일치한다.

정리:
- 원본: "raw byte 값으로 학습"
- 현재: "token ID로 학습"

이 차이가 `train()` 변경의 가장 본질적인 수정이다.

---

### 3-3. 새 merge 토큰의 저장 방식이 바뀜

원본:
```python
new_token = self.id_to_token[curr_pair[0]] + self.id_to_token[curr_pair[1]]
self.id_to_token[new_id] = new_token
self.token_to_id[new_token] = new_id
```

현재:
```python
self.id_to_token[new_id] = curr_pair
self.token_to_id[curr_pair] = new_id
```

의미:
- 원본은 merge 결과를 실제 `bytes`로 미리 합쳐 저장하려고 했다.
- 현재는 merge 결과를 `(left_id, right_id)` 형태의 구조 정보로 저장한다.

왜 바뀌었나:
- 현재 데이터 모델에서는 merge token을 tuple로 저장하면 `self.merges`와의 대응이 직접적이다.
- `encode()`에서 `pair -> new_id` lookup이 쉬워진다.
- `decode()`에서 재귀적으로 원래 byte token까지 펼칠 수 있다.

원본 방식의 장단점:
- 장점: merge token이 바로 bytes라서 decode는 단순할 수 있다.
- 단점: 저장/복원 후에도 같은 구조를 안정적으로 유지하려면 bytes 결합 흐름과 ID lookup 흐름이 모두 맞아야 한다.
- 실제로 원본은 학습 시퀀스 ID 체계가 어긋나 있어 이 방식이 바로 깨졌다.

현재 방식의 장단점:
- 장점: merge rule 자체를 토큰에 보관하므로 추적과 복원이 명확하다.
- 단점: decode에서 재귀 복원이 필요하다.

#### 정확히 어떤 형태로 저장되는가

여기서 가장 헷갈리기 쉬운 부분은 "merge token이 실제로 `id_to_token` 안에 어떤 값으로 들어가느냐"이다.

생각할 수 있는 두 모델은 아래처럼 다르다.

| 모델 | 저장 예시 | 의미 |
| --- | --- | --- |
| bytes 저장 모델 | `id_to_token[260] = b"AB"` | merge 결과 바이트열 자체를 저장 |
| tuple 저장 모델 | `id_to_token[260] = (69, 70)` | "69번 토큰 뒤에 70번 토큰이 온다"는 구조를 저장 |

현재 코드는 두 번째 모델을 쓴다.

즉 현재 [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:137) 기준으로 merge token은:

```python
self.id_to_token[new_id] = curr_pair
```

형태로 저장된다.

예를 들어:

```python
id_to_token[69] = b"A"
id_to_token[70] = b"B"
id_to_token[260] = (69, 70)
```

이 뜻은:
- `260`번 토큰이 `b"AB"`를 직접 들고 있다는 뜻이 아니다.
- `260`번 토큰이 "69번 토큰 다음에 70번 토큰이 오는 구조"를 들고 있다는 뜻이다.

즉 현재 `260`은 "결과값" 저장이 아니라 "조합 규칙" 저장에 가깝다.

#### 왜 바이트가 이어진 형태처럼 보이지 않는가

현재 구조에서는 merge token이 완성된 bytes를 직접 보관하지 않는다.  
대신 "이 토큰은 왼쪽 하위 토큰과 오른쪽 하위 토큰으로 이루어졌다"는 정보만 가진다.

그래서:

```python
id_to_token[260] = (69, 70)
```

를 보면 즉시 `"AB"`처럼 안 보이지만, decode 시에는:

```python
decode(260)
= decode(69) + decode(70)
= b"A" + b"B"
= b"AB"
```

처럼 계산된다.

즉 `"AB"`가 저장되어 있는 게 아니라, `"AB"`를 다시 만들어낼 수 있는 구조가 저장되어 있는 것이다.

#### 중첩되면 트리 구조가 된다

이 구조는 한 단계 merge에서 끝나지 않고 계속 중첩될 수 있다.

예를 들어:

```python
id_to_token[69] = b"A"
id_to_token[70] = b"B"
id_to_token[71] = b"C"

id_to_token[260] = (69, 70)   # AB
id_to_token[261] = (260, 71)  # (AB) + C
```

이 경우 `261`은 `b"ABC"`를 직접 담고 있지 않다.  
대신 아래 같은 트리 구조를 담고 있다고 보는 편이 정확하다.

```text
261
├─ 260
│  ├─ 69 -> b"A"
│  └─ 70 -> b"B"
└─ 71 -> b"C"
```

즉 현재 merge token은 "이어진 바이트열"이라기보다 "바이트열을 다시 조립할 수 있는 구조"다.

#### bytes 모델과 tuple 모델의 차이

`b"AB"` 저장 모델과 `(69, 70)` 저장 모델의 차이를 한 줄씩 정리하면:

- bytes 모델: 결과를 바로 읽기 쉽다.
- tuple 모델: 결과를 읽으려면 decode 때 다시 풀어야 한다.
- bytes 모델: decode는 단순하지만, merge 구조 정보는 사라진다.
- tuple 모델: decode는 재귀가 필요하지만, 어떤 merge로 만들어졌는지 구조가 남는다.

현재 코드는 tuple 모델을 택했기 때문에, `decode()`가 "값을 바로 읽는 함수"가 아니라 "구조를 끝까지 펼치는 함수"가 되었다.

---

### 3-4. vocab 한계 조건이 미세 조정됨

원본:
```python
len(self.merges) <= self.vocab_size-260
```

현재:
```python
len(self.merges) < self.vocab_size-260
```

의미:
- merge를 몇 개까지 허용할지 off-by-one을 줄인 형태다.
- 기본 vocab이 260개이므로, 추가 merge 수는 최대 `vocab_size - 260`개여야 한다.
- `<`가 이 의도에 더 직접적으로 맞다.

실제 영향:
- vocab upper bound를 넘지 않게 하는 쪽으로 정리된 변경이다.

---

### 3-5. `train()` 변경의 결과 요약

`train()`은 이제 다음 성질을 가진다.

1. 단독 호출이 가능하다.
2. raw byte와 token ID를 혼동하지 않는다.
3. merge token을 구조적으로 보관한다.
4. 재학습 시 이전 merge 상태를 누적하지 않는다.

이 네 가지가 현재 `train()`의 핵심 변화다.

## 왜 `train()` 변경이 다른 함수 수정으로 이어졌는가

`train()`에서 merge token 저장 방식이 바뀌면, 다른 함수들도 같이 바뀔 수밖에 없다.

### 4. `save()`

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:144)

원본:
- `bytes`만 special handling 했다.

현재:
- `tuple`도 special handling 한다.

추가 코드:
```python
elif isinstance(token, tuple):
    serialized[str(token_id)] = {"type": "tuple", "value": list(token)}
```

왜 필요했나:
- JSON은 tuple을 직접 보존하지 못한다.
- 현재 `train()`이 merge token을 tuple로 저장하므로, 저장 시 type tag가 있어야 load 때 정확히 되살릴 수 있다.

원본 문제:
- tuple이 그냥 list로 저장되면, load 후 `token_to_id` 재구성 시 list를 dict key로 쓰려다 `TypeError: unhashable type: 'list'`가 날 수 있다.

---

### 5. `load()`

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:165)

현재 추가된 핵심:
```python
elif entry["type"] == "tuple":
    token = tuple(entry["value"])
```

의미:
- 저장된 merge token을 다시 hash 가능한 tuple로 복원한다.

왜 필요했나:
- 현재 `token_to_id`는 merge token tuple을 key로 사용한다.
- load 후에도 이 구조가 그대로 복원되어야 `encode()`가 `self.token_to_id[pair]` lookup을 할 수 있다.

---

### 6. `encode()`

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:187)

원본 대비 핵심 변경:

#### 6-1. merge 순회 버그 수정

원본:
```python
for pair in range(self.merges):
```

현재:
```python
for pair in self.merges:
```

원본 문제:
- `range()`는 정수를 받는다.
- `self.merges`는 list이므로 `range(self.merges)`는 바로 잘못된 코드다.

현재 의미:
- 저장된 merge pair들을 순서대로 순회한다.

#### 6-2. BOS/EOS 추가 방식 수정

원본:
```python
byte_id_list = BOS_TOKEN + byte_id_list + EOS_TOKEN
```

현재:
```python
byte_id_list = [self.get_bos_id()] + byte_id_list + [self.get_eos_id()]
```

원본 문제:
- `BOS_TOKEN`, `EOS_TOKEN`은 문자열이다.
- `byte_id_list`는 정수 ID 리스트다.
- 문자열과 리스트를 이어붙이는 것은 타입 자체가 맞지 않는다.

현재 의미:
- 특수 토큰도 실제 token ID로 앞뒤에 붙는다.

---

### 7. `decode()`

현재 위치: [src/bpe.py](/Users/wiseungcheol/Desktop/LLM_project/W14-gpt-lab/src/bpe.py:214)

원본:
```python
for id in ids:
    if skip_special and id < BYTE_OFFSET:
        continue
    byte_buffer += self.id_to_token[id]
```

현재:
```python
def append_token_bytes(token_id):
    token = self.id_to_token[token_id]
    if isinstance(token, bytes):
        byte_buffer.extend(token)
    elif isinstance(token, tuple):
        append_token_bytes(token[0])
        append_token_bytes(token[1])
    elif isinstance(token, str) and not skip_special:
        byte_buffer.extend(token.encode("utf-8"))
```

왜 바뀌었나:
- 원본은 `self.id_to_token[id]`가 bytes라고 가정한다.
- 하지만 현재 `train()`은 merge token을 tuple로 저장한다.
- 따라서 decode는 tuple을 만나면 원래 byte token까지 재귀적으로 펼쳐야 한다.

의미:
- 현재 decode는 "leaf byte token까지 내려가서 byte_buffer를 구성"하는 방식이다.

## `decode()`를 머릿속에 그리는 방법

현재 `decode()`는 "토큰 ID를 바로 글자로 바꾸는 함수"라기보다,  
"토큰 트리(tree)를 끝까지 내려가서 원래 byte들을 모은 다음, 마지막에 한 번만 문자열로 바꾸는 함수"라고 생각하면 이해가 쉽다.

핵심 아이디어는 이렇다.

1. `ids`에는 byte token도 있고 merge token도 있을 수 있다.
2. byte token은 바로 실제 byte를 담고 있다.
3. merge token은 실제 byte를 직접 담고 있지 않고, "왼쪽 token + 오른쪽 token" 구조만 담고 있다.
4. 따라서 merge token을 만나면 그 안으로 다시 들어가야 한다.
5. 끝까지 들어가서 byte token만 모으면, 그 byte들을 합쳐 원문을 복원할 수 있다.

즉, 현재 `decode()`는 다음 두 단계로 생각하면 된다.

- 1단계: 토큰 구조를 byte들로 평탄화한다.
- 2단계: 모인 byte 전체를 UTF-8로 한 번 decode한다.

## 함수 내부 동작을 순서대로 보면

### 단계 1. 빈 byte 버퍼를 만든다

```python
byte_buffer = bytearray()
```

역할:
- 최종적으로 복원할 모든 byte를 여기에 순서대로 쌓는다.

중요한 점:
- 여기서는 아직 문자열을 만들지 않는다.
- 중간에 token 하나씩 `decode()` 하지 않고, byte만 계속 쌓는다.

왜 이렇게 하나:
- UTF-8 문자는 1바이트가 아닐 수 있다.
- 특히 한글은 여러 byte가 모여야 한 글자가 된다.
- 그래서 중간중간 따로 decode하면 깨질 수 있고, 마지막에 한 번만 decode하는 것이 안전하다.

---

### 단계 2. `append_token_bytes(token_id)`라는 재귀 helper를 정의한다

```python
def append_token_bytes(token_id):
    token = self.id_to_token[token_id]
```

이 helper의 역할은 아주 단순하다.

- 입력: token ID 하나
- 출력: 그 token이 나타내는 "원래 byte들"을 `byte_buffer` 뒤에 붙임

즉 이 helper는 "이 토큰 하나를 실제 byte열로 풀어라"라는 일을 맡는다.

---

### 단계 3. token이 `bytes`면 바로 버퍼에 추가한다

```python
if isinstance(token, bytes):
    byte_buffer.extend(token)
```

이 경우는 가장 단순하다.

의미:
- 이 token은 더 쪼갤 필요가 없는 leaf token이다.
- 이미 실제 byte를 담고 있으므로 바로 buffer 뒤에 붙이면 된다.

머릿속 이미지:
- 나무 구조에서 "리프 노드"를 만난 상태다.
- 더 내려갈 필요 없이 값을 가져오면 된다.

예:
- `id_to_token[69] == b"A"`
- `append_token_bytes(69)`를 호출하면 버퍼에 `b"A"`가 붙는다.

---

### 단계 4. token이 `tuple`이면 왼쪽, 오른쪽을 다시 푼다

```python
elif isinstance(token, tuple):
    append_token_bytes(token[0])
    append_token_bytes(token[1])
```

이 부분이 현재 `decode()`의 핵심이다.

의미:
- 이 token은 실제 byte가 아니라 "두 token을 합쳐 만든 merge token"이다.
- 따라서 자기 자신을 바로 문자열로 바꾸는 게 아니라,
  안에 들어 있는 두 하위 token을 순서대로 다시 decode해야 한다.

아주 중요:
- `token[0]`을 먼저 풀고
- `token[1]`을 나중에 푼다

그래야 원래 순서가 보존된다.

즉 merge token `(left_id, right_id)`는 decode 시:

```python
decode(left_id) + decode(right_id)
```

와 같은 의미를 가진다.

머릿속 이미지:
- merge token은 "완성된 글자"가 아니라 "작은 블록 2개를 이어붙인 구조물"이다.
- 그래서 구조물을 바로 읽는 게 아니라, 왼쪽 블록부터 해체하고 오른쪽 블록을 해체해야 한다.

---

### 단계 5. 특수 토큰은 옵션에 따라 건너뛴다

실제 loop 쪽에서는:

```python
for token_id in ids:
    if skip_special and token_id < BYTE_OFFSET:
        continue
    append_token_bytes(token_id)
```

의미:
- `0~3`은 `<pad>`, `<unk>`, `<bos>`, `<eos>` 같은 special token이다.
- 보통 원문 복원에서는 이런 토큰을 출력하지 않으므로 `skip_special=True`면 그냥 무시한다.

중요한 점:
- special token을 건너뛰는 위치는 바깥 loop다.
- 즉 "입력 ID 목록을 순회할 때" 먼저 거르고, 통과한 것만 helper로 보낸다.

---

### 단계 6. 모든 byte를 모은 뒤 마지막에 한 번만 decode한다

```python
result = byte_buffer.decode("utf-8")
return result
```

이제야 비로소 문자열을 만든다.

의미:
- 앞 단계에서 모은 건 모두 "원래 텍스트를 이루는 실제 UTF-8 byte들"이다.
- 따라서 이 전체 byte열을 한 번만 decode하면 원문 문자열이 나온다.

왜 마지막에 한 번만 하나:
- UTF-8은 문자 경계가 byte 경계와 항상 일치하지 않는다.
- 예를 들어 한글 한 글자는 여러 byte로 이뤄질 수 있다.
- 그래서 byte를 조금씩 잘라 decode하면 잘못될 수 있다.

## 아주 단순한 예시 1: merge가 없는 경우

입력:

```python
ids = [69, 70]
```

가정:

```python
id_to_token[69] = b"A"
id_to_token[70] = b"B"
```

실행 흐름:

1. `byte_buffer = bytearray()`
2. `append_token_bytes(69)` 호출
3. `69`는 bytes token이므로 버퍼에 `b"A"` 추가
4. `append_token_bytes(70)` 호출
5. `70`도 bytes token이므로 버퍼에 `b"B"` 추가
6. 최종 buffer는 `b"AB"`
7. `b"AB".decode("utf-8") == "AB"`

이 경우는 원본 방식과 크게 다르지 않다.

## 예시 2: merge token이 1단계인 경우

가정:

```python
id_to_token[69] = b"A"
id_to_token[70] = b"B"
id_to_token[260] = (69, 70)
```

입력:

```python
ids = [260]
```

이때 `260`은 "A와 B를 합쳐 만든 토큰"이다.

실행 흐름:

1. `append_token_bytes(260)` 호출
2. `id_to_token[260]`은 `(69, 70)`이므로 tuple branch로 감
3. 먼저 `append_token_bytes(69)` 호출
4. `69`는 bytes token이므로 버퍼에 `b"A"` 추가
5. 다음 `append_token_bytes(70)` 호출
6. `70`는 bytes token이므로 버퍼에 `b"B"` 추가
7. 최종 buffer는 `b"AB"`
8. 최종 decode 결과는 `"AB"`

중요한 포인트:
- `260` 자체가 `b"AB"`를 직접 들고 있는 게 아니다.
- 대신 `(69, 70)` 구조를 들고 있고, decode가 그 구조를 따라 내려가면서 `b"A"`, `b"B"`를 찾아온다.

## 예시 3: merge token이 여러 단계로 중첩된 경우

이게 현재 구조를 이해하는 데 가장 중요하다.

가정:

```python
id_to_token[69] = b"A"
id_to_token[70] = b"B"
id_to_token[71] = b"C"

id_to_token[260] = (69, 70)   # "AB"
id_to_token[261] = (260, 71)  # "AB" + "C"
```

입력:

```python
ids = [261]
```

실행 흐름을 호출 스택 관점에서 써보면:

1. `append_token_bytes(261)`
2. `261`은 tuple이므로 `append_token_bytes(260)` 호출
3. `260`도 tuple이므로 `append_token_bytes(69)` 호출
4. `69`는 bytes이므로 버퍼에 `b"A"` 추가
5. `260`으로 돌아와서 `append_token_bytes(70)` 호출
6. `70`는 bytes이므로 버퍼에 `b"B"` 추가
7. `261`로 돌아와서 `append_token_bytes(71)` 호출
8. `71`는 bytes이므로 버퍼에 `b"C"` 추가
9. 최종 buffer는 `b"ABC"`
10. 최종 decode 결과는 `"ABC"`

이 과정을 트리로 그리면:

```text
261
├─ 260
│  ├─ 69 -> b"A"
│  └─ 70 -> b"B"
└─ 71 -> b"C"
```

`decode()`는 이 트리를 왼쪽부터 끝까지 내려가며 리프 bytes를 모은다고 보면 된다.

## 현재 `decode()`를 한 문장으로 표현하면

현재 `decode()`는:

> "입력된 token ID들을 왼쪽부터 순회하면서, merge token이면 재귀적으로 해체하고, 최종적으로 얻은 byte leaf들을 차례대로 이어 붙인 뒤, 마지막에 UTF-8 문자열로 복원하는 함수"

라고 볼 수 있다.

## 원본 `decode()`와의 사고방식 차이

원본 사고방식:
- `id_to_token[id]`는 바로 bytes일 것이다.
- 따라서 그냥 버퍼에 더하면 된다.

현재 사고방식:
- `id_to_token[id]`는 bytes일 수도 있고 tuple일 수도 있다.
- tuple이면 아직 "완성된 byte열"이 아니라 "더 풀어야 하는 구조"다.
- 따라서 바로 더하지 말고, 끝까지 분해해서 bytes만 모아야 한다.

즉 원본 decode는 "값을 읽는" 느낌이고,  
현재 decode는 "구조를 순회해서 잎사귀(byte)만 수집하는" 느낌이다.

## `decode()`가 현재 구조에서 꼭 필요한 이유

현재 `train()`은 merge token을 이렇게 저장한다.

```python
self.id_to_token[new_id] = curr_pair
```

즉 merge token은 bytes가 아니라 tuple 구조다.

그래서 decode는 반드시 아래 질문에 답할 수 있어야 한다.

- 이 token이 bytes인가?
- 아니면 다른 token 둘을 가리키는 merge 구조인가?
- merge 구조라면, 그 내부를 다시 풀었을 때 최종 byte 순서는 무엇인가?

현재 `append_token_bytes()`는 정확히 이 질문을 해결하는 장치다.

이 변화는 `train()`의 merge token 표현 변경에 직접 연결된다.

## 변경의 인과관계

이번 변경은 서로 독립적인 수정들의 집합이 아니라, `train()`을 중심으로 이어지는 연쇄 수정이다.

순서를 따라 보면:

1. `train()`이 raw byte 대신 token ID를 써야 했다.
2. `train()`이 새 merge token을 tuple로 저장하게 되었다.
3. 그래서 `save()`는 tuple을 저장할 수 있어야 했다.
4. 그래서 `load()`는 tuple을 복원할 수 있어야 했다.
5. 그래서 `encode()`는 `pair -> new_id`를 tuple key로 lookup하게 되었다.
6. 그래서 `decode()`는 tuple merge token을 재귀적으로 펼칠 수 있어야 했다.

즉, 이번 수정의 뿌리는 `train()`이고, 나머지는 그 구조를 따라간 변경이다.

## 변경 전/후 작동 방식 비교

### 원본 작동 방식

- byte sequence를 raw byte 값으로 만든다.
- pair를 찾는다.
- 새 merge token을 bytes 결합 결과로 저장한다.
- encode는 merge 리스트를 적용하려 하지만 `range(self.merges)` 버그가 있다.
- decode는 `id_to_token[id]`가 bytes라고 가정한다.

취약점:
- raw byte와 token ID 체계가 어긋난다.
- merge token 저장 방식과 save/load/encode/decode 계약이 완전히 맞물리지 않는다.

### 현재 작동 방식

- byte sequence를 token ID 시퀀스로 만든다.
- pair를 찾는다.
- 새 merge token은 `(left_id, right_id)` tuple로 저장한다.
- encode는 tuple pair를 순서대로 적용해 새 ID로 치환한다.
- decode는 tuple을 재귀적으로 펼쳐 원래 bytes로 되돌린다.

장점:
- 학습, 저장, 복원, 인코딩, 디코딩이 같은 데이터 모델을 공유한다.

## 특히 `train()` 관점에서 봐야 할 핵심 3가지

`train()`을 이해할 때 가장 중요한 포인트는 아래 셋이다.

### 1. 학습 단위가 raw byte가 아니라 token ID다

```python
byte_id_sequence = [BYTE_OFFSET + b for b in corpus.encode("utf-8")]
```

이 한 줄이 `KeyError: 32` 문제를 막는다.

### 2. merge token은 bytes가 아니라 구조 정보다

```python
self.id_to_token[new_id] = curr_pair
```

즉 새 토큰은 "결과 바이트열"이 아니라 "어떤 두 토큰을 합쳤는지"를 저장한다.

### 3. `train()`이 독립적으로 실행 가능하다

```python
self.merges = []
self._init_special_tokens()
```

이제 `train()`은 호출 전에 별도 준비를 기대하지 않는다.

## 최소 수정 관점에서 본 평가

현재 diff를 기능별로 나눠 보면:

### 꼭 필요했던 변경

- `train()`의 `BYTE_OFFSET` 적용
- `train()`의 `_init_special_tokens()` 호출
- `train()`의 merge token 저장 방식 변경
- `save()/load()`의 tuple 직렬화/복원
- `encode()`의 `for pair in self.merges`
- `encode()`의 BOS/EOS ID 처리
- `decode()`의 tuple 재귀 복원

### 부수적이지만 타당한 변경

- `_init_special_tokens()`의 debug print 제거
- `_init_special_tokens()`의 `self.token_to_id = {}`
- `find_freq_pair()`의 빈 입력 guard
- `find_freq_pair()`의 반환 타입 설명 정리

## 결론

이번 변경은 단순히 버그 몇 줄을 고친 정도가 아니라, `train()`을 기준으로 토크나이저 내부 데이터 모델을 정렬한 수정이다.

원본 코드의 가장 큰 문제는:
- 학습 시퀀스는 raw byte를 쓰고
- vocab lookup은 offset된 token ID를 쓰며
- merge token 저장 방식은 bytes인데
- save/load/encode/decode가 그 표현을 끝까지 안정적으로 유지하지 못했다는 점이다.

현재 코드는 이 불일치를 다음 방식으로 정리했다.

1. 학습은 token ID 기준으로 수행한다.
2. merge token은 tuple 구조로 저장한다.
3. 저장/복원/인코딩/디코딩이 그 구조를 일관되게 따른다.

그래서 `train()`을 이해하려면 "이제 학습은 byte가 아니라 token ID 위에서 돈다"는 점을 가장 먼저 잡고 보는 것이 좋다.
