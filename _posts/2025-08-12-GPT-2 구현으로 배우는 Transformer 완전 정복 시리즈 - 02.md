---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 02
date: 2025-08-12 08:10:18 +0900
categories: [artificial intelligence, machine learning]
tags: [machine learning, gpt, transformer, tokenization, bpe, embedding, nlp, deep, learning, pytorch, tiktoken, subword]
---

# GPT 완전 정복 2편: 데이터가 어떻게 "단어"가 되는가? - 토큰화의 비밀

> **이전 편 요약**: 1편에서는 GPT가 "다음 단어 맞히기"를 통해 언어의 모든 것을 학습한다는 혁신적 아이디어를 배웠습니다. 이번 편에서는 컴퓨터가 어떻게 텍스트를 "이해할 수 있는 숫자"로 변환하는지 알아보겠습니다.

---

## 들어가며: 컴퓨터는 왜 텍스트를 못 읽을까?

인간에게는 너무나 자연스러운 일이 컴퓨터에게는 불가능합니다:

```
인간이 보는 것: "Hello, world!"
컴퓨터가 보는 것: ??? (의미 없는 문자 배열)
```

컴퓨터는 오직 **숫자**만 계산할 수 있습니다. 그렇다면 어떻게 "Hello, world!"를 숫자로 바꿀 수 있을까요?

**이것이 바로 토큰화(Tokenization)의 핵심 문제입니다.**

---

## 1. 토큰화의 진화: 단순함에서 정교함으로

### 1단계: 문자 단위 토큰화 (Character-level)

가장 단순한 방법부터 시작해보겠습니다:

```python
# 문자 하나씩 숫자로 매핑
char_to_id = {'H': 1, 'e': 2, 'l': 3, 'o': 4, ',': 5, ' ': 6, 'w': 7, 'r': 8, 'd': 9, '!': 10}

text = "Hello, world!"
tokens = [char_to_id[char] for char in text]
print(tokens)  # [1, 2, 3, 3, 4, 5, 6, 7, 4, 8, 3, 9, 10]
```

**장점:**
- 구현이 매우 간단
- 어떤 언어든 처리 가능
- 어휘 크기가 작음 (영어는 ~100개)

**치명적 단점:**
- 의미 단위가 너무 작음
- "dog"와 "dogs"를 완전히 다른 것으로 인식
- 학습할 패턴이 너무 많음

### 2단계: 단어 단위 토큰화 (Word-level)

```python
# 단어 하나씩 숫자로 매핑
word_to_id = {'Hello': 1, 'world': 2, '!': 3, ',': 4}

text = "Hello, world!"
words = text.split()  # ['Hello,', 'world!']
# 문제 발생: "Hello,"와 "Hello"를 다르게 인식!
```

**장점:**
- 의미 단위가 명확
- 인간의 언어 직관과 일치

**치명적 단점:**
- 어휘 크기 폭발 (영어 단어 수십만 개)
- 새로운 단어 처리 불가 (Out-of-Vocabulary)
- 언어별로 다른 전처리 필요

### 3단계: 서브워드 토큰화 - BPE의 등장

**Byte Pair Encoding(BPE)**가 혁명을 일으켰습니다!

핵심 아이디어: **자주 나타나는 문자 조합을 점진적으로 합치기**

---

## 2. BPE 알고리즘: 단계별 완전 분석

### BPE 학습 과정 시뮬레이션

실제 예시로 BPE가 어떻게 작동하는지 보겠습니다:

```python
# 초기 말뭉치 (공백으로 단어 구분, </w>로 단어 끝 표시)
corpus = [
    "l o w </w>",      # low
    "l o w e r </w>",  # lower  
    "n e w e s t </w>", # newest
    "w i d e s t </w>"  # widest
]
```

#### 1단계: 문자별 빈도 계산
```
단일 문자 빈도:
'l': 2, 'o': 2, 'w': 4, 'e': 4, 'r': 1, 'n': 1, 's': 2, 't': 2, 'i': 1, 'd': 1
```

#### 2단계: 인접한 쌍의 빈도 계산
```
문자 쌍 빈도:
('l', 'o'): 2      # "low", "lower"에서
('o', 'w'): 2      # "low", "lower"에서  
('w', '</w>'): 2   # "low", "widest"에서
('e', 's'): 2      # "newest", "widest"에서
('s', 't'): 2      # "newest", "widest"에서
...
```

#### 3단계: 가장 빈번한 쌍 병합
가장 빈번한 쌍들을 찾아 병합합니다:

**1차 병합: ('e', 's') → 'es'**
```
l o w </w>
l o w e r </w>  
n e w es t </w>
w i d es t </w>
```

**2차 병합: ('es', 't') → 'est'**
```
l o w </w>
l o w e r </w>
n e w est </w>
w i d est </w>
```

**3차 병합: ('l', 'o') → 'lo'**
```
lo w </w>
lo w e r </w>
n e w est </w>
w i d est </w>
```

### BPE의 최종 결과

수천 번의 병합 후 얻어지는 토큰들:
```python
# 최종 BPE 어휘
vocab = [
    # 단일 문자
    'a', 'b', 'c', ..., 
    # 자주 쓰이는 조합들
    'ing', 'ed', 'er', 'est', 'ion',
    # 완전한 단어들  
    'the', 'and', 'to', 'of',
    # 특수 토큰
    '<|endoftext|>', '</w>'
]
```

---

## 3. GPT-2의 토큰화: tiktoken 라이브러리 완전 분석

### 실제 구현에서 토큰화 확인

[우리 레포지토리](https://github.com/BanHun28/gpt2_study)의 코드를 보면:

```python
# main.py의 get_data() 함수에서
def get_data():
    # 셰익스피어 텍스트 다운로드 후...
    
    try:
        import tiktoken  # OpenAI의 GPT-2 토크나이저
        enc = tiktoken.get_encoding("gpt2")  # GPT-2 인코더
        data = enc.encode(text)  # 텍스트 → 토큰 ID 리스트
        print(f"데이터 크기: {len(data):,} 토큰")
    except ImportError:
        print("tiktoken이 설치되지 않았습니다.")
        sys.exit(1)
```

### tiktoken 실제 사용해보기

```python
import tiktoken

# GPT-2와 동일한 토크나이저 로드
enc = tiktoken.get_encoding("gpt2")

# 다양한 텍스트 토큰화 실험
test_texts = [
    "Hello, world!",
    "안녕하세요",  # 한글
    "programming",
    "unprogrammable",  # 복합어
    "ChatGPT is amazing!",
]

for text in test_texts:
    tokens = enc.encode(text)
    decoded = enc.decode(tokens)
    
    print(f"원본: {text}")
    print(f"토큰: {tokens}")
    print(f"복원: {decoded}")
    print(f"토큰 수: {len(tokens)}")
    print("-" * 40)
```

**실행 결과:**
```
원본: Hello, world!
토큰: [15496, 11, 995, 0]
복원: Hello, world!
토큰 수: 4

원본: programming  
토큰: [23065]
복원: programming
토큰 수: 1

원본: unprogrammable
토큰: [403, 23065, 76, 540]  # "un" + "program" + "m" + "able" 
복원: unprogrammable
토큰 수: 4
```

### 놀라운 발견들

#### 1. 효율적인 압축
```python
# 긴 텍스트도 효율적으로 압축
long_text = "The quick brown fox jumps over the lazy dog. " * 100
tokens = enc.encode(long_text)

print(f"원본 문자 수: {len(long_text):,}")      # 4,300
print(f"토큰 수: {len(tokens):,}")              # 약 1,000
print(f"압축률: {len(tokens)/len(long_text):.2%}")  # 약 23%
```

#### 2. 언어별 차이
```python
# 영어 vs 한글 토큰화 효율성 비교
english = "This is a test sentence."
korean = "이것은 테스트 문장입니다."

eng_tokens = enc.encode(english)
kor_tokens = enc.encode(korean)

print(f"영어 - 문자 수: {len(english)}, 토큰 수: {len(eng_tokens)}")
print(f"한글 - 문자 수: {len(korean)}, 토큰 수: {len(kor_tokens)}")
# 한글이 토큰 수가 더 많음 - GPT-2가 영어 중심으로 학습되었기 때문
```

---

## 4. 임베딩: 토큰을 의미있는 벡터로

### 토큰화는 시작일 뿐

토큰 ID는 여전히 의미 없는 숫자입니다:
```
"cat" → 7163 (그냥 숫자)
"dog" → 3290 (그냥 숫자)
```

**문제**: 7163과 3290이 둘 다 "동물"이라는 걸 컴퓨터가 어떻게 알까요?

### 임베딩의 마법

```python
# main.py의 GPT 클래스에서
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict(dict(
            # 토큰 ID → 의미 벡터 변환
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # 50257개 토큰 각각을 768차원 벡터로 표현
        ))
```

#### 임베딩의 작동 원리

```python
# 예시: 간단한 임베딩 테이블
embedding_table = {
    7163: [0.2, -0.5, 1.3, 0.8, ...],  # "cat" 벡터 (768차원)
    3290: [0.1, -0.3, 1.1, 0.9, ...],  # "dog" 벡터 (768차원)  
    # ...
}

# 벡터 간 유사도 계산 가능
cat_vector = embedding_table[7163]
dog_vector = embedding_table[3290]
similarity = cosine_similarity(cat_vector, dog_vector)  # 높은 유사도!
```

#### 학습을 통한 의미 획득

```python
# 학습 과정에서 벡터들이 의미를 획득
초기 상태:
"cat": [0.01, 0.02, -0.01, ...]  # 랜덤 벡터
"dog": [-0.02, 0.01, 0.03, ...]  # 랜덤 벡터

수십만 번의 학습 후:
"cat": [0.2, -0.5, 1.3, ...]    # 동물의 특성을 나타내는 벡터
"dog": [0.1, -0.3, 1.1, ...]    # 동물의 특성을 나타내는 벡터
```

---

## 5. 위치 임베딩: 단어의 "자리"가 중요한 이유

### 순서가 바뀌면 의미도 바뀜

```
"나는 학교에 간다" vs "학교에 나는 간다"
```

Transformer는 순서 정보가 없으므로 **위치 임베딩**이 필요합니다.

```python
# main.py에서
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),    # 단어 임베딩
            wpe=nn.Embedding(config.block_size, config.n_embd),    # 위치 임베딩
        ))
    
    def forward(self, idx):
        b, t = idx.size()
        
        # 위치 인덱스 생성 (0, 1, 2, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # 단어와 위치 임베딩을 더함
        tok_emb = self.transformer.wte(idx)    # 단어 임베딩
        pos_emb = self.transformer.wpe(pos)    # 위치 임베딩  
        x = tok_emb + pos_emb                  # 최종 입력
```

#### 위치 임베딩 시각화

```
문장: "나는 학교에 간다"
토큰: [나는, 학교에, 간다]

단어 임베딩:
나는:   [0.2, 0.5, -0.1, ...]
학교에: [0.8, -0.2, 0.6, ...]  
간다:   [-0.1, 0.7, 0.3, ...]

위치 임베딩:
위치0:  [0.1, 0.0, 0.2, ...]
위치1:  [0.0, 0.1, -0.1, ...]
위치2:  [-0.1, 0.2, 0.0, ...]

최종 입력 (단어 + 위치):
나는:   [0.3, 0.5, 0.1, ...]  # 단어 + 위치0
학교에: [0.8, -0.1, 0.5, ...]  # 단어 + 위치1
간다:   [-0.2, 0.9, 0.3, ...]  # 단어 + 위치2
```

---

## 6. 실전 분석: 셰익스피어 데이터 토큰화

### 우리 프로젝트의 데이터 분석

```python
# main.py의 get_data() 함수 실행 결과 분석
def analyze_shakespeare_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    
    print(f"전체 텍스트 길이: {len(text):,} 문자")
    print(f"토큰 수: {len(tokens):,}")
    print(f"압축률: {len(tokens)/len(text):.2%}")
    
    # 고유 토큰 수 확인
    unique_tokens = set(tokens)
    print(f"고유 토큰 수: {len(unique_tokens):,}")
    print(f"전체 어휘 중 사용률: {len(unique_tokens)/50257:.2%}")
```

**예상 결과:**
```
전체 텍스트 길이: 1,115,394 문자
토큰 수: 338,025
압축률: 30.31%
고유 토큰 수: 16,738
전체 어휘 중 사용률: 33.31%
```

### 자주 나오는 토큰들 분석

```python
from collections import Counter

def analyze_frequent_tokens():
    # 토큰 빈도 분석
    token_counts = Counter(tokens)
    
    # 상위 20개 토큰
    for token_id, count in token_counts.most_common(20):
        token_text = enc.decode([token_id])
        print(f"'{token_text}': {count:,}회")
```

**예상 결과:**
```
' ': 50,234회     # 공백이 가장 많음
'the': 27,595회   # 관사 
'and': 26,285회   # 접속사
'to': 25,606회    # 전치사
...
```

---

## 7. 토큰화의 한계와 해결책

### 현재 토큰화의 문제점들

#### 1. 언어별 편향
```python
# 영어 vs 다른 언어 효율성 차이
english = "Hello world"      # 2 토큰
korean = "안녕하세요"         # 5 토큰 (비효율적!)
chinese = "你好世界"          # 6 토큰
```

#### 2. 새로운 도메인 적응 어려움
```python
# 코딩 관련 텍스트
code = "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"
tokens = enc.encode(code)
print(f"코드 토큰화: {len(tokens)}개 토큰")  # 비효율적으로 많은 토큰
```

#### 3. 토큰 경계의 애매함
```python
# 같은 의미, 다른 토큰화
text1 = "GPT-2"      # [GP, T, -, 2] - 4개 토큰
text2 = "GPT 2"      # [GP, T,  , 2] - 4개 토큰 (다른 구성)
```

### 차세대 토큰화 기법들

#### 1. SentencePiece
- 구글에서 개발
- 언어 독립적
- 더 균등한 언어 처리

#### 2. WordPiece
- BERT에서 사용
- [UNK] 토큰 최소화

#### 3. Unigram Language Model
- 확률 기반 토큰화
- 더 정교한 어휘 선택

---

## 8. 실습: 직접 토큰화 체험하기

### 실습 1: tiktoken 완전 정복

```python
# 1. 기본 토큰화
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# 다양한 텍스트 실험
texts = [
    "ChatGPT",
    "자연어처리",  
    "programming123",
    "Hello, 안녕하세요!",
    "🚀🤖",  # 이모지
]

for text in texts:
    tokens = enc.encode(text)
    decoded = enc.decode(tokens)
    print(f"{text} → {tokens} → {decoded}")
```

### 실습 2: 임베딩 차원 이해하기

```python
# 2. 임베딩 차원 실험
import torch
import torch.nn as nn

# 간단한 임베딩 실험
vocab_size = 1000  # 작은 어휘
embed_dim = 128    # 작은 차원

embedding = nn.Embedding(vocab_size, embed_dim)

# 토큰 ID → 벡터 변환
token_ids = torch.tensor([1, 15, 234, 567])
embeddings = embedding(token_ids)

print(f"토큰 IDs: {token_ids}")
print(f"임베딩 shape: {embeddings.shape}")  # [4, 128]
print(f"첫 번째 토큰 벡터: {embeddings[0][:10]}")  # 처음 10차원만
```

### 실습 3: 위치 임베딩 효과 확인

```python
# 3. 위치의 중요성 실험
def compare_with_without_position():
    text = "나는 학교에 간다"
    tokens = enc.encode(text)
    
    # 위치 정보 없이
    word_only = embedding(torch.tensor(tokens))
    
    # 위치 정보 포함
    positions = torch.arange(len(tokens))
    pos_emb = position_embedding(positions)
    word_with_pos = word_only + pos_emb
    
    print("위치 정보가 얼마나 다른 표현을 만드는지 확인 가능")
```

---

## 마무리: 숫자가 된 언어

### 오늘 배운 핵심 내용

1. **토큰화의 진화**: 문자 → 단어 → 서브워드(BPE)
2. **BPE 알고리즘**: 자주 나오는 조합을 점진적으로 병합
3. **tiktoken**: GPT-2와 동일한 토크나이저 실제 사용
4. **임베딩**: 토큰 ID → 의미있는 벡터 변환
5. **위치 임베딩**: 단어 순서 정보 추가

### 지금까지의 데이터 플로우

```
"Hello, world!" 
    ↓ (토큰화)
[15496, 11, 995, 0]
    ↓ (단어 임베딩)  
[[0.2, -0.5, ...], [0.8, 0.3, ...], ...]
    ↓ (위치 임베딩 추가)
[[0.3, -0.4, ...], [0.8, 0.4, ...], ...]  ← Transformer 입력 준비 완료!
```

### 다음 편 예고: Attention의 마법

다음 편에서는 드디어 **Transformer의 핵심인 Attention**을 배웁니다!

- **Query, Key, Value**가 정확히 무엇인지
- 단어들이 어떻게 **서로 소통**하는지  
- **Multi-Head Attention**이 왜 필요한지
- **Causal Masking**으로 미래를 차단하는 방법

**미리 생각해볼 질문:**
"나는 학교에 간다"에서 "간다"를 예측할 때, "나는"과 "학교에" 중 어느 것이 더 중요할까요? Attention이 바로 이런 **중요도를 자동으로 계산**하는 메커니즘입니다!

### 실습 과제

다음 편까지 해볼 과제:

1. **tiktoken 실험**: 다양한 언어로 토큰화 효율성 비교
2. **어휘 분석**: 셰익스피어 데이터에서 가장 자주 나오는 토큰 찾기
3. **코드 실행**: `python main.py`에서 토큰화 과정 관찰

다음 편에서 Attention의 신비로운 세계로 들어가봅시다! 🔍

