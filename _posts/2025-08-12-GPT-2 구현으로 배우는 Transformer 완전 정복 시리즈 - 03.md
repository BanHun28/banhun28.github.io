---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 03
date: 2025-08-12 08:10:20 +0900
categories: [machine learning, GPT]
tags: [machine learning, GPT, Transformer]      # TAG names should always be lowercase
---

# GPT 완전 정복 3편: Attention의 마법 - 컴퓨터가 "문맥"을 이해하는 방법

> **이전 편 요약**: 2편에서는 텍스트가 어떻게 토큰으로 변환되고, 임베딩을 통해 의미 있는 벡터가 되는지 배웠습니다. 이제 이 벡터들이 어떻게 서로 "소통"하여 문맥을 이해하는지 알아보겠습니다.

---

## 들어가며: 인간은 어떻게 문맥을 이해할까?

다음 문장을 읽어보세요:

```
"그 남자는 공원에서 개를 산책시키고 있었다. 개가 갑자기 뛰어갔다."
```

두 번째 문장의 "개"를 읽는 순간, 우리는 자동으로 첫 번째 문장의 "개"와 연결합니다. 이것이 바로 **문맥 이해**입니다.

**컴퓨터는 어떻게 이런 연결을 만들 수 있을까요?**

이것이 바로 **Attention 메커니즘**이 해결하는 핵심 문제입니다.

---

## 1. Attention의 직관적 이해: 도서관에서 정보 찾기

### Attention = 똑똑한 검색 시스템

도서관에서 책을 찾는 상황을 상상해보세요:

```
상황: "셰익스피어의 햄릿에 대한 정보가 필요해"

1. Query (질문): "햄릿에 대해 알고 싶다"
2. Key (색인): 각 책의 제목과 키워드들
3. Value (내용): 각 책의 실제 내용

과정:
1. 사서가 질문을 듣는다 (Query)
2. 도서 목록에서 관련 있는 책들을 찾는다 (Key와 Query 매칭)
3. 관련도에 따라 가중치를 부여한다 (Attention Score)
4. 가장 관련 있는 책들의 내용을 종합해서 답한다 (Weighted Sum of Values)
```

### 실제 문장에서의 Attention

```
문장: "나는 학교에 간다"
현재 예측할 단어: "간다"

Query: "간다"가 묻는 질문 - "나는 누구이고, 어디로 가는가?"
Key: 각 단어가 답할 수 있는 것
- "나는": "주체에 대한 정보를 줄 수 있어"
- "학교에": "목적지에 대한 정보를 줄 수 있어"
Value: 실제로 전달할 정보  
- "나는": [주체, 1인칭, 능동적, ...]
- "학교에": [목적지, 교육기관, 장소, ...]

결과: "간다"는 "나는"에서 주체 정보를, "학교에"에서 목적지 정보를 가져와 
      "1인칭이 교육기관으로 이동한다"는 의미를 구성
```

---

## 2. CausalSelfAttention 클래스 완전 분해

### 우리 구현 코드 살펴보기

[레포지토리](https://github.com/BanHun28/gpt2_study)의 `main.py`에서:

```python
class CausalSelfAttention(nn.Module):
    """
    인과관계 자기 주의기제(Causal Self-Attention) 클래스
    이전 단어들만 참고할 수 있고, 미래 단어는 볼 수 없도록 마스킹된 Attention
    """
    def __init__(self, config):
        super().__init__()
        
        # 【핵심】Q, K, V를 한 번에 생성하는 레이어
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 【핵심】Multi-Head 결과를 합치는 레이어  
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 【핵심】미래를 볼 수 없게 하는 마스크
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
```

### 1단계: Q, K, V 생성의 비밀

#### 왜 하나의 레이어로 3개를 만들까?

```python
# 비효율적인 방법 (3개 별도 레이어)
self.query_layer = nn.Linear(config.n_embd, config.n_embd)
self.key_layer = nn.Linear(config.n_embd, config.n_embd)  
self.value_layer = nn.Linear(config.n_embd, config.n_embd)

# 효율적인 방법 (1개 레이어로 3배 출력)
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
```

**장점:**
1. **메모리 효율성**: 3번의 행렬 곱셈 → 1번의 행렬 곱셈
2. **병렬 처리**: GPU에서 더 빠른 연산
3. **캐시 효율성**: 메모리 접근 패턴 최적화

#### Q, K, V 분할 과정

```python
def forward(self, x):
    B, T, C = x.size()  # 배치, 시퀀스 길이, 임베딩 차원
    
    # 1. 하나의 레이어로 3배 크기 출력 생성
    qkv = self.c_attn(x)  # (B, T, C) → (B, T, 3*C)
    
    # 2. 3등분으로 분할
    q, k, v = qkv.split(self.n_embd, dim=2)  # 각각 (B, T, C)
    
    # 3. Multi-Head를 위한 차원 재배열
    # (B, T, C) → (B, T, n_head, C//n_head) → (B, n_head, T, C//n_head)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

### 2단계: Scaled Dot-Product Attention 수식 이해

#### 핵심 수식 분해

$$Attention(Q, K, V) = softmax(QK^T / √d_k)V$$

이 수식을 단계별로 분해해보겠습니다:

```python
# 1단계: Attention Score 계산
att = (q @ k.transpose(-2, -1))  # QK^T: 질문과 키의 유사도

# 2단계: 스케일링  
att = att * (1.0 / math.sqrt(k.size(-1)))  # √d_k로 나누기

# 3단계: 인과관계 마스킹 (미래 차단)
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

# 4단계: 확률로 변환
att = F.softmax(att, dim=-1)  # 각 행의 합이 1이 되도록

# 5단계: Value와 가중합
y = att @ v  # 가중 평균 계산
```

#### 각 단계의 직관적 의미

**1단계: QK^T (질문-키 매칭)**
```
Q: "간다"의 질문 벡터 [0.1, 0.5, -0.2, ...]
K: "나는"의 키 벡터   [0.2, 0.3, -0.1, ...]

내적 = 0.1*0.2 + 0.5*0.3 + (-0.2)*(-0.1) + ... = 0.37

의미: "간다"가 "나는"과 얼마나 관련 있는가? → 0.37
```

**2단계: √d_k 스케일링**
```python
# 왜 √d_k로 나눌까?
d_k = 64  # 각 헤드의 차원

# 스케일링 전: 내적 값이 너무 클 수 있음
raw_score = 37.2  # 큰 값

# 스케일링 후: 적절한 크기로 조정
scaled_score = 37.2 / math.sqrt(64) = 37.2 / 8 = 4.65

# 효과: softmax에서 극단적 확률 방지 (모든 확률이 0 또는 1로 수렴하는 것 방지)
```

**3단계: 인과관계 마스킹**
```python
# 미래를 볼 수 없게 하는 마스크
mask = [[1, 0, 0, 0],    # 첫 번째 단어는 자기만 봄
        [1, 1, 0, 0],    # 두 번째 단어는 첫 번째와 자기만 봄  
        [1, 1, 1, 0],    # 세 번째 단어는 이전 모든 것과 자기만 봄
        [1, 1, 1, 1]]    # 네 번째 단어는 모든 이전 단어 봄

# 0인 위치를 -∞로 설정 → softmax에서 확률 0이 됨
```

**4-5단계: 확률 변환 및 가중합**
```python
# 예시: "간다" 위치에서의 Attention
attention_weights = [0.1, 0.7, 0.2]  # "나는", "학교에", "간다" 각각에 대한 가중치

# Value들의 가중합
result = 0.1 * value_나는 + 0.7 * value_학교에 + 0.2 * value_간다
# → "간다"는 주로 "학교에"의 정보를 참고함!
```

---

## 3. Multi-Head Attention: 여러 시각으로 보기

### 왜 하나의 Attention으로 부족할까?

하나의 Attention은 하나의 관점만 가집니다:

```
단일 Attention이 놓칠 수 있는 것들:

문장: "The bank can guarantee deposits will eventually cover future tuition costs"

Head 1: 문법적 관계만 파악
- "bank" ↔ "guarantee" (주어-동사)
- "deposits" ↔ "cover" (주어-동사)

Head 2: 의미적 연관만 파악  
- "bank" ↔ "deposits" (금융 관련)
- "tuition" ↔ "costs" (교육비 관련)

Head 3: 시간적 관계만 파악
- "eventually" ↔ "future" (시간 순서)
```

### Multi-Head의 구현 원리

```python
# main.py에서
self.n_head = config.n_head  # 보통 12개

# 768차원을 12개 헤드로 나누면
head_dim = 768 // 12 = 64  # 각 헤드는 64차원 담당

# 차원 재배열
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# (B, T, 768) → (B, T, 12, 64) → (B, 12, T, 64)
```

#### 각 헤드가 보는 다른 시각

```python
# 12개 헤드의 서로 다른 역할 (학습으로 자동 분화)
헤드 1: 주어-동사 관계 전문
헤드 2: 수식어-피수식어 관계 전문  
헤드 3: 시간 순서 관계 전문
헤드 4: 인과관계 전문
헤드 5: 동의어/반의어 관계 전문
...
헤드 12: 복합적 의미 관계 전문
```

### 헤드들의 결과 통합

```python
# 각 헤드의 출력을 다시 합치기
y = y.transpose(1, 2).contiguous().view(B, T, C)
# (B, 12, T, 64) → (B, T, 12, 64) → (B, T, 768)

# 최종 프로젝션으로 정보 통합
y = self.c_proj(y)  # 여러 시각을 하나로 종합
```

---

## 4. Causal Masking: 시간의 방향성 지키기

### GPT는 왜 미래를 볼 수 없을까?

**GPT의 목표**: 이전 단어들만 보고 다음 단어 예측

```
잘못된 예 (미래를 본다면):
"나는 ??? 간다" → "???" 예측 시 "간다"를 미리 본다면?
→ 너무 쉬워짐! 학습이 제대로 안 됨

올바른 예 (미래를 차단):
"나는 ???" → "???" 예측 시 "간다"를 볼 수 없음
→ "나는" 정보만으로 다음 단어 추론해야 함
```

### 마스크 구현의 세부사항

```python
# 마스크 생성
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

# torch.tril(): 하삼각 행렬 생성
mask = torch.tril(torch.ones(4, 4))
print(mask)
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],  
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

#### 마스킹 적용 과정

```python
# Attention Score에 마스크 적용
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

# 예시: 4개 단어 문장에서
원본 Attention Score:
[[2.1, 1.5, 0.8, 1.2],   # 첫 번째 단어
 [1.3, 2.5, 1.1, 0.9],   # 두 번째 단어  
 [0.7, 1.8, 2.2, 1.4],   # 세 번째 단어
 [1.0, 1.6, 1.9, 2.3]]   # 네 번째 단어

마스킹 후:
[[2.1, -∞,  -∞,  -∞ ],   # 첫 번째는 자기만 봄
 [1.3, 2.5, -∞,  -∞ ],   # 두 번째는 이전까지만 봄
 [0.7, 1.8, 2.2, -∞ ],   # 세 번째는 이전까지만 봄  
 [1.0, 1.6, 1.9, 2.3]]   # 네 번째는 모든 이전 봄

Softmax 후:
[[1.0, 0.0, 0.0, 0.0],   # -∞는 확률 0이 됨
 [0.2, 0.8, 0.0, 0.0],
 [0.1, 0.3, 0.6, 0.0],
 [0.1, 0.2, 0.3, 0.4]]
```

---

## 5. 최신 최적화: Flash Attention

### PyTorch 2.0+의 혁신

우리 코드에서 볼 수 있는 최신 최적화:

```python
# PyTorch 2.0+에서 제공하는 최적화된 scaled_dot_product_attention 사용
if hasattr(F, 'scaled_dot_product_attention'):
    # 최신 버전의 고성능 Attention 구현 사용
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                      dropout_p=self.dropout if self.training else 0, 
                                      is_causal=True)
else:
    # 이전 PyTorch 버전을 위한 수동 구현
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v
```

### Flash Attention의 혁신

**기존 방식의 문제점:**
```python
# 메모리 사용량: O(n²)
attention_matrix = Q @ K.T  # (seq_len, seq_len) 크기의 행렬 저장
# 1024 토큰 → 1M개 원소 → 4MB (float32 기준)
# 4096 토큰 → 16M개 원소 → 64MB
```

**Flash Attention의 해결책:**
- **메모리 효율성**: 중간 행렬을 저장하지 않고 즉시 계산
- **속도 향상**: GPU 메모리 계층을 효율적으로 사용
- **수학적 동일성**: 결과는 완전히 동일

---

## 6. 실전 분석: Attention 가중치 시각화

### Attention Pattern 이해하기

```python
# Attention 가중치 추출 및 분석
def analyze_attention_patterns():
    model.eval()
    
    # 예시 문장
    text = "ROMEO: But soft, what light through yonder window breaks?"
    tokens = enc.encode(text)
    
    # 중간 결과 추출을 위한 hook 설정
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Attention 가중치 저장
        attention_weights.append(output[1])  # [1]이 attention weights
    
    # 모델 실행
    with torch.no_grad():
        output = model(torch.tensor([tokens]))
    
    # 첫 번째 레이어, 첫 번째 헤드의 패턴 분석
    att_weights = attention_weights[0][0, 0]  # [seq_len, seq_len]
    
    # 시각화용 토큰 텍스트
    token_texts = [enc.decode([token]) for token in tokens]
    
    # 각 토큰이 어떤 토큰들에 주목하는지 출력
    for i, token in enumerate(token_texts):
        top_indices = att_weights[i].topk(3).indices
        top_tokens = [token_texts[j] for j in top_indices]
        print(f"'{token}' 주목 → {top_tokens}")
```

**예상 출력:**
```
'ROMEO' 주목 → ['ROMEO', 'But', 'soft']
'But' 주목 → ['ROMEO', 'But', 'soft']  
'soft' 주목 → ['But', 'soft', 'what']
'what' 주목 → ['soft', 'what', 'light']
'light' 주목 → ['what', 'light', 'through']
...
```

### 흥미로운 Attention 패턴들

**1. 지역적 패턴**: 인접한 단어들에 주목
**2. 구문적 패턴**: 주어-동사, 수식어-피수식어 관계
**3. 의미적 패턴**: 의미적으로 연관된 단어들
**4. 위치적 패턴**: 특정 위치(첫 번째, 마지막 등)에 주목

---

## 7. Attention의 한계와 해결책

### 현재 Attention의 한계

#### 1. 계산 복잡도: O(n²)
```python
# 시퀀스 길이가 2배 → 계산량 4배
seq_len_512 = 512
attention_ops_512 = 512 * 512 = 262,144

seq_len_1024 = 1024  
attention_ops_1024 = 1024 * 1024 = 1,048,576  # 4배!
```

#### 2. 메모리 사용량 폭증
```python
# GPU 메모리 사용량 (float32 기준)
def memory_usage(seq_len, batch_size=1):
    attention_matrix = seq_len * seq_len * 4  # bytes
    total_memory = attention_matrix * batch_size
    return f"{total_memory / (1024**2):.1f} MB"

print(f"1K 토큰: {memory_usage(1024)}")    # 4.0 MB
print(f"4K 토큰: {memory_usage(4096)}")    # 64.0 MB  
print(f"16K 토큰: {memory_usage(16384)}")  # 1024.0 MB = 1GB!
```

### 차세대 해결책들

#### 1. Sparse Attention
- 모든 토큰을 보지 않고 일부만 선택적으로 참조
- Longformer, BigBird 등에서 사용

#### 2. Linear Attention  
- Attention 계산을 선형 복잡도로 근사
- Performer, FNet 등에서 시도

#### 3. Retrieval-Augmented Generation
- 필요한 정보만 외부에서 가져와 사용
- RAG, FiD 등의 방법론

---

## 8. 실습: Attention 직접 구현하기

### 실습 1: 미니 Attention 구현

```python
import torch
import torch.nn.functional as F
import math

def simple_attention(query, key, value, mask=None):
    """
    간단한 Attention 구현
    """
    # Q, K의 내적으로 Attention Score 계산
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 스케일링
    d_k = key.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # 마스킹 (옵션)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax로 확률 변환
    attention_weights = F.softmax(scores, dim=-1)
    
    # Value와 가중합
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# 테스트
seq_len, d_model = 4, 8
q = torch.randn(1, seq_len, d_model)
k = torch.randn(1, seq_len, d_model)  
v = torch.randn(1, seq_len, d_model)

output, weights = simple_attention(q, k, v)
print(f"입력 크기: {q.shape}")
print(f"출력 크기: {output.shape}")
print(f"Attention 가중치: {weights.shape}")
```

### 실습 2: Multi-Head 구현

```python
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Q, K, V 생성
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Multi-Head로 분할
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention 계산
        output, _ = simple_attention(Q, K, V)
        
        # 헤드 합치기
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # 최종 프로젝션
        return self.w_o(output)

# 테스트
mha = SimpleMultiHeadAttention(d_model=128, n_heads=8)
x = torch.randn(2, 10, 128)
output = mha(x)
print(f"Multi-Head Attention 출력: {output.shape}")
```

### 실습 3: Causal Mask 실험

```python
def create_causal_mask(seq_len):
    """하삼각 마스크 생성"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

def visualize_attention_pattern(seq_len=8):
    """Attention 패턴 시각화"""
    # 랜덤 Attention Score 생성
    scores = torch.randn(seq_len, seq_len)
    
    # Causal Mask 적용
    mask = create_causal_mask(seq_len)
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax 적용
    attention_weights = F.softmax(masked_scores, dim=-1)
    
    print("Causal Attention Pattern:")
    print(attention_weights.numpy().round(3))
    
    # 각 위치가 이전 위치들만 참조함을 확인
    for i in range(seq_len):
        future_sum = attention_weights[i, i+1:].sum()
        print(f"위치 {i}: 미래 토큰 가중치 합 = {future_sum:.6f}")

visualize_attention_pattern()
```

---

## 마무리: Attention의 마법을 이해했습니다

### 오늘 배운 핵심 내용

1. **Attention의 직관**: 도서관에서 정보 찾기와 같은 메커니즘
2. **Q, K, V 시스템**: Query(질문), Key(색인), Value(내용)의 역할
3. **Scaled Dot-Product**: 수식의 각 단계별 의미
4. **Multi-Head**: 여러 시각으로 동시에 보는 지혜
5. **Causal Masking**: 미래를 차단하여 올바른 학습 유도

### Attention의 혁신성

```
이전 (RNN): A → B → C → D (순차적, 느림)
이후 (Attention): A ↔ B ↔ C ↔ D (병렬적, 빠름, 직접 연결)
```

**Attention이 가능하게 한 것들:**
- **병렬 처리**: 모든 위치 동시 계산
- **장거리 의존성**: 거리에 상관없이 직접 연결
- **해석 가능성**: 어떤 단어에 주목했는지 확인 가능
- **스케일링**: 더 긴 문장, 더 큰 모델 처리 가능

### 다음 편 예고: Transformer Block의 완성

다음 편에서는 Attention과 함께 **Transformer Block**을 구성하는 나머지 요소들을 배웁니다:

- **Layer Normalization**: 안정적인 학습을 위한 정규화
- **Residual Connection**: 깊은 네트워크의 그래디언트 소실 해결
- **Feed-Forward Network**: Attention이 놓친 패턴 보완
- **Block 전체 구조**: 모든 요소들의 유기적 결합

**미리 생각해볼 질문:**
Attention으로 단어 간 관계는 파악했는데, 각 단어의 "의미 변환"은 누가 담당할까요? 바로 **MLP(Feed-Forward Network)**가 그 역할을 합니다!

### 실습 과제

다음 편까지 해볼 과제:

1. **Attention 시각화**: 실제 문장에서 어떤 단어들이 서로 주목하는지 확인
2. **마스크 실험**: Causal Mask를 제거하면 어떻게 될지 테스트
3. **헤드 수 실험**: n_head를 바꿔가며 성능 변화 관찰

Attention의 마법을 이해했으니, 이제 완전한 Transformer Block으로 넘어갑시다! 🔮
