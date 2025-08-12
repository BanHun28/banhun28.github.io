---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 04
date: 2025-08-12 08:10:22 +0900
categories: [machine learning, GPT]
tags: [machine learning, gpt, transformer, layer normalization, residual connection, mlp, feed forward, attention, deep learning, pytorch, neural networks, gelu]
---

# GPT 완전 정복 4편: Transformer Block 해부학 - 지능의 구조

> **이전 편 요약**: 3편에서는 Attention 메커니즘을 통해 단어들이 어떻게 서로 소통하는지 배웠습니다. 이번 편에서는 Attention과 함께 Transformer Block을 구성하는 나머지 핵심 요소들을 완전히 이해해보겠습니다.

---

## 들어가며: 하나의 Block = 한 단계의 사고

인간이 복잡한 문제를 해결할 때를 생각해보세요:

```
문제: "셰익스피어 스타일로 사랑 시를 쓰세요"

1단계 사고: 단어들 간의 관계 파악 (Attention)
- "사랑"과 "시"의 연관성 이해
- "셰익스피어"와 "스타일"의 관계 파악

2단계 사고: 각 개념의 의미 심화 (Feed-Forward)  
- "사랑" → 열정, 그리움, 아름다움 등의 풍부한 의미
- "셰익스피어" → 운율, 소네트, 엘리자베스 시대 등

3단계 사고: 이전 지식과 통합 (Residual Connection)
- 새로운 아이디어 + 기존 지식 = 더 풍부한 이해
```

**하나의 Transformer Block이 바로 이런 "한 단계의 사고 과정"을 구현합니다.**

---

## 1. Block 클래스 전체 구조 이해

### 우리 구현의 핵심 코드

[레포지토리](https://github.com/BanHun28/gpt2_study)의 `main.py`에서:

```python
class Block(nn.Module):
    """
    Transformer 블록 - GPT의 기본 구성 단위
    하나의 블록은 Self-Attention과 MLP로 구성되며,
    각각 Layer Normalization과 Residual Connection을 가짐
    """
    def __init__(self, config):
        super().__init__()
        
        # 【1단계】 Attention 전 정규화
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        
        # 【2단계】 관계 학습 (3편에서 배운 Attention)
        self.attn = CausalSelfAttention(config)
        
        # 【3단계】 MLP 전 정규화  
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        
        # 【4단계】 의미 변환 (이번 편의 핵심)
        self.mlp = MLP(config)

    def forward(self, x):
        # 【핵심】 두 개의 Residual Connection
        x = x + self.attn(self.ln_1(x))  # 관계 정보 추가
        x = x + self.mlp(self.ln_2(x))   # 변환된 의미 추가
        return x
```

### Block의 정보 흐름 시각화

```
입력 x (단어 임베딩들)
    ↓
┌─────────────────────────────────────┐
│  ln_1(x) → Attention → +x           │  ← 첫 번째 Residual
│  ↓                                  │
│  ln_2(x) → MLP → +x                 │  ← 두 번째 Residual  
└─────────────────────────────────────┘
    ↓
출력 (풍부해진 표현들)
```

이 간단해 보이는 구조가 어떻게 놀라운 지능을 만들어내는지 하나씩 분석해보겠습니다.

---

## 2. Layer Normalization: 학습의 안정성을 위한 마법

### Batch Normalization vs Layer Normalization

#### Batch Normalization의 한계

```python
# Batch Normalization (CNN에서 주로 사용)
# 같은 위치의 뉴런들을 배치 전체에서 정규화

배치 데이터:
문장1: [나는, 학교에, 간다]
문장2: [고양이가, 집에서, 잔다]  
문장3: [비가, 밖에, 온다]

# 첫 번째 위치 단어들 ("나는", "고양이가", "비가")을 함께 정규화
# 문제: 문장마다 길이가 다르고, 의미가 완전히 다름!
```

#### Layer Normalization의 해결책

```python
# Layer Normalization (Transformer에서 사용)
# 각 문장의 모든 차원을 개별적으로 정규화

문장: "나는 학교에 간다"
임베딩: [
    [0.1, 0.5, -0.2, 0.8],  # "나는"
    [0.3, -0.1, 0.9, 0.4],  # "학교에"  
    [-0.2, 0.7, 0.1, -0.3]  # "간다"
]

# 각 단어별로 차원들의 평균과 분산 계산하여 정규화
정규화된 "나는": [(0.1-μ)/σ, (0.5-μ)/σ, (-0.2-μ)/σ, (0.8-μ)/σ]
```

### Layer Normalization 구현 분석

```python
# main.py에서
self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)

# LayerNorm의 수학적 정의
def layer_norm(x, gamma, beta, eps=1e-5):
    # 마지막 차원에 대해 평균과 분산 계산
    mean = x.mean(dim=-1, keepdim=True)  # 각 단어별 평균
    var = x.var(dim=-1, keepdim=True)    # 각 단어별 분산
    
    # 정규화
    normalized = (x - mean) / torch.sqrt(var + eps)
    
    # 학습 가능한 스케일과 시프트 적용
    return gamma * normalized + beta
```

#### eps=1e-5가 필요한 이유

```python
# 분산이 0에 가까울 때 문제 발생
var = 0.0000001  # 매우 작은 분산
sqrt_var = math.sqrt(var)  # 0.0003... 

# 나누기 연산에서 수치 불안정 발생 가능
# eps를 더해서 안전장치 제공
safe_sqrt_var = math.sqrt(var + 1e-5)  # 더 안정적
```

### Pre-LN vs Post-LN 구조

#### Post-LN (원래 Transformer 논문)
```python
# 구식 방법
def post_ln_block(x):
    # Residual 후에 정규화
    x = layer_norm(x + attention(x))
    x = layer_norm(x + mlp(x))
    return x
```

#### Pre-LN (GPT-2, 우리 구현)
```python
# 현재 표준
def pre_ln_block(x):
    # 정규화 후에 Residual
    x = x + attention(layer_norm(x))
    x = x + mlp(layer_norm(x))
    return x
```

#### 왜 Pre-LN이 더 좋을까?

```python
# Pre-LN의 장점:
1. 학습 안정성: 그래디언트가 더 안정적으로 흐름
2. 초기화 민감도 감소: 가중치 초기값에 덜 의존
3. 더 깊은 모델 가능: 수백 개 레이어도 학습 가능
4. 빠른 수렴: 더 적은 에포크로 좋은 성능 달성

# 실험 결과:
Post-LN: 12 레이어까지 안정적
Pre-LN: 48+ 레이어까지 안정적 (GPT-2 XL: 48 레이어)
```

---

## 3. MLP: Attention이 놓친 패턴을 보완하는 역할

### MLP의 설계 철학

```python
class MLP(nn.Module):
    """
    다층 퍼셉트론(Multi-Layer Perceptron) 클래스
    Transformer의 Feed-Forward 네트워크 부분
    입력 차원을 4배로 늘렸다가 다시 원래 차원으로 줄임
    """
    def __init__(self, config):
        super().__init__()
        
        # 【확장】 768 → 3072 (4배)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # 【활성화】 비선형성 추가
        self.gelu = nn.GELU(approximate='tanh')
        
        # 【압축】 3072 → 768 (원래 크기)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
        # 【정규화】 과적합 방지
        self.dropout = nn.Dropout(config.dropout)
```

### 왜 4배로 확장할까?

#### 정보 처리 공간의 확장

```python
# 비유: 작업 공간의 크기

작은 책상 (768차원):
- 기본적인 작업만 가능
- "cat" → "고양이" 정도의 단순 변환

큰 작업실 (3072차원):  
- 복잡한 작업 가능
- "cat" → [애완동물, 포유류, 털복숭이, 야옹, 독립적, ...] 
  풍부한 의미 공간에서 변환

다시 정리된 책상 (768차원):
- 핵심 정보만 추려서 다음 단계로 전달
```

#### 실제 정보 변환 과정

```python
# 예시: "bank" 단어의 의미 변환
input_768 = [0.2, -0.1, 0.5, ...]  # "bank"의 기본 임베딩

# 1단계: 확장 (768 → 3072)
expanded_3072 = c_fc(input_768)
# 이제 다양한 의미를 동시에 표현 가능:
# [은행 관련 뉴런들, 강둑 관련 뉴런들, 저장 관련 뉴런들, ...]

# 2단계: 비선형 활성화 (맥락에 따라 적절한 의미 선택)
activated = gelu(expanded_3072)
# 문맥에 따라 "은행" 또는 "강둑" 의미가 강화됨

# 3단계: 압축 (3072 → 768)
output_768 = c_proj(activated)
# 맥락에 맞는 핵심 의미만 추출하여 표준 크기로 반환
```

### GELU 활성화 함수의 비밀

#### ReLU vs GELU 비교

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 1000)

# ReLU: 0 이하는 완전 차단
relu_output = F.relu(x)

# GELU: 부드러운 곡선, 음수도 약간 통과
gelu_output = F.gelu(x)

# 시각화 비교
print("ReLU: 급격한 변화, 음수 완전 차단")
print("GELU: 부드러운 변화, 음수 일부 보존")
```

#### GELU의 장점들

```python
# 1. 그래디언트 흐름 개선
ReLU의 문제: x < 0일 때 그래디언트 = 0 (Dead ReLU)
GELU의 해결: x < 0일 때도 작은 그래디언트 존재

# 2. 정보 손실 감소  
ReLU: 음수 정보 완전 손실
GELU: 음수 정보도 일부 보존

# 3. 더 풍부한 표현력
ReLU: 단순한 on/off 스위치
GELU: 연속적인 강도 조절
```

#### approximate='tanh'의 의미

```python
# 정확한 GELU (계산 복잡)
def exact_gelu(x):
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))

# Tanh 근사 GELU (계산 빠름)  
def approx_gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

# 성능 차이: 무시할 수 있을 정도
# 속도 차이: 약 2-3배 빠름
```

---

## 4. Residual Connection: 정보 고속도로

### 그래디언트 소실 문제의 해결

#### 깊은 네트워크의 고질적 문제

```python
# 12개 레이어를 거치는 그래디언트

without_residual:
layer_12 ← layer_11 ← ... ← layer_2 ← layer_1
그래디언트: 0.9^12 = 0.28 (72% 소실!)

with_residual:
layer_12 ← (layer_11 + shortcut) ← ... ← (layer_2 + shortcut) ← layer_1  
그래디언트: 최소 1.0은 보장 (소실 방지!)
```

#### Residual Connection의 수학적 의미

```python
# 일반적인 레이어
y = F(x)  # 입력을 완전히 변환

# Residual 레이어  
y = x + F(x)  # 입력 + 변화량

# 역전파 시:
dy/dx = 1 + dF/dx  # 최소 1의 그래디언트 보장!
```

### 우리 구현에서의 Residual Connection

```python
def forward(self, x):
    # 첫 번째 Residual: 관계 정보 추가
    x = x + self.attn(self.ln_1(x))
    #     ↑        ↑
    #   원본    새로운 관계 정보
    
    # 두 번째 Residual: 변환된 의미 추가  
    x = x + self.mlp(self.ln_2(x))
    #     ↑       ↑
    #   기존    새로운 의미 변환
    
    return x
```

### Residual의 정보 누적 효과

```python
# 12개 Block을 거치는 정보의 변화

초기 입력: "cat" 임베딩 = [0.1, 0.2, 0.3, ...]

Block 1 후: 원본 + 기본 관계 + 기본 의미
Block 2 후: 원본 + 기본 관계 + 기본 의미 + 구문 관계 + 구문 의미  
Block 3 후: 원본 + ... + 의미적 관계 + 의미적 변환
...
Block 12 후: 원본 + 모든 층위의 관계와 의미가 누적된 풍부한 표현

# 핵심: 원본 정보는 절대 손실되지 않음!
```

---

## 5. Block 전체의 동작 과정 상세 분석

### 단계별 정보 변환 추적

```python
# 예시 입력: "Romeo loves Juliet"
input_tokens = ["Romeo", "loves", "Juliet"]
```

#### 1단계: 첫 번째 Layer Normalization

```python
# 입력 임베딩 (예시)
x = torch.tensor([
    [0.1, 0.5, -0.2],  # Romeo
    [0.3, -0.1, 0.8],  # loves  
    [-0.2, 0.7, 0.4]   # Juliet
])

# ln_1(x): 각 토큰을 개별적으로 정규화
normalized_x = self.ln_1(x)
# [[0.0, 0.7, -0.7],   # Romeo (정규화됨)
#  [-0.2, -0.9, 1.1],  # loves (정규화됨)
#  [-0.6, 0.8, -0.2]]  # Juliet (정규화됨)
```

#### 2단계: Attention으로 관계 파악

```python
# Attention에서 발견되는 관계들
attention_output = self.attn(normalized_x)

# 예상되는 Attention 패턴:
# Romeo  → Romeo(0.7) + loves(0.2) + Juliet(0.1)  
# loves  → Romeo(0.3) + loves(0.4) + Juliet(0.3)  
# Juliet → Romeo(0.1) + loves(0.2) + Juliet(0.7)

# 결과: 각 단어가 다른 단어들과의 관계 정보를 획득
```

#### 3단계: 첫 번째 Residual Connection

```python
# 원본 + 관계 정보
x = x + attention_output

# 효과: 
# - 원본 의미 보존
# - 새로운 관계 정보 추가
# - "Romeo"가 "loves", "Juliet"과의 관계를 인식
```

#### 4단계: 두 번째 Layer Normalization

```python
# 다시 정규화하여 MLP 입력 준비
normalized_x2 = self.ln_2(x)
```

#### 5단계: MLP로 의미 변환

```python
# MLP에서 일어나는 변환
mlp_output = self.mlp(normalized_x2)

# 예상되는 변환:
# "Romeo" → [남성, 주인공, 열정적, 젊은, ...]
# "loves" → [감정, 동작, 긍정적, 강한, ...]  
# "Juliet" → [여성, 주인공, 아름다운, 젊은, ...]
```

#### 6단계: 두 번째 Residual Connection

```python
# 최종 출력
final_x = x + mlp_output

# 최종 결과:
# 원본 의미 + 관계 정보 + 변환된 의미 = 풍부한 표현
```

---

## 6. 여러 Block의 계층적 학습

### 각 Block이 학습하는 서로 다른 패턴

```python
# GPT-2는 12개 Block 사용 (우리 구현도 동일)
h=nn.ModuleList([Block(config) for _ in range(config.n_layer)])

# 각 Block의 역할 (연구를 통해 관찰된 패턴)
Block 1-2:   기본 구문 분석 (품사, 기본 문법)
Block 3-4:   구 단위 분석 (명사구, 동사구)
Block 5-6:   문장 구조 분석 (주어-동사-목적어)
Block 7-8:   의미 관계 분석 (상위-하위 개념)
Block 9-10:  담화 분석 (문장 간 관계)
Block 11-12: 추상적 추론 (함의, 유추)
```

### 점진적 추상화 과정

```python
# 문장: "The cat sat on the mat"

Block 1 출력: [관사, 명사, 동사, 전치사, 관사, 명사] (품사 인식)
Block 3 출력: [명사구: "The cat"], [동사구: "sat on"], [명사구: "the mat"] 
Block 6 출력: [주어: cat], [동작: sitting], [장소: mat]
Block 9 출력: [상황: 고양이가 매트 위에 앉아있는 평화로운 장면]
Block 12 출력: [추상적 의미: 안정감, 편안함, 일상성 등]
```

---

## 7. 실전 분석: Block 내부 관찰하기

### Hook을 이용한 중간 결과 관찰

```python
def analyze_block_internals():
    model.eval()
    
    # 각 Block의 중간 결과를 저장할 리스트
    block_outputs = []
    attention_patterns = []
    mlp_outputs = []
    
    def register_hooks():
        def hook_fn(name):
            def hook(module, input, output):
                if 'attn' in name:
                    attention_patterns.append(output.detach())
                elif 'mlp' in name:
                    mlp_outputs.append(output.detach())
                else:
                    block_outputs.append(output.detach())
            return hook
        
        # 각 Block과 서브모듈에 hook 등록
        for i, block in enumerate(model.transformer.h):
            block.register_forward_hook(hook_fn(f'block_{i}'))
            block.attn.register_forward_hook(hook_fn(f'attn_{i}'))
            block.mlp.register_forward_hook(hook_fn(f'mlp_{i}'))
    
    register_hooks()
    
    # 테스트 문장 실행
    test_text = "Romeo loves Juliet deeply"
    tokens = enc.encode(test_text)
    
    with torch.no_grad():
        output = model(torch.tensor([tokens]))
    
    # 결과 분석
    print(f"총 {len(block_outputs)}개 Block의 출력 수집")
    print(f"Block별 출력 변화:")
    
    for i, block_out in enumerate(block_outputs[:3]):  # 처음 3개 Block만
        # 각 Block 출력의 통계적 특성 분석
        mean_activation = block_out.mean().item()
        std_activation = block_out.std().item()
        print(f"Block {i}: 평균={mean_activation:.3f}, 표준편차={std_activation:.3f}")

analyze_block_internals()
```

### Layer Normalization 효과 측정

```python
def measure_layer_norm_effect():
    """Layer Norm 전후의 분포 변화 측정"""
    
    # 임의의 입력 생성 (다양한 스케일)
    x = torch.randn(1, 10, 768) * 5  # 큰 분산
    
    print("Layer Norm 전:")
    print(f"평균: {x.mean(-1)}")
    print(f"표준편차: {x.std(-1)}")
    
    # Layer Norm 적용
    ln = nn.LayerNorm(768)
    normalized = ln(x)
    
    print("\nLayer Norm 후:")
    print(f"평균: {normalized.mean(-1)}")
    print(f"표준편차: {normalized.std(-1)}")
    
    # 결과: 평균 ≈ 0, 표준편차 ≈ 1로 정규화됨

measure_layer_norm_effect()
```

---

## 8. 실습: 미니 Transformer Block 구현

### 실습 1: 간단한 Block 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = SimpleMLP(d_model)
        
    def forward(self, x):
        # 첫 번째 Residual Connection
        attn_out, _ = self.attn(x, x, x, is_causal=True)
        x = x + attn_out
        x = self.ln1(x)
        
        # 두 번째 Residual Connection
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)
        
        return x

# 테스트
block = SimpleBlock(d_model=128, n_heads=8)
x = torch.randn(2, 10, 128)  # (batch, seq_len, d_model)
output = block(x)
print(f"입력: {x.shape}, 출력: {output.shape}")
```

### 실습 2: Residual Connection 효과 실험

```python
def compare_with_without_residual():
    """Residual Connection 유무에 따른 차이 실험"""
    
    class BlockWithoutResidual(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, d_model)
            self.linear2 = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            # Residual 없음 - 정보 손실 가능
            x = self.linear1(self.ln1(x))
            x = self.linear2(self.ln2(x))
            return x
    
    class BlockWithResidual(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, d_model)
            self.linear2 = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            # Residual 있음 - 정보 보존
            x = x + self.linear1(self.ln1(x))
            x = x + self.linear2(self.ln2(x))
            return x
    
    # 동일한 입력으로 실험
    x = torch.randn(1, 5, 64)
    
    # 여러 레이어 쌓기
    blocks_without = nn.Sequential(*[BlockWithoutResidual(64) for _ in range(10)])
    blocks_with = nn.Sequential(*[BlockWithResidual(64) for _ in range(10)])
    
    # 결과 비교
    output_without = blocks_without(x)
    output_with = blocks_with(x)
    
    print("10개 레이어 통과 후:")
    print(f"Residual 없음 - 분산: {output_without.var():.6f}")
    print(f"Residual 있음 - 분산: {output_with.var():.6f}")
    
    # 입력과 출력의 유사도 측정
    similarity_without = F.cosine_similarity(x.flatten(), output_without.flatten(), dim=0)
    similarity_with = F.cosine_similarity(x.flatten(), output_with.flatten(), dim=0)
    
    print(f"입력과의 유사도 (Residual 없음): {similarity_without:.3f}")
    print(f"입력과의 유사도 (Residual 있음): {similarity_with:.3f}")

compare_with_without_residual()
```

### 실습 3: Layer Norm 위치 실험

```python
def compare_pre_post_ln():
    """Pre-LN vs Post-LN 비교"""
    
    class PostLNBlock(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.linear = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            # Post-LN: Residual 후 정규화
            x = self.ln1(x + self.linear(x))
            return x
    
    class PreLNBlock(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.linear = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            # Pre-LN: 정규화 후 Residual
            x = x + self.linear(self.ln1(x))
            return x
    
    # 깊은 네트워크로 실험
    x = torch.randn(1, 5, 64)
    
    post_ln_model = nn.Sequential(*[PostLNBlock(64) for _ in range(20)])
    pre_ln_model = nn.Sequential(*[PreLNBlock(64) for _ in range(20)])
    
    # 그래디언트 흐름 확인을 위한 더미 손실
    post_output = post_ln_model(x)
    pre_output = pre_ln_model(x)
    
    post_loss = post_output.sum()
    pre_loss = pre_output.sum()
    
    post_loss.backward()
    pre_loss.backward()
    
    # 첫 번째 레이어의 그래디언트 크기 비교
    post_grad_norm = post_ln_model[0].linear.weight.grad.norm()
    pre_grad_norm = pre_ln_model[0].linear.weight.grad.norm()
    
    print(f"20개 레이어 후 첫 번째 레이어 그래디언트 크기:")
    print(f"Post-LN: {post_grad_norm:.6f}")
    print(f"Pre-LN: {pre_grad_norm:.6f}")

compare_pre_post_ln()
```

---

## 마무리: 지능의 구조를 이해했습니다

### 오늘 배운 핵심 내용

1. **Layer Normalization**: 각 토큰의 차원별 정규화로 학습 안정성 확보
2. **MLP**: 4배 확장-압축 구조로 풍부한 의미 변환 수행
3. **Residual Connection**: 정보 보존과 그래디언트 흐름 개선
4. **Pre-LN 구조**: Post-LN보다 안정적인 깊은 네트워크 학습
5. **계층적 학습**: 각 Block이 서로 다른 추상화 수준 담당

### Block의 정보 처리 요약

```
하나의 Block = 한 단계의 지능적 사고

입력 → 정규화 → Attention (관계 파악) → Residual (정보 보존)
     ↓
     정규화 → MLP (의미 변환) → Residual (정보 통합) → 출력

결과: 원본 + 관계 정보 + 변환된 의미 = 더 풍부한 표현
```

### 12개 Block의 협력

```
Block 1: 기본 구문 분석
Block 2: 구 단위 분석  
Block 3: 문장 구조 분석
...
Block 12: 추상적 추론

각 단계마다 이전 정보는 보존하면서 새로운 정보를 추가
→ 점진적으로 깊어지는 이해
```

### 다음 편 예고: 전체 GPT 모델의 완성

다음 편에서는 지금까지 배운 모든 구성요소들이 어떻게 **하나의 완전한 GPT 모델**로 통합되는지 배웁니다:

- **Token + Position Embedding**: 입력 준비의 완성
- **12개 Block의 순차 처리**: 점진적 이해 과정
- **Language Model Head**: 확률 예측으로의 변환
- **Weight Sharing**: 입출력 임베딩 공유의 이유
- **전체 모델 구조**: 퍼즐의 최종 완성

**미리 생각해볼 질문:**
지금까지 배운 토큰화, 임베딩, Attention, Block들이 어떻게 연결되어 "셰익스피어 스타일 텍스트"를 생성할 수 있을까요?

### 실습 과제

다음 편까지 해볼 과제:

1. **Block 내부 관찰**: Hook을 사용해 각 Block의 출력 변화 확인
2. **Residual 효과 실험**: Residual Connection 유무에 따른 학습 차이 관찰  
3. **Layer Norm 위치 비교**: Pre-LN vs Post-LN 성능 차이 테스트

이제 모든 구성요소를 이해했으니, 완전한 GPT 모델의 전체 그림을 그려봅시다! 🏗️
