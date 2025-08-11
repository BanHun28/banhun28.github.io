---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 05
date: 2025-08-12 08:10:24 +0900
categories: [machine learning, GPT]
tags: [machine learning, GPT, Transformer]       # TAG names should always be lowercase
---

# GPT 완전 정복 5편: GPT 모델 전체 구조 - 퍼즐의 완성

> **이전 편 요약**: 4편에서는 Transformer Block의 내부 구조를 완전히 이해했습니다. 이제 모든 퍼즐 조각을 맞춰서 완전한 GPT 모델의 전체 그림을 그려보겠습니다.

---

## 들어가며: 교향악단의 완성

지금까지 우리는 개별 악기들을 배웠습니다:

```
1편: GPT의 철학 (지휘자의 비전)
2편: 토큰화 (악보를 읽는 방법)  
3편: Attention (악기들 간의 조화)
4편: Transformer Block (각 악기의 연주법)
```

**이제 모든 악기가 함께 연주하는 완전한 교향곡을 감상할 시간입니다.**

하나의 GPT 모델이 어떻게 "ROMEO:"라는 입력으로부터 셰익스피어 스타일의 완전한 대사를 만들어내는지, 그 전체 과정을 처음부터 끝까지 추적해보겠습니다.

---

## 1. GPT 클래스 전체 구조: 마스터 아키텍트

### 우리 구현의 핵심 코드

[레포지토리](https://github.com/BanHun28/gpt2_study)의 `main.py`에서:

```python
class GPT(nn.Module):
    """
    GPT(Generative Pre-trained Transformer) 메인 모델 클래스
    전체 모델을 통합하여 텍스트 생성 기능을 제공
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # 설정 저장 (generate 등에서 필요)
        
        # 【ModuleDict를 사용하는 이유】
        # - 구성요소들을 체계적으로 관리
        # - state_dict 저장/로드 시 자동 처리  
        # - 네임스페이스 구분으로 디버깅 용이
        self.transformer = nn.ModuleDict(dict(
            
            # 【입력층】 토큰을 벡터로 변환
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            
            # 【위치층】 순서 정보 추가
            wpe=nn.Embedding(config.block_size, config.n_embd),
            
            # 【처리층】 12개의 Transformer Block
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            # 【정규화층】 최종 출력 안정화
            ln_f=nn.LayerNorm(config.n_embd, eps=1e-5),
        ))
        
        # 【출력층】 벡터를 확률로 변환
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 【핵심 혁신】 입출력 가중치 공유
        self.transformer.wte.weight = self.lm_head.weight
        
        # 【초기화】 학습 시작점 설정
        self.apply(self._init_weights)
```

### 전체 아키텍처 시각화

```
                    GPT 모델 전체 구조

입력: "ROMEO:" → [15496, 11] (토큰 ID)
                        ↓
        ┌─────────────────────────────────────────┐
        │              Token Embedding            │
        │   [15496] → [0.1, 0.5, -0.2, ...]     │
        │   [11] → [0.3, -0.1, 0.8, ...]        │
        └─────────────────────────────────────────┘
                        ↓
        ┌─────────────────────────────────────────┐
        │            Position Embedding           │
        │   위치0 → [0.1, 0.0, 0.2, ...]        │
        │   위치1 → [0.0, 0.1, -0.1, ...]       │
        └─────────────────────────────────────────┘
                        ↓ (더하기)
        ┌─────────────────────────────────────────┐
        │              Block 1                    │
        │     ln → attention → +x                 │
        │     ln → mlp → +x                       │
        └─────────────────────────────────────────┘
                        ↓
        ┌─────────────────────────────────────────┐
        │              Block 2                    │
        │     ...동일한 구조...                   │
        └─────────────────────────────────────────┘
                        ↓
                     ... × 12 ...
                        ↓
        ┌─────────────────────────────────────────┐
        │            Final LayerNorm              │
        │        출력 벡터 정규화                  │
        └─────────────────────────────────────────┘
                        ↓
        ┌─────────────────────────────────────────┐
        │            Language Model Head          │
        │   벡터 → 50257개 단어별 확률             │
        └─────────────────────────────────────────┘
                        ↓
출력: [0.001, 0.234, 0.156, ...] → "But" (가장 높은 확률)
```

---

## 2. 데이터 플로우: 한 토큰의 여행

### "ROMEO:"에서 "But"까지의 완전한 여정

#### 단계 0: 토큰화 (2편에서 배운 내용)

```python
# 입력 텍스트
text = "ROMEO:"

# tiktoken으로 토큰화
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)  # [15496, 11]

print(f"'{text}' → {tokens}")
# 'ROMEO:' → [15496, 11]
```

#### 단계 1: 임베딩 변환

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()  # batch_size=1, seq_len=2
    
    # 위치 인덱스 생성
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # [0, 1]
    
    # 토큰 임베딩: ID → 768차원 벡터
    tok_emb = self.transformer.wte(idx)  # (1, 2, 768)
    # [15496] → [0.12, -0.45, 0.33, ..., 0.67]  # 768개 숫자
    # [11]    → [0.89, 0.23, -0.11, ..., -0.34] # 768개 숫자
    
    # 위치 임베딩: 위치 → 768차원 벡터  
    pos_emb = self.transformer.wpe(pos)  # (2, 768)
    # 위치0 → [0.05, 0.12, -0.03, ..., 0.21]
    # 위치1 → [-0.08, 0.07, 0.15, ..., -0.09]
    
    # 두 임베딩을 더해서 최종 입력 생성
    x = tok_emb + pos_emb  # (1, 2, 768)
```

#### 단계 2-13: 12개 Block 순차 처리

```python
# 12개 Transformer Block을 순차적으로 통과
for block in self.transformer.h:
    x = block(x)  # 각 Block에서 관계 파악 + 의미 변환
    
# Block별 점진적 변화 (개념적 표현)
x_after_block_1  = x + attention_relations_basic + mlp_transform_basic
x_after_block_2  = x + attention_relations_syntax + mlp_transform_syntax  
x_after_block_3  = x + attention_relations_semantic + mlp_transform_semantic
# ...
x_after_block_12 = x + attention_relations_abstract + mlp_transform_abstract
```

#### 단계 14: 최종 정규화

```python
# 12개 Block을 거친 후 최종 Layer Normalization
x = self.transformer.ln_f(x)  # (1, 2, 768)

# 효과: 출력층 입력 안정화
# - 12개 Block을 거치면서 값의 크기가 변했을 수 있음
# - softmax 전에 적절한 스케일로 조정
```

#### 단계 15: 확률 예측

```python
if targets is not None:
    # 훈련 모드: 모든 위치에서 예측
    logits = self.lm_head(x)  # (1, 2, 50257)
else:
    # 추론 모드: 마지막 위치만 예측 (메모리 효율성)
    logits = self.lm_head(x[:, [-1], :])  # (1, 1, 50257)

# logits: 각 단어에 대한 "점수" (확률 아님)
# 예시: [1.2, 3.4, 0.8, 2.1, ...] (50257개)
```

#### 단계 16: 다음 단어 선택

```python
# logits를 확률로 변환
probs = F.softmax(logits, dim=-1)
# [0.001, 0.234, 0.003, 0.156, ...] (합이 1)

# 가장 높은 확률의 토큰 선택
next_token = torch.argmax(probs, dim=-1)  # 예: 284 ("But")

# 또는 확률적 샘플링
next_token = torch.multinomial(probs, num_samples=1)
```

---

## 3. Weight Sharing: 입출력 가중치 공유의 혁신

### 왜 가중치를 공유할까?

```python
# 【핵심 혁신】 입출력 가중치 공유
self.transformer.wte.weight = self.lm_head.weight
```

#### 1. 메모리 효율성

```python
# 가중치 공유 없이
wte_params = 50257 * 768 = 38,597,376  # 약 38M 파라미터
lm_head_params = 768 * 50257 = 38,597,376  # 약 38M 파라미터
total = 77,194,752  # 약 77M 파라미터

# 가중치 공유 후
shared_params = 50257 * 768 = 38,597,376  # 약 38M 파라미터만!
절약된_메모리 = 38M * 4바이트 = 152MB  # float32 기준
```

#### 2. 의미적 일관성

```python
# 입력 임베딩: 토큰 ID → 의미 벡터
"cat" (ID: 7163) → [0.2, -0.5, 1.3, 0.8, ...]

# 출력 예측: 의미 벡터와 각 토큰의 유사도
output_vector = [0.1, -0.3, 1.1, 0.9, ...]

# 가중치 공유로 "cat" 토큰과의 내적 계산
similarity = output_vector @ cat_embedding  # 높은 값 → "cat" 선택 확률 높음

# 직관: 출력 벡터가 "cat"의 의미에 가까우면 "cat"을 예측
```

#### 3. 학습 효율성

```python
# 입출력 모두에서 같은 임베딩이 업데이트됨
# → 같은 단어에 대해 두 번의 학습 신호

입력에서의 학습: "cat"을 보고 의미 벡터 학습
출력에서의 학습: "cat"을 예측하며 의미 벡터 정제

결과: 더 빠르고 일관된 학습
```

### Weight Sharing 구현 세부사항

```python
def __init__(self, config):
    # ... 다른 초기화 ...
    
    # 임베딩과 출력층 생성
    self.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    # 【핵심】 같은 가중치를 참조하도록 설정
    self.transformer.wte.weight = self.lm_head.weight
    
    # 이제 둘은 완전히 같은 메모리를 공유
    # wte.weight를 업데이트 → lm_head.weight도 자동 업데이트
    # lm_head.weight를 업데이트 → wte.weight도 자동 업데이트
```

---

## 4. 모델 크기와 성능: 스케일링 법칙

### GPT 모델 크기 비교

```python
# 우리 구현 (GPT-2 Small)
our_config = GPTConfig(
    n_layer=12,    # 12개 Block  
    n_head=12,     # 12개 Attention Head
    n_embd=768,    # 768차원 임베딩
)

# 파라미터 수 계산
def count_parameters(config):
    # Token + Position Embedding
    embedding_params = config.vocab_size * config.n_embd + config.block_size * config.n_embd
    
    # 각 Block의 파라미터
    # Attention: 3 * n_embd^2 (Q,K,V) + n_embd^2 (projection)
    # MLP: n_embd * 4*n_embd + 4*n_embd * n_embd  
    # LayerNorm: 2 * n_embd (각 Block에 2개)
    block_params = (4 * config.n_embd**2 + 8 * config.n_embd**2 + 2 * config.n_embd) * config.n_layer
    
    # Final LayerNorm
    final_ln_params = config.n_embd
    
    total = embedding_params + block_params + final_ln_params
    return total

print(f"우리 모델: {count_parameters(our_config)/1e6:.1f}M 파라미터")
```

#### 공식 GPT 모델 크기들

```python
# GPT-2 모델 계열
configs = {
    "GPT-2 Small":  {"n_layer": 12, "n_head": 12, "n_embd": 768},   # 117M
    "GPT-2 Medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 345M  
    "GPT-2 Large":  {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M
    "GPT-2 XL":     {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1.5B
}

# GPT-3 계열 (추정치)
gpt3_configs = {
    "GPT-3 Small":    {"n_layer": 12, "n_head": 12, "n_embd": 768},    # 125M
    "GPT-3 Medium":   {"n_layer": 24, "n_head": 16, "n_embd": 1024},   # 350M
    "GPT-3 Large":    {"n_layer": 24, "n_head": 16, "n_embd": 1536},   # 760M  
    "GPT-3 XL":       {"n_layer": 24, "n_head": 24, "n_embd": 2048},   # 1.3B
    "GPT-3":          {"n_layer": 96, "n_head": 96, "n_embd": 12288},  # 175B
}
```

### 스케일링 법칙의 발견

#### 1. 더 큰 모델 = 더 좋은 성능

```python
# 연구 결과: 파라미터 수가 10배 증가하면
# 손실(perplexity)이 일정 비율로 감소

모델 크기     | 손실  | 성능
117M (Small)  | 3.2   | 기본
345M (Medium) | 2.8   | 향상  
774M (Large)  | 2.5   | 더 향상
1.5B (XL)     | 2.3   | 훨씬 향상
175B (GPT-3)  | 1.8   | 놀라운 성능
```

#### 2. 창발적 능력 (Emergent Abilities)

```python
# 모델이 커지면서 예상치 못한 능력들이 나타남

Small 모델 (117M):
- 기본적인 텍스트 생성
- 단순한 패턴 학습

Large 모델 (774M):  
- 더 자연스러운 텍스트
- 기본적인 추론

XL 모델 (1.5B):
- 문맥 유지 능력 향상
- 간단한 질의응답

GPT-3 (175B):
- 복잡한 추론
- 코딩 능력  
- 수학 문제 해결
- 창작 능력
```

---

## 5. 전체 모델의 학습과 추론

### 학습 모드 vs 추론 모드

#### 학습 모드: 병렬 처리

```python
# 훈련 중: 모든 위치에서 동시에 예측
input_sequence = ["나는", "학교에", "간다"]
target_sequence = ["학교에", "간다", "<END>"]

# 모든 위치의 예측을 한 번에 계산
logits = model(input_tokens)  # (batch, seq_len, vocab_size)

# 각 위치에서의 손실 계산
position_0: "나는" 다음에 "학교에" 예측했는가?
position_1: "나는 학교에" 다음에 "간다" 예측했는가?  
position_2: "나는 학교에 간다" 다음에 "<END>" 예측했는가?

# 모든 위치의 손실을 평균하여 학습
total_loss = mean(all_position_losses)
```

#### 추론 모드: 순차적 생성

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """텍스트 생성 메서드"""
    for _ in range(max_new_tokens):
        # 1. 컨텍스트 길이 제한
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        # 2. 다음 토큰 예측
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # 마지막 위치만 사용
        
        # 3. Top-k 필터링
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 4. 확률적 샘플링
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 5. 기존 시퀀스에 추가
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

### 생성 과정의 단계별 추적

```python
# 예시: "ROMEO:" → "ROMEO: But soft, what light"

step_0: "ROMEO:" [15496, 11]
        ↓ model forward
        예측: "But" (확률 23.4%)
        
step_1: "ROMEO: But" [15496, 11, 284]  
        ↓ model forward
        예측: "soft" (확률 18.7%)
        
step_2: "ROMEO: But soft" [15496, 11, 284, 2705]
        ↓ model forward  
        예측: "," (확률 31.2%)

step_3: "ROMEO: But soft," [15496, 11, 284, 2705, 11]
        ↓ model forward
        예측: "what" (확률 27.8%)

# 이런 식으로 계속 진행...
```

---

## 6. 모델 성능 최적화 기법들

### 1. Mixed Precision Training

```python
# main.py에서 사용되는 최적화
scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

# 훈련 루프에서
with torch.cuda.amp.autocast(enabled=(device=='cuda')):
    logits, loss = model(xb, yb)  # 순전파는 float16으로

scaler.scale(loss).backward()  # 역전파는 스케일링하여 안정성 확보
```

#### 메모리와 속도 향상

```python
# Float32 vs Float16 비교
memory_float32 = model_params * 4  # bytes
memory_float16 = model_params * 2  # bytes  
memory_saving = 50%  # 메모리 절약

speed_improvement = 1.5~2.0배  # GPU 종류에 따라
```

### 2. Gradient Clipping

```python
# 그래디언트 폭발 방지
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# 효과: 그래디언트 노름이 grad_clip을 넘으면 스케일 다운
# 안정적인 학습 보장
```

### 3. Learning Rate Scheduling

```python
# Warm-up + Decay 스케줄링
if iter_num < 100:  # 처음 100 이터에서는 워밍업
    lr = learning_rate * iter_num / 100
else:
    lr = learning_rate  # 이후 고정 (또는 decay 적용 가능)
```

---

## 7. 실전 분석: 전체 모델 성능 측정

### 모델 크기별 성능 비교

```python
def benchmark_model_sizes():
    """다양한 모델 크기의 성능 벤치마크"""
    
    configs = [
        {"name": "Tiny", "n_layer": 4, "n_head": 4, "n_embd": 128},
        {"name": "Small", "n_layer": 6, "n_head": 6, "n_embd": 384},  # 우리 설정
        {"name": "Medium", "n_layer": 8, "n_head": 8, "n_embd": 512},
    ]
    
    results = []
    
    for config_dict in configs:
        config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "name"})
        model = GPT(config)
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        
        # 메모리 사용량 측정
        model_size_mb = total_params * 4 / (1024**2)  # float32 기준
        
        # 추론 속도 측정
        test_input = torch.randint(0, 1000, (1, 100))
        
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        inference_time = (time.time() - start_time) / 10
        
        results.append({
            "name": config_dict["name"],
            "params": f"{total_params/1e6:.1f}M",
            "size_mb": f"{model_size_mb:.1f}MB",
            "inference_ms": f"{inference_time*1000:.1f}ms"
        })
    
    # 결과 출력
    print("모델 크기별 성능 비교:")
    for result in results:
        print(f"{result['name']:6} | {result['params']:6} | {result['size_mb']:8} | {result['inference_ms']:8}")

benchmark_model_sizes()
```

### 생성 품질 평가

```python
def evaluate_generation_quality():
    """생성된 텍스트의 품질 평가"""
    
    model.eval()
    start_text = "ROMEO:"
    
    # 다양한 생성 설정으로 테스트
    settings = [
        {"temperature": 0.8, "top_k": 40, "name": "Creative"},
        {"temperature": 0.5, "top_k": 20, "name": "Balanced"},  
        {"temperature": 0.2, "top_k": 10, "name": "Conservative"},
    ]
    
    for setting in settings:
        print(f"\n=== {setting['name']} 설정 ===")
        
        tokens = enc.encode(start_text)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            generated = model.generate(
                input_ids, 
                max_new_tokens=50,
                temperature=setting['temperature'],
                top_k=setting['top_k']
            )
        
        generated_text = enc.decode(generated[0].tolist())
        print(f"생성 결과: {generated_text}")
        
        # 간단한 품질 지표들
        lines = generated_text.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        unique_words = len(set(generated_text.lower().split()))
        
        print(f"평균 줄 길이: {avg_line_length:.1f}")
        print(f"고유 단어 수: {unique_words}")

evaluate_generation_quality()
```

---

## 8. 실습: 완전한 GPT 모델 커스터마이징

### 실습 1: 설정 변경 실험

```python
# 다양한 설정으로 모델 실험
experimental_configs = [
    # 원본 설정
    {"name": "Original", "n_layer": 6, "n_head": 6, "n_embd": 384},
    
    # 더 깊고 좁은 모델
    {"name": "Deep_Narrow", "n_layer": 12, "n_head": 4, "n_embd": 256},
    
    # 더 얕고 넓은 모델  
    {"name": "Shallow_Wide", "n_layer": 3, "n_head": 8, "n_embd": 512},
    
    # 헤드 수 변경
    {"name": "More_Heads", "n_layer": 6, "n_head": 12, "n_embd": 384},
]

def compare_configurations():
    for config_dict in experimental_configs:
        print(f"\n=== {config_dict['name']} 모델 ===")
        
        # 설정 적용
        config = GPTConfig(**{k: v for k, v in config_dict.items() if k != "name"})
        model = GPT(config)
        
        # 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters())
        print(f"파라미터 수: {total_params/1e6:.2f}M")
        
        # 메모리 사용량
        memory_mb = total_params * 4 / (1024**2)
        print(f"메모리 사용량: {memory_mb:.1f}MB")
        
        # 텍스트 생성 테스트
        test_input = torch.tensor([[15496, 11]])  # "ROMEO:"
        with torch.no_grad():
            output = model.generate(test_input, max_new_tokens=10)
        
        generated_text = enc.decode(output[0].tolist())
        print(f"생성 예시: {generated_text[:50]}...")

compare_configurations()
```

### 실습 2: 커스텀 생성 함수

```python
def advanced_generate(model, prompt, max_length=100, **kwargs):
    """고급 텍스트 생성 함수"""
    
    # 기본 설정
    settings = {
        'temperature': 0.8,
        'top_k': 40,
        'top_p': 0.9,  # nucleus sampling
        'repetition_penalty': 1.1,
    }
    settings.update(kwargs)
    
    model.eval()
    tokens = enc.encode(prompt)
    input_ids = torch.tensor([tokens])
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # 현재 시퀀스로 예측
            current_input = torch.tensor([generated_tokens[-model.config.block_size:]])
            logits, _ = model(current_input)
            logits = logits[0, -1, :]  # 마지막 위치
            
            # Repetition penalty 적용
            for token in set(generated_tokens):
                logits[token] /= settings['repetition_penalty']
            
            # Temperature 적용
            logits = logits / settings['temperature']
            
            # Top-k 필터링
            if settings['top_k'] > 0:
                indices_to_remove = logits < torch.topk(logits, settings['top_k'])[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Top-p 필터링 (nucleus sampling)
            if settings['top_p'] < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > settings['top_p']
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('Inf')
            
            # 샘플링
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token)
            
            # 종료 조건 확인
            if next_token == enc.encode('\n')[0]:  # 줄바꿈으로 종료
                break
    
    return enc.decode(generated_tokens)

# 사용 예시
result = advanced_generate(
    model, 
    "ROMEO:", 
    max_length=50,
    temperature=0.7,
    top_k=30,
    top_p=0.8,
    repetition_penalty=1.2
)
print("고급 생성 결과:", result)
```

### 실습 3: 모델 분석 도구

```python
def analyze_model_behavior():
    """모델의 내부 동작 분석"""
    
    # 1. Attention 패턴 시각화
    def get_attention_patterns(text):
        tokens = enc.encode(text)
        input_ids = torch.tensor([tokens])
        
        # Hook으로 attention weights 수집
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'attn_weights'):
                attention_weights.append(output.attn_weights.detach())
        
        # 모든 attention 레이어에 hook 등록
        hooks = []
        for block in model.transformer.h:
            hook = block.attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = model(input_ids)
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        return attention_weights, [enc.decode([token]) for token in tokens]
    
    # 2. 임베딩 유사도 분석
    def embedding_similarity(word1, word2):
        token1 = enc.encode(word1)[0]
        token2 = enc.encode(word2)[0]
        
        emb1 = model.transformer.wte.weight[token1]
        emb2 = model.transformer.wte.weight[token2]
        
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()
    
    # 테스트
    print("=== 임베딩 유사도 분석 ===")
    word_pairs = [("king", "queen"), ("love", "hate"), ("Romeo", "Juliet")]
    
    for word1, word2 in word_pairs:
        sim = embedding_similarity(word1, word2)
        print(f"{word1} - {word2}: {sim:.3f}")
    
    # 3. 레이어별 표현 변화
    def layer_representations(text):
        tokens = enc.encode(text)
        input_ids = torch.tensor([tokens])
        
        representations = []
        
        def hook_fn(module, input, output):
            representations.append(output.detach())
        
        # 각 레이어 출력에 hook
        hooks = []
        for i, block in enumerate(model.transformer.h):
            hook = block.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for hook in hooks:
            hook.remove()
        
        return representations
    
    print("\n=== 레이어별 표현 변화 ===")
    reps = layer_representations("Romeo loves Juliet")
    
    for i, rep in enumerate(reps[:3]):  # 처음 3개 레이어만
        mean_norm = rep.norm(dim=-1).mean().item()
        print(f"레이어 {i+1}: 평균 노름 = {mean_norm:.3f}")

analyze_model_behavior()
```

---

## 마무리: 퍼즐의 완성

### 전체 여정 요약

지금까지 우리는 완전한 GPT 모델의 모든 구성요소를 마스터했습니다:

```
1편: GPT의 혁신적 아이디어 → "다음 단어 맞히기"의 철학
2편: 토큰화와 임베딩 → 텍스트를 숫자로 변환하는 기술  
3편: Attention 메커니즘 → 단어들이 소통하는 방법
4편: Transformer Block → 지능의 기본 구조
5편: 전체 모델 통합 → 모든 퍼즐 조각의 완성
```

### GPT의 전체 동작 원리

```
입력 "ROMEO:" 
    ↓ 토큰화
[15496, 11]
    ↓ 임베딩 (단어 + 위치)
768차원 벡터들
    ↓ 12개 Transformer Block 
점진적 이해 심화 (문법 → 의미 → 추론)
    ↓ 출력층
50257개 단어별 확률
    ↓ 샘플링
"But" (다음 단어)
    ↓ 반복
"ROMEO: But soft, what light through yonder window breaks?"
```

### 스케일링의 마법

```
작은 모델 (우리 구현): 기본적인 패턴 학습
중간 모델: 더 자연스러운 텍스트  
큰 모델 (GPT-3): 추론, 코딩, 창작까지

핵심: 같은 구조, 다른 크기 → 질적으로 다른 능력
```

### 다음 편 예고: 학습의 과학

다음 편에서는 이 모든 구조가 어떻게 **학습을 통해 지능을 획득**하는지 배웁니다:

- **Cross-Entropy Loss**: 예측 오차를 측정하는 방법
- **Backpropagation**: 오차를 역산하여 가중치 업데이트  
- **AdamW Optimizer**: 효율적인 학습 알고리즘
- **Learning Rate Scheduling**: 최적의 학습 속도 조절
- **정규화 기법들**: 과적합 방지와 일반화

**미리 생각해볼 질문:**
"ROMEO:"에서 "But"을 잘못 예측했을 때, 모델은 어떻게 자신의 실수를 깨닫고 다음에는 더 나은 예측을 할 수 있을까요?

### 실습 과제

다음 편까지 해볼 과제:

1. **모델 크기 실험**: 다양한 설정으로 모델 크기와 성능 관계 확인
2. **생성 품질 비교**: temperature, top_k 등 설정 변경 효과 관찰
3. **전체 코드 실행**: `python main.py`로 완전한 학습-생성 과정 체험

이제 GPT의 전체 구조를 완전히 이해했으니, 어떻게 이 모든 것이 "학습"을 통해 지능을 얻는지 알아봅시다! 🧠

---

**이전 편**: [4편: Transformer Block 해부학 - 지능의 구조](https://github.com/BanHun28/gpt2_study)  
**다음 편**: [6편: 학습의 과학 - 모델이 "배우는" 과정](https://github.com/BanHun28/gpt2_study)  
**시리즈 전체**: [GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈](https://github.com/BanHun28/gpt2_study)  

