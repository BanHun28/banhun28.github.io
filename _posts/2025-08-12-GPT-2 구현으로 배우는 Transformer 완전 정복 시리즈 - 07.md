---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 07
date: 2025-08-12 08:10:27 +0900
categories: [machine learning, GPT]
tags: [machine learning, GPT, Transformer]       # TAG names should always be lowercase
---

# GPT 완전 정복 7편: 텍스트 생성의 예술 - 창작하는 AI

> **이전 편 요약**: 6편에서는 모델이 어떻게 학습을 통해 지능을 획득하는지 배웠습니다. 이제 학습된 모델이 어떻게 새로운 텍스트를 창작하는지, 그 예술적이면서도 과학적인 과정을 완전히 이해해보겠습니다.

---

## 들어가며: 창작의 순간

시인이 빈 종이 앞에서 첫 단어를 고민하는 순간을 상상해보세요:

```
시인의 고민: "사랑에 대한 시를 쓰고 싶어..."
첫 번째 단어: "사랑"? "그대"? "밤하늘"? "꽃잎"?
선택: "그대" (감정적 울림을 고려)
두 번째 단어: "그대의"? "그대는"? "그대여"?
선택: "그대의" (자연스러운 흐름)
...계속 이어짐
```

**GPT도 똑같은 과정을 거칩니다.**

```
GPT의 창작: "ROMEO:"에서 시작
첫 번째 단어 후보: "But"(23%), "O"(18%), "What"(12%), "Alas"(8%)...
선택 전략: 확률적 샘플링으로 "But" 선택
두 번째 단어 후보: "soft"(31%), "hark"(15%), "what"(12%)...
선택: "soft" 
...셰익스피어 스타일 대사 완성
```

이것이 바로 **자기회귀적 생성(Autoregressive Generation)**의 마법입니다.

---

## 1. Autoregressive Generation: 한 단어씩 세상을 만들어가기

### 우리 구현의 핵심: generate() 함수

[레포지토리](https://github.com/BanHun28/gpt2_study)의 `main.py`에서:

```python
@torch.no_grad()  # 그래디언트 계산을 비활성화 (추론시 메모리 절약)
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    텍스트 생성 메서드
    주어진 시작 토큰들에서 시작하여 새로운 토큰들을 생성
    """
    for _ in range(max_new_tokens):  # 지정된 개수만큼 토큰 생성
        # 컨텍스트가 모델의 최대 처리 길이를 초과하면 뒤쪽만 사용
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        # 다음 토큰 예측
        logits, _ = self(idx_cond)  # 모델 예측 수행
        logits = logits[:, -1, :] / temperature  # 마지막 위치만 사용하고 온도 적용
        
        # top-k 필터링: 상위 k개 토큰만 고려하여 다양성 조절
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 로짓을 확률로 변환하고 샘플링
        probs = F.softmax(logits, dim=-1)  # 확률 분포로 변환
        idx_next = torch.multinomial(probs, num_samples=1)  # 확률에 따라 다음 토큰 샘플링
        
        # 생성된 토큰을 기존 시퀀스에 추가
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx
```

### 단계별 생성 과정 완전 추적

```python
# 예시: "ROMEO:" → "ROMEO: But soft, what light"

# 초기 상태
current_sequence = [15496, 11]  # "ROMEO:"

# Step 1: 첫 번째 단어 생성
input_to_model = [15496, 11]
model_output = model(input_to_model)
logits = model_output[:, -1, :]  # 마지막 위치의 logits

# 50257개 단어에 대한 점수 (예시)
logits_sample = {
    "But": 2.34,     # 높은 점수
    "O": 1.87,       # 중간 점수  
    "What": 1.45,    # 중간 점수
    "The": 0.23,     # 낮은 점수
    "xyz": -3.45,    # 매우 낮은 점수
    ...
}

# 확률로 변환
probabilities = softmax(logits_sample)
# {"But": 0.234, "O": 0.187, "What": 0.145, ...}

# 샘플링으로 "But" 선택 (284번 토큰)
next_token = 284

# 시퀀스 업데이트
current_sequence = [15496, 11, 284]  # "ROMEO: But"

# Step 2: 두 번째 단어 생성
input_to_model = [15496, 11, 284]
# 모델이 이제 "ROMEO: But" 전체를 보고 다음 단어 예측
...
```

### Autoregressive의 핵심 특징들

#### 1. 순차적 의존성

```python
# 각 단어는 이전 모든 단어에 의존
word_1 = P(w1 | start_token)
word_2 = P(w2 | start_token, w1)  
word_3 = P(w3 | start_token, w1, w2)
word_4 = P(w4 | start_token, w1, w2, w3)
...

# 전체 시퀀스의 확률
P(전체_시퀀스) = P(w1) × P(w2|w1) × P(w3|w1,w2) × ...
```

#### 2. 불가역성

```python
# 한 번 선택한 단어는 되돌릴 수 없음
if current_word == "wrong_choice":
    # 이미 늦음! 다음 단어들도 영향받음
    # 처음부터 다시 생성해야 함

# 이것이 생성의 흥미로운 점: 매번 다른 결과
```

#### 3. 컨텍스트 길이 제한

```python
# 모델이 기억할 수 있는 길이 제한
max_context = 1024  # GPT-2의 경우

if len(current_sequence) > max_context:
    # 오래된 토큰들을 제거 (sliding window)
    context = current_sequence[-max_context:]
else:
    context = current_sequence

# 장문 생성 시 초반 내용을 "잊어버림"
```

---

## 2. Temperature: 창의성과 일관성의 균형

### Temperature의 수학적 정의

```python
# 원본 logits
logits = [2.0, 1.0, 0.5, 0.3, 0.1]

# Temperature 적용
def apply_temperature(logits, temperature):
    return logits / temperature

# 다양한 Temperature 효과
temp_0_5 = apply_temperature(logits, 0.5)  # [4.0, 2.0, 1.0, 0.6, 0.2]
temp_1_0 = apply_temperature(logits, 1.0)  # [2.0, 1.0, 0.5, 0.3, 0.1] (원본)
temp_2_0 = apply_temperature(logits, 2.0)  # [1.0, 0.5, 0.25, 0.15, 0.05]
```

### Temperature별 확률 분포 변화

```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, 0.3, 0.1])

# Temperature = 0.1 (매우 보수적)
probs_01 = F.softmax(logits / 0.1, dim=-1)
print("T=0.1:", probs_01)  # [0.99, 0.01, 0.00, 0.00, 0.00] - 거의 확정적

# Temperature = 1.0 (원본)  
probs_10 = F.softmax(logits / 1.0, dim=-1)
print("T=1.0:", probs_10)  # [0.53, 0.19, 0.11, 0.09, 0.08] - 균형적

# Temperature = 2.0 (창의적)
probs_20 = F.softmax(logits / 2.0, dim=-1)  
print("T=2.0:", probs_20)  # [0.40, 0.22, 0.16, 0.14, 0.12] - 더 고른 분포
```

### 실제 텍스트 생성에서의 Temperature 효과

```python
# "ROMEO:" 시작으로 다양한 Temperature 실험

Temperature = 0.2 (보수적):
"ROMEO: But soft, what light through yonder window breaks?
It is the east, and Juliet is the sun."
→ 예측 가능, 정확한 셰익스피어 인용

Temperature = 0.8 (균형):  
"ROMEO: But soft, what gentle spirit walks these halls?
Methinks I hear the whispers of sweet love."
→ 셰익스피어 스타일이지만 새로운 내용

Temperature = 1.5 (창의적):
"ROMEO: But soft, the moonbeams dance like silver tears,
And every shadow speaks of forgotten dreams."
→ 더 시적이고 독창적이지만 다소 추상적

Temperature = 3.0 (과도하게 창의적):
"ROMEO: But zebra quantum pickle seventeen blue..."
→ 의미 없는 내용, 일관성 상실
```

### Temperature 선택 가이드라인

```python
def choose_temperature(purpose):
    """용도별 적절한 Temperature 추천"""
    
    guidelines = {
        "정확한_인용": 0.1,        # 기존 텍스트 재현
        "기술_문서": 0.3,          # 정확성이 중요
        "일반_대화": 0.7,          # 자연스럽고 적당히 창의적
        "창작_글쓰기": 1.0,        # 창의성과 일관성 균형
        "시_창작": 1.3,            # 더 시적이고 독창적
        "실험적_창작": 2.0,        # 예상치 못한 조합
    }
    
    return guidelines.get(purpose, 0.8)

print("소설 창작용:", choose_temperature("창작_글쓰기"))  # 1.0
print("기술 문서용:", choose_temperature("기술_문서"))    # 0.3
```

---

## 3. Top-k Sampling: 현실적인 선택지 제한

### Top-k의 동작 원리

```python
def top_k_sampling_example():
    """Top-k 샘플링 동작 방식 시연"""
    
    # 원본 확률 분포 (50257개 단어 중 상위 10개만 표시)
    word_probs = {
        "But": 0.23,     # 1순위
        "O": 0.18,       # 2순위  
        "What": 0.12,    # 3순위
        "Soft": 0.08,    # 4순위
        "Hark": 0.06,    # 5순위
        "Come": 0.05,    # 6순위
        "Now": 0.04,     # 7순위
        "Yet": 0.03,     # 8순위
        "Fair": 0.02,    # 9순위
        "Sweet": 0.02,   # 10순위
        # ... 나머지 50247개 단어들 (매우 낮은 확률)
    }
    
    # Top-k=5 적용
    top_k_5 = {k: v for k, v in list(word_probs.items())[:5]}
    
    # 확률 재정규화 (선택된 단어들의 확률 합이 1이 되도록)
    total_prob = sum(top_k_5.values())  # 0.67
    normalized_probs = {k: v/total_prob for k, v in top_k_5.items()}
    
    print("Top-k=5 재정규화 후:")
    for word, prob in normalized_probs.items():
        print(f"{word}: {prob:.3f}")
    
    # 결과: 상위 5개 단어만으로 선택 범위 제한
    # "xyz", "nonsense" 같은 이상한 단어들 배제

top_k_sampling_example()
```

### Top-k 값에 따른 효과 비교

```python
def compare_top_k_effects():
    """다양한 Top-k 값의 효과 비교"""
    
    scenarios = {
        "top_k_1": "가장 확률 높은 1개만 → 항상 같은 결과 (Greedy)",
        "top_k_5": "상위 5개만 → 안전하고 품질 높음",  
        "top_k_20": "상위 20개 → 적당한 다양성",
        "top_k_50": "상위 50개 → 창의적이지만 가끔 이상함",
        "top_k_None": "모든 단어 고려 → 매우 창의적, 때로는 횡설수설"
    }
    
    # 실제 생성 예시 (개념적)
    examples = {
        "top_k_1": "ROMEO: But soft, what light through yonder window breaks?",
        "top_k_5": "ROMEO: But soft, what gentle voice calls from above?", 
        "top_k_20": "ROMEO: But soft, what strange melody fills the night air?",
        "top_k_50": "ROMEO: But soft, what peculiar shadows dance in moonlight?",
        "top_k_None": "ROMEO: But soft, what purple elephants sing opera tonight?"
    }
    
    print("=== Top-k 값별 생성 결과 비교 ===")
    for k, description in scenarios.items():
        print(f"\n{k}: {description}")
        print(f"예시: {examples[k]}")

compare_top_k_effects()
```

### Top-k의 장단점

```python
# 장점:
✅ 이상한 단어 배제: "xyz", "qwerty" 같은 무의미한 토큰 제거
✅ 품질 보장: 최소한의 문법적/의미적 타당성 확보
✅ 계산 효율성: 모든 어휘 대신 일부만 고려
✅ 조절 가능: k 값으로 다양성 정도 제어

# 단점:  
❌ 고정된 제한: 상황에 관계없이 항상 k개만 고려
❌ 확률 분포 무시: 2순위와 k순위의 확률 차이를 고려하지 않음
❌ 창의성 제한: 때로는 낮은 확률이지만 훌륭한 선택을 배제
```

---

## 4. Top-p (Nucleus) Sampling: 동적인 선택지 조절

### Top-p의 혁신적 아이디어

```python
def nucleus_sampling_example():
    """Nucleus (Top-p) 샘플링 시연"""
    
    # 확률 분포 (내림차순 정렬)
    sorted_probs = [
        ("But", 0.30),
        ("O", 0.25), 
        ("What", 0.15),
        ("Soft", 0.10),
        ("Hark", 0.08),
        ("Come", 0.05),
        ("Now", 0.03),
        ("Yet", 0.02),
        ("Fair", 0.01),
        ("Sweet", 0.01)
    ]
    
    # Top-p = 0.8 적용
    cumulative_prob = 0.0
    selected_words = []
    
    for word, prob in sorted_probs:
        cumulative_prob += prob
        selected_words.append((word, prob))
        
        print(f"{word}: {prob:.2f} (누적: {cumulative_prob:.2f})")
        
        if cumulative_prob >= 0.8:  # 80%에 도달하면 중단
            print(f"→ Top-p=0.8 도달, {len(selected_words)}개 단어 선택")
            break
    
    # 결과: 상황에 따라 선택되는 단어 수가 다름
    # 확률 분포가 집중되어 있으면 적은 수, 고르면 많은 수

nucleus_sampling_example()
```

### Top-k vs Top-p 비교

```python
def compare_top_k_vs_top_p():
    """Top-k와 Top-p 샘플링 비교"""
    
    # 시나리오 1: 확률이 집중된 경우
    concentrated_dist = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
    
    print("=== 확률 집중된 상황 ===")
    print("분포:", concentrated_dist)
    
    # Top-k=3: 항상 3개
    print("Top-k=3: 상위 3개 선택 (0.5, 0.3, 0.1)")
    
    # Top-p=0.9: 확률 90%까지
    cumul = 0
    for i, p in enumerate(concentrated_dist):
        cumul += p
        if cumul >= 0.9:
            print(f"Top-p=0.9: 상위 {i+1}개 선택 (누적 {cumul:.1f})")
            break
    
    print("\n=== 확률 분산된 상황 ===")
    # 시나리오 2: 확률이 분산된 경우  
    distributed_dist = [0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.07, 0.05, 0.03, 0.02]
    print("분포:", distributed_dist)
    
    print("Top-k=3: 여전히 3개만 선택")
    
    cumul = 0
    for i, p in enumerate(distributed_dist):
        cumul += p
        if cumul >= 0.9:
            print(f"Top-p=0.9: 상위 {i+1}개 선택 (더 많은 선택지)")
            break

compare_top_k_vs_top_p()
```

### Top-p 구현 세부사항

```python
def top_p_sampling(logits, p=0.9):
    """Top-p (Nucleus) 샘플링 구현"""
    
    # 1. 확률로 변환
    probs = F.softmax(logits, dim=-1)
    
    # 2. 내림차순 정렬
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 3. 누적 확률 계산
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 4. p 임계값을 넘는 부분 마스킹
    # cumulative_probs > p인 위치를 찾되, 첫 번째는 항상 포함
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # 첫 번째는 항상 유지
    
    # 5. 원래 순서로 되돌리기
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0
    
    # 6. 재정규화
    probs = probs / probs.sum()
    
    # 7. 샘플링
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
```

---

## 5. Beam Search: 더 나은 품질을 찾아서

### Greedy vs Beam Search

#### Greedy Decoding의 한계

```python
# Greedy: 각 단계에서 가장 확률 높은 단어만 선택

Step 1: "ROMEO:" → "But" (0.3)
Step 2: "ROMEO: But" → "soft" (0.4)  
Step 3: "ROMEO: But soft" → "what" (0.35)

# 전체 확률: 0.3 × 0.4 × 0.35 = 0.042

# 문제: 국소 최적해에 빠질 수 있음
# 더 나은 전체 경로가 있을 수도 있음:
Alternative path:
Step 1: "ROMEO:" → "O" (0.2)  # 낮은 시작
Step 2: "ROMEO: O" → "Juliet" (0.8)  # 높은 확률
Step 3: "ROMEO: O Juliet" → "my" (0.7)  # 높은 확률

# 전체 확률: 0.2 × 0.8 × 0.7 = 0.112 (더 높음!)
```

#### Beam Search의 해결책

```python
def beam_search_example(beam_size=3):
    """Beam Search 동작 과정 시연"""
    
    # 초기 상태: "ROMEO:"
    beams = [
        {"sequence": ["ROMEO:"], "score": 0.0}  # log 확률 사용
    ]
    
    print("=== Beam Search 과정 ===")
    print(f"Beam Size: {beam_size}")
    
    # Step 1: 첫 번째 단어 생성
    candidates = []
    for beam in beams:
        # 각 beam에서 가능한 다음 단어들
        next_words = [
            ("But", -1.20),   # log(0.3)
            ("O", -1.61),     # log(0.2)  
            ("What", -1.90),  # log(0.15)
            ("Soft", -2.30),  # log(0.1)
        ]
        
        for word, log_prob in next_words:
            new_sequence = beam["sequence"] + [word]
            new_score = beam["score"] + log_prob
            candidates.append({
                "sequence": new_sequence,
                "score": new_score
            })
    
    # 상위 beam_size개만 유지
    beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]
    
    print("Step 1 결과:")
    for i, beam in enumerate(beams):
        print(f"Beam {i+1}: {' '.join(beam['sequence'])} (점수: {beam['score']:.2f})")
    
    # Step 2: 두 번째 단어 생성 (같은 방식으로 진행)
    print("\nStep 2 후 상위 beam들:")
    # ... 실제로는 같은 과정 반복

beam_search_example()
```

### Beam Search의 장단점

```python
# 장점:
✅ 더 나은 전체 품질: 국소 최적해 회피
✅ 일관성 있는 텍스트: 더 논리적인 구조
✅ 예측 가능한 결과: 같은 입력에 대해 항상 같은 출력

# 단점:
❌ 계산 비용: beam_size배 만큼 연산량 증가  
❌ 다양성 부족: 안전한 선택 위주
❌ 반복 문제: 같은 패턴 반복 경향
❌ 길이 편향: 짧은 문장 선호 (확률 곱이므로)
```

### Beam Search 개선 기법들

```python
def improved_beam_search():
    """개선된 Beam Search 기법들"""
    
    improvements = {
        "Length Normalization": {
            "문제": "긴 문장일수록 확률이 낮아짐 (곱셈 때문에)",
            "해결": "점수를 길이로 나누거나 길이 패널티 적용",
            "공식": "score / (length ** alpha)"
        },
        
        "Coverage Penalty": {
            "문제": "같은 내용 반복",
            "해결": "이미 언급된 내용에 패널티",
            "효과": "더 다양한 내용 생성"
        },
        
        "Diverse Beam Search": {
            "문제": "비슷한 beam들만 생성",  
            "해결": "beam들 간의 다양성 강제",
            "방법": "그룹별로 다른 방향 탐색"
        }
    }
    
    for technique, details in improvements.items():
        print(f"\n=== {technique} ===")
        for aspect, description in details.items():
            print(f"{aspect}: {description}")

improved_beam_search()
```

---

## 6. 실제 생성 과정 완전 분석

### 우리 구현의 generate() 함수 심층 분석

```python
def analyze_generation_step_by_step():
    """실제 생성 과정을 단계별로 분석"""
    
    # 1. 입력 준비
    prompt = "ROMEO:"
    tokens = enc.encode(prompt)  # [15496, 11]
    input_ids = torch.tensor([tokens])
    
    print("=== 생성 과정 단계별 분석 ===")
    print(f"초기 입력: {prompt} → {tokens}")
    
    # 2. 모델 설정
    model.eval()
    max_new_tokens = 10
    temperature = 0.8
    top_k = 40
    
    current_sequence = input_ids.clone()
    
    for step in range(max_new_tokens):
        print(f"\n--- Step {step + 1} ---")
        
        # 2.1. 컨텍스트 길이 확인
        seq_len = current_sequence.size(1)
        if seq_len > model.config.block_size:
            context = current_sequence[:, -model.config.block_size:]
            print(f"컨텍스트 잘림: {seq_len} → {model.config.block_size}")
        else:
            context = current_sequence
            print(f"컨텍스트 길이: {seq_len}")
        
        # 2.2. 모델 순전파
        with torch.no_grad():
            logits, _ = model(context)
            next_token_logits = logits[0, -1, :]  # 마지막 위치
        
        # 2.3. Temperature 적용
        scaled_logits = next_token_logits / temperature
        print(f"Temperature {temperature} 적용")
        
        # 2.4. Top-k 필터링
        if top_k is not None:
            values, indices = torch.topk(scaled_logits, top_k)
            filtered_logits = torch.full_like(scaled_logits, float('-inf'))
            filtered_logits[indices] = values
            print(f"Top-{top_k} 필터링 적용")
        else:
            filtered_logits = scaled_logits
        
        # 2.5. 확률 변환 및 샘플링
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 2.6. 결과 출력
        next_word = enc.decode([next_token.item()])
        probability = probs[next_token].item()
        
        print(f"선택된 토큰: {next_token.item()} → '{next_word}'")
        print(f"선택 확률: {probability:.3f}")
        
        # 2.7. 시퀀스 업데이트
        current_sequence = torch.cat([current_sequence, next_token.unsqueeze(0)], dim=1)
        
        # 현재까지 생성된 텍스트
        current_text = enc.decode(current_sequence[0].tolist())
        print(f"현재 시퀀스: {current_text}")
    
    return current_text

# 실행
final_text = analyze_generation_step_by_step()
print(f"\n최종 결과: {final_text}")
```

### 생성 품질에 영향을 주는 요소들

```python
def generation_quality_factors():
    """생성 품질에 영향을 주는 주요 요소들"""
    
    factors = {
        "모델 크기": {
            "영향": "큰 모델일수록 더 풍부한 표현과 일관성",
            "예시": "GPT-2 Small vs XL의 품질 차이",
            "권장": "가능한 범위에서 큰 모델 사용"
        },
        
        "프롬프트 품질": {
            "영향": "좋은 시작이 좋은 결과를 만듦",
            "예시": "'Write a story' vs 'In the misty mountains where dragons dwell'",
            "권장": "구체적이고 흥미로운 프롬프트 작성"
        },
        
        "하이퍼파라미터 조합": {
            "영향": "Temperature, Top-k, Top-p의 균형",
            "예시": "창의적 글쓰기 vs 기술 문서의 다른 설정",
            "권장": "용도에 맞는 파라미터 튜닝"
        },
        
        "학습 데이터 품질": {
            "영향": "모델이 학습한 데이터의 품질이 생성에 직접 영향",
            "예시": "셰익스피어 데이터로 학습 → 셰익스피어 스타일 생성",
            "권장": "고품질, 다양한 학습 데이터 사용"
        }
    }
    
    print("=== 생성 품질 영향 요소 ===")
    for factor, details in factors.items():
        print(f"\n{factor}:")
        for aspect, description in details.items():
            print(f"  {aspect}: {description}")

generation_quality_factors()
```

---

## 7. 생성 품질 평가 방법들

### 자동 평가 지표들

```python
def automatic_evaluation_metrics():
    """자동 생성 텍스트 평가 지표들"""
    
    metrics = {
        "Perplexity": {
            "정의": "모델이 텍스트를 얼마나 '예상'했는지 측정",
            "계산": "exp(평균 교차엔트로피 손실)",
            "해석": "낮을수록 좋음 (더 예측 가능한 텍스트)",
            "한계": "예측 가능 != 좋은 품질"
        },
        
        "BLEU Score": {
            "정의": "생성 텍스트와 참조 텍스트의 n-gram 일치도",
            "계산": "1-gram~4-gram 정밀도의 기하평균",
            "해석": "0~1, 높을수록 좋음",
            "한계": "창의적 텍스트에는 부적합"
        },
        
        "Diversity Metrics": {
            "정의": "생성된 텍스트의 다양성 측정",
            "종류": "Distinct-1, Distinct-2 (고유 n-gram 비율)",
            "해석": "높을수록 다양하고 반복적이지 않음",
            "중요성": "반복 문제 감지"
        },
        
        "Semantic Coherence": {
            "정의": "문장 간 의미적 일관성",
            "측정": "문장 임베딩 간 코사인 유사도",
            "해석": "적절한 범위의 일관성 필요",
            "도구": "BERT, Sentence-BERT 등 활용"
        }
    }
    
    print("=== 자동 평가 지표 ===")
    for metric, details in metrics.items():
        print(f"\n{metric}:")
        for aspect, description in details.items():
            print(f"  {aspect}: {description}")

automatic_evaluation_metrics()
```

### 인간 평가 기준들

```python
def human_evaluation_criteria():
    """인간이 평가하는 텍스트 품질 기준"""
    
    criteria = {
        "Fluency (유창성)": {
            "질문": "문법적으로 올바르고 자연스러운가?",
            "평가": "1~5점 척도",
            "예시": {
                5: "완벽한 문법과 자연스러운 흐름",
                3: "약간의 어색함이 있지만 이해 가능",
                1: "문법 오류가 많고 이해하기 어려움"
            }
        },
        
        "Coherence (일관성)": {
            "질문": "논리적으로 일관되고 주제가 명확한가?",
            "평가": "1~5점 척도", 
            "예시": {
                5: "명확한 주제와 논리적 전개",
                3: "대체로 일관되지만 일부 혼란",
                1: "주제가 불분명하고 논리적 연결 부족"
            }
        },
        
        "Creativity (창의성)": {
            "질문": "독창적이고 흥미로운 내용인가?",
            "평가": "1~5점 척도",
            "주의": "창의성과 일관성의 균형 필요"
        },
        
        "Relevance (관련성)": {
            "질문": "주어진 프롬프트와 얼마나 관련 있는가?",
            "평가": "1~5점 척도",
            "중요성": "프롬프트 의도 파악 능력 측정"
        }
    }
    
    print("=== 인간 평가 기준 ===")
    for criterion, details in criteria.items():
        print(f"\n{criterion}:")
        for aspect, description in details.items():
            if isinstance(description, dict):
                print(f"  {aspect}:")
                for score, example in description.items():
                    print(f"    {score}: {example}")
            else:
                print(f"  {aspect}: {description}")

human_evaluation_criteria()
```

---

## 8. 실습: 고급 생성 기법 구현

### 실습 1: 다중 샘플링 방법 비교

```python
def compare_sampling_methods():
    """다양한 샘플링 방법 비교 실험"""
    
    prompt = "ROMEO:"
    max_tokens = 30
    
    methods = {
        "greedy": {"temperature": 0.0, "top_k": 1},
        "low_temp": {"temperature": 0.3, "top_k": None},
        "balanced": {"temperature": 0.8, "top_k": 40},
        "creative": {"temperature": 1.2, "top_k": 100},
        "wild": {"temperature": 2.0, "top_k": None}
    }
    
    print("=== 샘플링 방법별 생성 결과 비교 ===")
    
    for method_name, params in methods.items():
        print(f"\n=== {method_name.upper()} ===")
        print(f"설정: Temperature={params['temperature']}, Top-k={params['top_k']}")
        
        # 같은 프롬프트로 3번 생성
        for trial in range(3):
            tokens = enc.encode(prompt)
            input_ids = torch.tensor([tokens])
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=params['temperature'],
                    top_k=params['top_k']
                )
            
            result = enc.decode(generated[0].tolist())
            print(f"시도 {trial + 1}: {result}")
        
        # 특성 분석
        if params['temperature'] < 0.5:
            print("→ 특성: 예측 가능, 일관성 높음, 창의성 낮음")
        elif params['temperature'] < 1.0:
            print("→ 특성: 균형잡힌 품질과 다양성")
        else:
            print("→ 특성: 높은 창의성, 예측 불가능성")

compare_sampling_methods()
```

### 실습 2: 커스텀 샘플링 전략

```python
def custom_sampling_strategy():
    """커스텀 샘플링 전략 구현"""
    
    def adaptive_temperature_sampling(logits, position, sequence_length):
        """위치에 따라 적응적으로 temperature 조절"""
        
        # 시작 부분: 보수적 (일관성 중요)
        if position < sequence_length * 0.3:
            temp = 0.6
        # 중간 부분: 균형적 (창의성과 일관성)
        elif position < sequence_length * 0.7:
            temp = 0.9
        # 마지막 부분: 창의적 (마무리의 임팩트)
        else:
            temp = 1.2
            
        return logits / temp
    
    def repetition_penalty_sampling(logits, generated_tokens, penalty=1.2):
        """반복 방지를 위한 페널티 적용"""
        
        # 이미 생성된 토큰들에 페널티
        for token in set(generated_tokens):
            logits[token] /= penalty
            
        return logits
    
    def combined_sampling(prompt, max_tokens=50):
        """여러 전략을 조합한 샘플링"""
        
        tokens = enc.encode(prompt)
        generated_tokens = tokens.copy()
        
        for position in range(max_tokens):
            # 현재 컨텍스트로 예측
            input_ids = torch.tensor([generated_tokens[-model.config.block_size:]])
            
            with torch.no_grad():
                logits, _ = model(input_ids)
                next_logits = logits[0, -1, :].clone()
            
            # 1. 적응적 temperature 적용
            next_logits = adaptive_temperature_sampling(
                next_logits, position, max_tokens
            )
            
            # 2. 반복 페널티 적용
            next_logits = repetition_penalty_sampling(
                next_logits, generated_tokens
            )
            
            # 3. Top-k 필터링
            top_k = 50
            if top_k > 0:
                values, indices = torch.topk(next_logits, top_k)
                filtered_logits = torch.full_like(next_logits, float('-inf'))
                filtered_logits[indices] = values
                next_logits = filtered_logits
            
            # 4. 샘플링
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token)
            
            # 종료 조건 (문장 끝)
            if next_token == enc.encode('.')[0]:
                break
        
        return enc.decode(generated_tokens)
    
    # 테스트
    print("=== 커스텀 샘플링 결과 ===")
    for i in range(3):
        result = combined_sampling("ROMEO:")
        print(f"\n생성 {i+1}: {result}")

custom_sampling_strategy()
```

### 실습 3: 생성 품질 자동 평가

```python
def automatic_quality_assessment():
    """생성된 텍스트의 품질 자동 평가"""
    
    def calculate_perplexity(text, model):
        """Perplexity 계산"""
        tokens = enc.encode(text)
        if len(tokens) < 2:
            return float('inf')
        
        input_ids = torch.tensor([tokens[:-1]])
        target_ids = torch.tensor([tokens[1:]])
        
        with torch.no_grad():
            logits, loss = model(input_ids, target_ids)
        
        return torch.exp(loss).item()
    
    def calculate_diversity(text):
        """텍스트 다양성 계산"""
        words = text.lower().split()
        
        if len(words) < 2:
            return 0.0
        
        # Distinct-2: 고유한 2-gram 비율
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        return distinct_2
    
    def calculate_repetition_rate(text):
        """반복률 계산"""
        words = text.lower().split()
        
        if len(words) < 4:
            return 0.0
        
        # 4-gram 반복 검사
        fourgrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
        repeated = len(fourgrams) - len(set(fourgrams))
        
        return repeated / len(fourgrams) if fourgrams else 0
    
    # 다양한 설정으로 텍스트 생성 및 평가
    test_configs = [
        {"name": "Conservative", "temp": 0.3, "top_k": 20},
        {"name": "Balanced", "temp": 0.8, "top_k": 40},
        {"name": "Creative", "temp": 1.2, "top_k": 100}
    ]
    
    print("=== 생성 품질 자동 평가 ===")
    
    for config in test_configs:
        print(f"\n--- {config['name']} 설정 ---")
        
        # 텍스트 생성
        tokens = enc.encode("ROMEO:")
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=config['temp'],
                top_k=config['top_k']
            )
        
        text = enc.decode(generated[0].tolist())
        print(f"생성 텍스트: {text}")
        
        # 품질 지표 계산
        perplexity = calculate_perplexity(text, model)
        diversity = calculate_diversity(text)
        repetition = calculate_repetition_rate(text)
        
        print(f"Perplexity: {perplexity:.2f} (낮을수록 좋음)")
        print(f"Diversity: {diversity:.3f} (높을수록 좋음)")
        print(f"Repetition Rate: {repetition:.3f} (낮을수록 좋음)")
        
        # 종합 점수 (간단한 가중합)
        score = (1/perplexity) * diversity * (1-repetition) * 100
        print(f"종합 점수: {score:.2f}")

automatic_quality_assessment()
```

---

## 마무리: 창작하는 AI의 비밀을 마스터했습니다

### 오늘 배운 핵심 내용

1. **Autoregressive Generation**: 한 단어씩 순차적으로 생성하는 원리
2. **Temperature Sampling**: 창의성과 일관성의 균형 조절 기법
3. **Top-k Sampling**: 현실적인 선택지로 제한하여 품질 보장
4. **Top-p (Nucleus) Sampling**: 동적으로 선택지를 조절하는 혁신
5. **Beam Search**: 더 나은 전체 품질을 위한 탐색 전략

### 생성 전략의 완전한 이해

```
좋은 텍스트 생성 = 적절한 샘플링 전략 + 품질 평가 + 반복 개선

핵심 원칙:
1. 용도에 맞는 파라미터 선택
2. 창의성과 일관성의 균형
3. 반복과 이상한 표현 방지
4. 지속적인 품질 모니터링
```

### 실무 적용 가이드라인

```python
# 용도별 추천 설정

창작 글쓰기:
- Temperature: 0.9~1.1
- Top-k: 40~60  
- Top-p: 0.8~0.95

기술 문서:
- Temperature: 0.3~0.5
- Top-k: 10~20
- Top-p: 0.7~0.8

대화형 챗봇:
- Temperature: 0.7~0.9
- Top-k: 30~50
- Top-p: 0.85~0.95

코드 생성:
- Temperature: 0.2~0.4
- Top-k: 5~15
- Beam Search 고려
```

### 다음 편 예고: 실전 구현의 모든 것

다음 편에서는 지금까지 배운 모든 지식을 실전에 적용하는 **고급 구현 기법들**을 다룹니다:

- **메모리 최적화**: GPU 메모리를 효율적으로 사용하는 방법
- **속도 최적화**: 추론 속도를 극대화하는 기술들
- **배치 처리**: 여러 텍스트를 동시에 효율적으로 생성
- **모델 압축**: 성능 유지하면서 크기 줄이기
- **실전 디버깅**: 문제 상황 진단과 해결
- **배포 최적화**: 실제 서비스에서 사용하기 위한 엔지니어링

**미리 생각해볼 질문:**
지금까지 배운 아름다운 이론들을 실제 서비스에 적용할 때 어떤 실무적 고려사항들이 있을까요? 성능과 품질, 비용의 균형을 어떻게 맞출까요?

### 실습 과제

다음 편까지 해볼 과제:

1. **샘플링 전략 실험**: 다양한 Temperature와 Top-k 조합으로 최적 설정 찾기
2. **품질 평가 도구 개발**: 생성된 텍스트를 자동으로 평가하는 스크립트 작성
3. **창의적 프롬프트 실험**: 흥미로운 시작 문구들로 다양한 텍스트 생성 테스트

이제 GPT의 모든 이론을 완전히 마스터했으니, 실전에서 활용하는 엔지니어링 노하우를 배워봅시다! 🚀

---

**이전 편**: [6편: 학습의 과학 - 모델이 "배우는" 과정](링크)  
**다음 편**: [8편: 실전 구현 분석 - 코드 한 줄씩 완전 분해](링크)  
**시리즈 전체**: [GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈](링크)

