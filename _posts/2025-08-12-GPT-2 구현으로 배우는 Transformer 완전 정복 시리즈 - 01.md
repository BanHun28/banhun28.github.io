---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 01
date: 2025-08-12 08:10:17 +0900
categories: [machine learning, GPT]
tags: [machine learning, GPT, Transformer]      # TAG names should always be lowercase
---

# GPT 완전 정복 1편: GPT란 무엇인가? - 혁신의 시작점 이해하기

> **시리즈 소개**: 이 글은 [BanHun28/gpt2_study](https://github.com/BanHun28/gpt2_study) 레포지토리의 실제 GPT-2 구현 코드를 기반으로 한 학습 시리즈의 첫 번째 편입니다. 이론과 실제 구현을 완벽하게 연결하여 Transformer 아키텍처를 완전히 이해하는 것이 목표입니다.

---

## 들어가며: 왜 GPT를 배워야 할까?

2023년 ChatGPT가 세상을 뒤흔들었습니다. 하지만 이 혁신의 뿌리는 2017년 구글이 발표한 "Attention Is All You Need" 논문으로 거슬러 올라갑니다. 그리고 OpenAI의 GPT(Generative Pre-trained Transformer) 시리즈가 이 기술을 실용화의 단계로 끌어올렸죠.

**왜 모든 AI 개발자가 GPT를 이해해야 할까요?**

1. **현대 AI의 기반 기술**: 거의 모든 최신 언어 모델이 Transformer 기반
2. **실무 필수 지식**: GPT 구조를 이해해야 모델을 효과적으로 활용 가능
3. **미래 기술의 출발점**: GPT-4, Llama, Claude 등 모든 최신 모델의 기본 원리

이 시리즈에서는 단순히 이론만 배우는 것이 아니라, **실제로 동작하는 GPT-2 모델을 직접 구현**하면서 모든 개념을 체득할 것입니다.

---

## 1. GPT의 혁신: 기존 방식과 무엇이 다른가?

### 기존 언어 모델의 한계

2017년 이전의 언어 모델들을 살펴보겠습니다:

#### RNN/LSTM 시대 (2010년대)
```
입력: "나는 학교에 간다"

RNN 처리 과정:
단계1: "나는" → 은닉상태1
단계2: 은닉상태1 + "학교에" → 은닉상태2  
단계3: 은닉상태2 + "간다" → 출력
```

**문제점:**
- **순차 처리**: 앞 단어를 처리해야 다음 단어 처리 가능 → 병렬화 불가
- **장기 의존성**: 문장이 길어지면 초반 정보가 손실
- **학습 불안정**: 그래디언트 소실/폭발 문제

#### CNN 기반 모델들
- 지역적 패턴은 잘 잡아내지만 긴 문맥 이해에 한계
- 단어 간 거리가 멀면 관계 파악 어려움

### GPT의 혁신적 접근

```python
# 우리가 구현할 GPT의 핵심 아이디어 (main.py에서)

class CausalSelfAttention(nn.Module):
    """
    모든 단어가 동시에 서로를 바라봄
    하지만 미래는 볼 수 없음 (Causal)
    """
    def forward(self, x):
        # 모든 위치가 병렬로 처리됨!
        # 각 단어가 이전의 모든 단어들과 직접 소통
```

**GPT의 3대 혁신:**

1. **병렬 처리**: 모든 단어를 동시에 처리
2. **직접 연결**: 단어 간 거리에 상관없이 직접 소통
3. **스케일링**: 모델이 클수록 성능이 좋아짐

---

## 2. GPT의 핵심 철학: "다음 단어 맞히기"의 마법

### 언어 모델링의 본질

GPT의 학습은 놀랍도록 단순합니다:

```
주어진 문장: "나는 학교에 간다"

학습 과정:
"나는" → 다음 단어는? "학교에" ✓
"나는 학교에" → 다음 단어는? "간다" ✓
"나는 학교에 간다" → 다음 단어는? [문장끝] ✓
```

**왜 이것만으로 충분할까?**

다음 단어를 정확히 예측하려면:
- **문법**을 알아야 함 (주어 다음에는 동사가 올 확률이 높음)
- **의미**를 이해해야 함 (학교와 관련된 동사 선택)
- **문맥**을 파악해야 함 (화자의 의도 이해)
- **상식**이 필요함 (학교에 가는 것이 자연스러운 행동)

즉, **단순한 "다음 단어 맞히기"가 실제로는 언어의 모든 측면을 학습하게 만듭니다.**

### 우리 구현에서 확인해보기

```python
# main.py의 forward 함수에서
def forward(self, idx, targets=None):
    """
    idx: 현재까지의 단어들 [나는, 학교에]
    targets: 정답 단어들 [학교에, 간다]
    """
    # ... 복잡한 Transformer 연산들 ...
    
    if targets is not None:
        # 예측과 정답 비교 → 학습
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                              targets.view(-1))
    
    return logits, loss
```

---

## 3. Transformer: GPT의 두뇌 구조

### 핵심 아이디어: Attention

**"Attention"이란?**

사람이 문장을 읽을 때를 생각해보세요:

```
"그 남자는 공원에서 개를 산책시키고 있었다. 개가 갑자기 뛰어갔다."
```

두 번째 문장의 "개"를 이해하려면 첫 번째 문장의 "개"를 기억해야 합니다. 이것이 바로 **Attention**의 핵심입니다.

### Self-Attention의 마법

```python
# main.py의 CausalSelfAttention에서
def forward(self, x):
    # Query: "나는 무엇을 찾고 있나?"
    # Key: "나는 어떤 정보를 가지고 있나?"  
    # Value: "실제로 전달할 정보는 무엇인가?"
    
    qkv = self.c_attn(x)  # 하나의 입력을 3개로 변환
    q, k, v = qkv.split(self.n_embd, dim=2)
    
    # Attention Score 계산
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

**직관적 이해:**
- **Query**: "현재 단어가 궁금해하는 것"
- **Key**: "각 단어가 답해줄 수 있는 것"  
- **Value**: "실제로 전해줄 정보"

### Multi-Head Attention: 다면적 사고

```python
self.n_head = config.n_head  # 보통 12개의 헤드 사용
```

왜 여러 개의 "헤드"가 필요할까요?

**비유: 문장을 읽는 여러 시각**
- **헤드 1**: 문법적 관계에 집중 (주어-동사)
- **헤드 2**: 의미적 연관성에 집중 (동의어, 반의어)
- **헤드 3**: 시간적 순서에 집중 (과거-현재-미래)
- **...**: 각 헤드가 다른 패턴 학습

---

## 4. GPT 진화사: GPT-1에서 ChatGPT까지

### GPT-1 (2018): 개념 증명
- **파라미터**: 117M (1억 1700만개)
- **혁신**: Unsupervised Pre-training + Supervised Fine-tuning
- **성과**: Transfer Learning이 NLP에서도 가능함을 증명

### GPT-2 (2019): 스케일의 힘
- **파라미터**: 1.5B (15억개)
- **혁신**: Zero-shot Learning (별도 훈련 없이 다양한 작업 수행)
- **충격**: "인간 수준의 텍스트 생성"이라는 평가

### GPT-3 (2020): 범용 AI의 등장
- **파라미터**: 175B (1750억개)
- **혁신**: In-context Learning (예시만 보여줘도 학습)
- **파급**: API 서비스로 AI 대중화 시작

### ChatGPT/GPT-4 (2022-2023): AI의 대중화
- **혁신**: RLHF (인간 피드백을 통한 강화학습)
- **결과**: 전세계적 AI 열풍

### 우리가 구현할 모델

```python
# main.py의 설정에서
@dataclass
class GPTConfig:
    block_size: int = 1024    # GPT-2와 동일
    vocab_size: int = 50257   # GPT-2와 동일
    n_layer: int = 12         # GPT-2 Small과 동일
    n_head: int = 12          # GPT-2 Small과 동일
    n_embd: int = 768         # GPT-2 Small과 동일
```

우리는 **GPT-2 Small 모델**을 구현하여 GPT의 핵심 원리를 학습할 것입니다.

---

## 5. 왜 GPT가 이렇게 강력한가?

### 1. 스케일링 법칙
- **모델이 클수록** → 성능이 좋아짐
- **데이터가 많을수록** → 더 다양한 지식 습득
- **계산 자원이 많을수록** → 더 복잡한 패턴 학습

### 2. 일반화 능력
```python
# 한 번 학습하면 여러 작업 수행 가능
model.generate("번역: Hello")       # 번역
model.generate("요약: 긴 텍스트")   # 요약  
model.generate("질문: 답변")        # 질의응답
```

### 3. 창발적 능력 (Emergent Abilities)
모델이 커지면서 **예상치 못한 능력**들이 나타남:
- 추론 능력
- 코딩 능력
- 수학 문제 해결
- 창작 능력

---

## 6. 우리의 학습 여정 미리보기

### 다음 편에서 배울 내용

**2편: 토큰화의 비밀**
- 컴퓨터가 어떻게 "Hello" → [15496] 숫자로 변환하는지
- BPE(Byte Pair Encoding) 알고리즘의 원리
- tiktoken 라이브러리 실제 사용법

```python
# 2편에서 자세히 분석할 코드
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, world!")  # [15496, 11, 995, 0]
```

### 전체 여정 로드맵

```
1편 (현재) : GPT 개요 및 혁신성 이해
2편 : 토큰화 - 텍스트를 숫자로
3편 : Attention - 문맥 이해의 핵심  
4편 : Transformer Block - 지능의 구조
5편 : 전체 모델 - 퍼즐의 완성
6편 : 학습 과정 - AI가 배우는 방법
7편 : 텍스트 생성 - 창작하는 AI
8편 : 실전 최적화 - 실무 노하우
```

---

## 마무리: 첫 걸음을 내디뎠습니다

### 오늘 배운 핵심 내용

1. **GPT의 혁신**: 순차 처리 → 병렬 처리
2. **핵심 원리**: "다음 단어 맞히기"로 언어의 모든 것을 학습
3. **Attention**: 단어들이 서로 소통하는 메커니즘
4. **스케일의 힘**: 크면 클수록 더 똑똑해짐

### 다음 편 예고

다음 편에서는 **"Hello, world!"가 어떻게 [15496, 11, 995, 0]이 되는지** 알아보겠습니다. 

컴퓨터가 텍스트를 이해하는 첫 번째 단계인 **토큰화(Tokenization)**의 모든 것을 배우고, 실제 GPT-2에서 사용하는 tiktoken 라이브러리를 완전히 마스터해보겠습니다.

### 실습 과제

다음 편까지 해볼 수 있는 간단한 실습:

1. **레포지토리 클론**: [BanHun28/gpt2_study](https://github.com/BanHun28/gpt2_study)를 로컬에 복사
2. **환경 설정**: Python, PyTorch, tiktoken 설치
3. **코드 실행**: `python main.py`로 실제 GPT-2 학습 과정 체험
4. **결과 관찰**: 모델이 생성하는 셰익스피어 스타일 텍스트 확인

### 궁금한 점이 있다면?

이 시리즈는 **완전한 이해**를 목표로 합니다. 어떤 부분이 이해되지 않더라도 걱정하지 마세요. 다음 편들에서 모든 퍼즐 조각이 맞춰질 것입니다.

다음 편에서 만나요! 🚀
