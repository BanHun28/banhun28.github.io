---
title: GPT-2 구현으로 배우는 Transformer 완전 정복 시리즈 - 06
date: 2025-08-12 08:10:25 +0900
categories: [machine learning, GPT]
tags: [machine learning, GPT, Transformer]      # TAG names should always be lowercase
---

# GPT 완전 정복 6편: 학습의 과학 - 모델이 "배우는" 과정

> **이전 편 요약**: 5편에서는 완전한 GPT 모델의 전체 구조를 이해했습니다. 이제 이 모든 구조가 어떻게 "학습"을 통해 지능을 획득하는지, 그 과학적 원리를 완전히 파악해보겠습니다.

---

## 들어가며: 실수에서 배우는 지능

인간이 언어를 배우는 과정을 생각해보세요:

```
아이: "나는 학교에 갔어요" (올바름)
아이: "나는 학교에 갈래요" (맥락에 맞지 않음)
어른: "그건 '가고 싶어요'라고 해야 해"
아이: (다음에는 올바르게) "나는 학교에 가고 싶어요"
```

**GPT도 마찬가지로 "실수"에서 배웁니다.**

```
모델: "ROMEO:" 다음에 "xyz"를 예측 (틀림)
정답: "ROMEO:" 다음에는 "But"이 정답
손실함수: 예측과 정답의 차이를 수치화
역전파: 실수를 모든 가중치에 전파하여 수정
결과: 다음에는 "But"을 더 잘 예측
```

이것이 바로 **딥러닝 학습의 핵심 원리**입니다.

---

## 1. Cross-Entropy Loss: 예측 오차의 과학적 측정

### 우리 구현에서의 손실 계산

[레포지토리](https://github.com/BanHun28/gpt2_study)의 `main.py`에서:

```python
def forward(self, idx, targets=None):
    # ... 모델 forward 과정 ...
    
    if targets is not None:
        # 훈련 모드: 모든 위치에서 예측 계산
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Cross-Entropy 손실 계산 (2D로 변형하여 계산)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                              targets.view(-1), ignore_index=-1)
    else:
        # 추론 모드: 마지막 위치만 예측 (메모리 효율성)
        logits = self.lm_head(x[:, [-1], :])
        loss = None
        
    return logits, loss
```

### Cross-Entropy의 직관적 이해

#### 단순한 예시: 3개 단어만 있는 세상

```python
# 어휘: ["the", "cat", "dog"]
vocab_size = 3

# 모델의 예측 (logits)
logits = torch.tensor([2.0, 1.0, 0.5])  # "the"에 높은 점수

# 확률로 변환
probabilities = F.softmax(logits, dim=-1)
print(probabilities)  # [0.659, 0.242, 0.099]

# 정답: "cat" (index 1)
target = 1

# Cross-Entropy Loss 계산
loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
print(f"손실: {loss:.3f}")  # 1.424
```

#### 확률과 손실의 관계

```python
# 모델이 정답에 부여한 확률에 따른 손실 변화

correct_prob = 0.90  # 90% 확률로 정답 예측
loss_90 = -math.log(0.90)  # 0.105 (낮은 손실)

correct_prob = 0.50  # 50% 확률로 정답 예측  
loss_50 = -math.log(0.50)  # 0.693 (중간 손실)

correct_prob = 0.10  # 10% 확률로 정답 예측
loss_10 = -math.log(0.10)  # 2.303 (높은 손실)

# 핵심: 정답 확률이 높을수록 손실이 낮아짐
# 정답 확률이 낮을수록 손실이 급격히 증가
```

### 실제 문장에서의 손실 계산

```python
# 예시: "Romeo loves Juliet"
input_sequence = ["Romeo", "loves", "Juliet"]
target_sequence = ["loves", "Juliet", "<END>"]

# 각 위치에서의 예측과 손실
position_0: 
    입력: "Romeo"
    예측: ["loves": 0.7, "hates": 0.2, "is": 0.1]  
    정답: "loves"
    손실: -log(0.7) = 0.357

position_1:
    입력: "Romeo loves" 
    예측: ["Juliet": 0.8, "Mary": 0.1, "pizza": 0.1]
    정답: "Juliet"
    손실: -log(0.8) = 0.223

position_2:
    입력: "Romeo loves Juliet"
    예측: ["<END>": 0.9, "deeply": 0.08, "not": 0.02]
    정답: "<END>" 
    손실: -log(0.9) = 0.105

# 전체 손실: 평균 = (0.357 + 0.223 + 0.105) / 3 = 0.228
```

---

## 2. Backpropagation: 오차의 역방향 전파

### 그래디언트의 의미: 어디로 가야 할까?

#### 간단한 1차원 예시

```python
# 단순한 함수: y = x^2
# 목표: y를 최소화하고 싶음 (x = 0에서 최소)

x = 3.0  # 현재 위치
y = x**2  # 9.0 (높은 값)

# 그래디언트 계산: dy/dx = 2x
gradient = 2 * x  # 6.0

# 그래디언트의 의미:
# - 양수: x를 줄이면 y가 감소
# - 음수: x를 늘리면 y가 감소  
# - 크기: 변화의 급격함

# 가중치 업데이트
learning_rate = 0.1
x_new = x - learning_rate * gradient  # 3.0 - 0.1 * 6.0 = 2.4
y_new = x_new**2  # 5.76 (감소!)
```

### Transformer에서의 그래디언트 흐름

#### 순전파 과정 (Forward Pass)

```python
# 순전파: 입력 → 출력
input_tokens → embedding → attention → mlp → ... → logits → loss

# 각 단계에서 중간 결과 저장 (역전파에서 필요)
```

#### 역전파 과정 (Backward Pass)

```python
# 역전파: 손실 → 입력 방향으로 그래디언트 계산
loss.backward()  # PyTorch가 자동으로 모든 그래디언트 계산

# 내부적으로 일어나는 과정:
1. ∂loss/∂logits 계산
2. ∂loss/∂(final_layer_norm) 계산 
3. ∂loss/∂(block_12_output) 계산
4. ∂loss/∂(block_11_output) 계산
   ...
5. ∂loss/∂(embedding) 계산

# 각 파라미터의 그래디언트:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.6f}")
```

### 그래디언트 소실과 폭발 문제

#### 깊은 네트워크의 문제점

```python
# 12개 레이어를 거치는 그래디언트

# 그래디언트 소실 (Vanishing Gradient)
# 각 레이어에서 그래디언트가 0.9배씩 감소한다면:
final_gradient = initial_gradient * (0.9 ** 12)  # 0.28배로 감소

# 그래디언트 폭발 (Exploding Gradient)  
# 각 레이어에서 그래디언트가 1.1배씩 증가한다면:
final_gradient = initial_gradient * (1.1 ** 12)  # 3.14배로 증가
```

#### Transformer의 해결책들

```python
# 1. Residual Connection: 그래디언트 고속도로
def block_forward(x):
    return x + self.attn(self.ln_1(x))  # +x가 핵심
    
# 역전파 시: ∂output/∂x = 1 + ∂attn/∂x  
# 최소한 1의 그래디언트는 항상 보장!

# 2. Layer Normalization: 안정적인 스케일 유지
# 각 레이어 입력을 정규화하여 극단적 값 방지

# 3. Gradient Clipping: 폭발 방지
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 3. AdamW Optimizer: 똑똑한 학습 알고리즘

### SGD vs Adam vs AdamW 비교

#### SGD (Stochastic Gradient Descent)

```python
# 가장 기본적인 방법
for param in model.parameters():
    param.data -= learning_rate * param.grad

# 문제점:
# 1. 모든 파라미터에 같은 학습률 적용
# 2. 과거 정보 무시
# 3. 노이즈가 많은 그래디언트에 민감
```

#### Adam (Adaptive Moment Estimation)

```python
# 각 파라미터별로 적응적 학습률
# 1차 모멘텀 (과거 그래디언트의 지수 이동 평균)
m_t = beta1 * m_{t-1} + (1 - beta1) * grad

# 2차 모멘텀 (과거 그래디언트 제곱의 지수 이동 평균)  
v_t = beta2 * v_{t-1} + (1 - beta2) * grad**2

# 편향 보정
m_hat = m_t / (1 - beta1**t)
v_hat = v_t / (1 - beta2**t)

# 가중치 업데이트
param -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

#### AdamW (Adam with Weight Decay)

```python
# Adam + Weight Decay의 올바른 구현
# main.py에서 사용하는 방법

optimizer = torch.optim.AdamW(
    optim_groups,           # 파라미터 그룹
    lr=learning_rate,       # 학습률 
    betas=(beta1, beta2),   # 모멘텀 계수
    weight_decay=weight_decay  # 가중치 감쇠
)

# 내부 동작:
# 1. Adam 업데이트 수행
# 2. Weight Decay 별도 적용 (Adam의 모멘텀에 영향받지 않음)
param = param - learning_rate * weight_decay * param
```

### 우리 구현에서의 옵티마이저 설정

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    """AdamW 옵티마이저 설정 함수"""
    
    # 가중치 감쇠를 적용할 파라미터와 적용하지 않을 파라미터를 분리
    decay = set()      # 감쇠 적용할 파라미터들
    no_decay = set()   # 감쇠 적용하지 않을 파라미터들
    
    # 감쇠 적용 기준
    whitelist_weight_modules = (torch.nn.Linear, )        # 선형 레이어
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)  # 정규화, 임베딩
    
    for mn, m in self.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            
            if pn.endswith('bias'):
                # 편향은 감쇠 적용 안 함
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # 선형 레이어 가중치는 감쇠 적용
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # LayerNorm, Embedding 가중치는 감쇠 적용 안 함
                no_decay.add(fpn)
```

#### 왜 이렇게 나눌까?

```python
# Weight Decay를 적용하는 이유와 예외

적용할 파라미터 (선형 레이어 가중치):
- 크기가 커질수록 과적합 위험 증가
- 정규화 효과로 일반화 성능 향상
- 예: attention의 Q,K,V 가중치, MLP 가중치

적용하지 않을 파라미터:
1. 편향 (bias): 크기가 작고 과적합에 영향 적음
2. LayerNorm: 정규화 파라미터는 학습된 스케일 유지 필요  
3. Embedding: 단어별 고유 특성 보존 필요
```

---

## 4. Learning Rate Scheduling: 학습 속도의 예술

### Warm-up + Constant 스케줄

```python
# main.py의 학습 루프에서
for iter_num in range(max_iters):
    # 학습률 스케줄링 - 초기에는 낮은 학습률로 시작 (워밍업)
    lr = learning_rate  # 기본 학습률
    if iter_num < 100:  # 처음 100 이터에서는 워밍업 적용
        lr = learning_rate * iter_num / 100  # 점진적으로 학습률 증가
    
    # 모든 옵티마이저 그룹에 새 학습률 적용
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

#### 왜 Warm-up이 필요할까?

```python
# 초기 상태의 문제점

초기 상태:
- 가중치: 랜덤하게 초기화됨
- 그래디언트: 매우 클 수 있음  
- 높은 학습률 + 큰 그래디언트 = 불안정한 학습

Warm-up 효과:
이터 1:   lr = 0.001 * 1/100   = 0.00001 (매우 작음)
이터 50:  lr = 0.001 * 50/100  = 0.0005  (점진적 증가)
이터 100: lr = 0.001 * 100/100 = 0.001   (목표 학습률)

결과: 안정적인 학습 시작 → 빠른 수렴
```

### 다양한 학습률 스케줄 비교

```python
def compare_lr_schedules():
    """다양한 학습률 스케줄링 방법 비교"""
    
    max_iters = 1000
    base_lr = 0.001
    
    schedules = {}
    
    # 1. Constant (고정)
    schedules['constant'] = [base_lr] * max_iters
    
    # 2. Linear Decay (선형 감소)
    schedules['linear_decay'] = [base_lr * (1 - i/max_iters) for i in range(max_iters)]
    
    # 3. Cosine Annealing (코사인 감소)
    schedules['cosine'] = [base_lr * 0.5 * (1 + math.cos(math.pi * i / max_iters)) 
                          for i in range(max_iters)]
    
    # 4. Step Decay (단계적 감소)
    schedules['step'] = [base_lr * (0.5 ** (i // 200)) for i in range(max_iters)]
    
    # 5. Warm-up + Cosine (우리가 사용할 수 있는 개선된 방법)
    warmup_iters = 100
    schedules['warmup_cosine'] = []
    for i in range(max_iters):
        if i < warmup_iters:
            lr = base_lr * i / warmup_iters
        else:
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * (i - warmup_iters) / (max_iters - warmup_iters)))
        schedules['warmup_cosine'].append(lr)
    
    # 시각화 (개념적)
    for name, schedule in schedules.items():
        print(f"{name:15} | 시작: {schedule[0]:.6f} | 중간: {schedule[500]:.6f} | 끝: {schedule[-1]:.6f}")

compare_lr_schedules()
```

---

## 5. 정규화 기법들: 과적합 방지의 다층 방어

### 1. Dropout: 랜덤한 뉴런 비활성화

```python
# main.py의 여러 위치에서 사용
self.attn_dropout = nn.Dropout(config.dropout)  # Attention에서
self.resid_dropout = nn.Dropout(config.dropout)  # Residual connection에서  
self.dropout = nn.Dropout(config.dropout)        # MLP에서

# 동작 원리
def dropout_example():
    x = torch.randn(3, 4)
    dropout = nn.Dropout(p=0.2)  # 20% 확률로 0으로 만듦
    
    # 훈련 모드
    dropout.train()
    output_train = dropout(x)
    print("훈련 모드:", output_train)
    # 일부 값이 0이 되고, 나머지는 1/(1-0.2) = 1.25배 증가
    
    # 평가 모드  
    dropout.eval()
    output_eval = dropout(x)
    print("평가 모드:", output_eval)
    # 모든 값이 그대로 (dropout 비활성화)
```

#### Dropout의 과적합 방지 원리

```python
# 앙상블 효과
훈련 step 1: 일부 뉴런만 활성화 → 서브네트워크 1 학습
훈련 step 2: 다른 뉴런들 활성화 → 서브네트워크 2 학습  
훈련 step 3: 또 다른 조합 → 서브네트워크 3 학습
...

추론 시: 모든 뉴런 활성화 → 여러 서브네트워크의 앙상블 효과

결과: 특정 뉴런에 과도하게 의존하지 않는 robust한 모델
```

### 2. Weight Decay: 가중치 크기 제한

```python
# L2 정규화의 효과
original_loss = cross_entropy_loss
regularized_loss = cross_entropy_loss + weight_decay * sum(param**2)

# 효과:
# - 가중치가 커지면 손실도 증가
# - 모델이 "단순한" 해를 선호하도록 유도
# - 작은 가중치들로 표현 가능한 패턴 학습
```

### 3. Layer Normalization: 내부 공변량 이동 방지

```python
# 각 레이어 입력의 분포 안정화
def layer_norm_regularization_effect():
    # LayerNorm 없이: 레이어를 거칠수록 분포가 변함
    x = torch.randn(100, 768)
    
    for i in range(12):  # 12개 레이어
        x = torch.matmul(x, torch.randn(768, 768) * 0.1)
        print(f"레이어 {i+1}: 평균={x.mean():.3f}, 표준편차={x.std():.3f}")
    
    # 문제: 레이어가 깊어질수록 분포가 극단적으로 변화
    
    # LayerNorm 있을 때: 각 레이어에서 분포 정규화
    x = torch.randn(100, 768)
    ln = nn.LayerNorm(768)
    
    for i in range(12):
        x = torch.matmul(x, torch.randn(768, 768) * 0.1)  
        x = ln(x)  # 정규화 적용
        print(f"레이어 {i+1}: 평균={x.mean():.3f}, 표준편차={x.std():.3f}")
    
    # 결과: 안정적인 분포 유지

layer_norm_regularization_effect()
```

---

## 6. 실제 학습 과정 분석: Loss Curve 해석

### 우리 구현의 학습 모니터링

```python
# main.py의 학습 루프에서
for iter_num in range(max_iters):
    # 주기적으로 모델 성능 평가
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss(model, config, batch_size, train_data, val_data, device)
        elapsed = time.time() - start_time
        print(f"Iter {iter_num:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Time: {elapsed:.1f}s")
```

### 이상적인 학습 곡선

```python
# 건강한 학습의 특징

이터레이션    | 훈련 손실 | 검증 손실 | 상태
0           | 4.200    | 4.205    | 초기 (높은 손실)
300         | 3.800    | 3.820    | 빠른 감소
600         | 3.400    | 3.450    | 계속 감소  
900         | 3.100    | 3.180    | 점진적 감소
1200        | 2.900    | 3.000    | 수렴 시작
1500        | 2.800    | 2.950    | 안정적 수렴
```

### 문제가 있는 학습 패턴들

#### 1. 과적합 (Overfitting)

```python
이터레이션    | 훈련 손실 | 검증 손실 | 상태
0           | 4.200    | 4.205    | 정상
500         | 3.000    | 3.100    | 정상
1000        | 2.500    | 2.800    | 약간 차이
1500        | 2.000    | 3.200    | 과적합 시작!
2000        | 1.500    | 3.800    | 심각한 과적합

해결책:
- Dropout 증가 (0.1 → 0.2)
- Weight Decay 증가
- 조기 종료 (Early Stopping)
- 더 많은 데이터
```

#### 2. 학습률 과대 (Learning Rate Too High)

```python
이터레이션    | 훈련 손실 | 상태
0           | 4.200    | 정상
100         | 5.800    | 손실 증가!
200         | 7.200    | 발산!
300         | NaN      | 완전 실패

해결책:
- 학습률 감소 (0.001 → 0.0003)
- Warm-up 기간 증가
- Gradient Clipping 강화
```

#### 3. 학습률 과소 (Learning Rate Too Small)

```python
이터레이션    | 훈련 손실 | 상태  
0           | 4.200    | 정상
1000        | 4.150    | 매우 느린 감소
2000        | 4.100    | 여전히 느림
3000        | 4.050    | 수렴 너무 느림

해결책:
- 학습률 증가
- Adam의 beta 값 조정
- 더 긴 학습 시간
```

---

## 7. 실전 학습 최적화 기법들

### 1. Gradient Accumulation

```python
# GPU 메모리가 부족할 때 배치 크기를 효과적으로 늘리는 방법

accumulation_steps = 4  # 4번 누적 후 업데이트
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    # 작은 배치로 forward
    micro_batch_x, micro_batch_y = get_micro_batch()
    
    with torch.cuda.amp.autocast():
        logits, loss = model(micro_batch_x, micro_batch_y)
        loss = loss / accumulation_steps  # 평균화
    
    # 그래디언트 누적 (업데이트 안 함)
    scaler.scale(loss).backward()

# 누적된 그래디언트로 한 번에 업데이트
scaler.step(optimizer)
scaler.update()
```

### 2. Mixed Precision Training

```python
# main.py에서 사용하는 혼합 정밀도 학습

# GradScaler로 수치 안정성 확보
scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

# 훈련 루프에서
with torch.cuda.amp.autocast(enabled=(device=='cuda')):
    logits, loss = model(xb, yb)  # forward는 float16

scaler.scale(loss).backward()     # backward는 float32로 스케일링
scaler.step(optimizer)            # 안전한 업데이트
scaler.update()                   # 다음 스텝을 위한 스케일 조정

# 효과:
# - 메모리 사용량 ~50% 감소
# - 학습 속도 ~1.5배 향상  
# - 수치 안정성 유지
```

### 3. Checkpointing과 Resume

```python
def save_checkpoint(model, optimizer, iter_num, loss):
    """체크포인트 저장"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'loss': loss,
        'config': model.config,
    }
    torch.save(checkpoint, f'checkpoint_iter_{iter_num}.pt')

def load_checkpoint(model, optimizer, checkpoint_path):
    """체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter_num']
    return start_iter

# 사용 예시
if os.path.exists('latest_checkpoint.pt'):
    start_iter = load_checkpoint(model, optimizer, 'latest_checkpoint.pt')
    print(f"체크포인트에서 재개: iteration {start_iter}")
else:
    start_iter = 0
```

---

## 8. 실습: 학습 과정 심층 분석

### 실습 1: Loss Curve 시각화

```python
def plot_training_curves():
    """학습 곡선 시각화 및 분석"""
    
    # 학습 중 손실 기록
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # 간단한 학습 루프 (기록용)
    for iter_num in range(1000):
        # 학습률 기록
        current_lr = learning_rate if iter_num >= 100 else learning_rate * iter_num / 100
        learning_rates.append(current_lr)
        
        # 손실 기록 (실제로는 model.train() 호출)
        if iter_num % 50 == 0:
            # 여기서는 가상의 손실 생성 (실제로는 estimate_loss 사용)
            train_loss = 4.0 - 1.5 * (iter_num / 1000) + 0.1 * random.random()
            val_loss = train_loss + 0.1 + 0.05 * (iter_num / 1000)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Iter {iter_num:4d} | Train: {train_loss:.3f} | Val: {val_loss:.3f} | LR: {current_lr:.6f}")
    
    # 분석
    print("\n=== 학습 곡선 분석 ===")
    print(f"최종 훈련 손실: {train_losses[-1]:.3f}")
    print(f"최종 검증 손실: {val_losses[-1]:.3f}")
    print(f"과적합 정도: {val_losses[-1] - train_losses[-1]:.3f}")
    
    # 수렴 속도 분석
    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    reduction_rate = (initial_loss - final_loss) / initial_loss
    print(f"손실 감소율: {reduction_rate:.1%}")

plot_training_curves()
```

### 실습 2: 그래디언트 분석

```python
def analyze_gradients(model):
    """그래디언트 크기와 분포 분석"""
    
    # 가상의 손실로 역전파
    dummy_input = torch.randint(0, 1000, (2, 10))
    dummy_target = torch.randint(0, 1000, (2, 10))
    
    logits, loss = model(dummy_input, dummy_target)
    loss.backward()
    
    # 레이어별 그래디언트 분석
    print("=== 그래디언트 분석 ===")
    
    gradient_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms[name] = grad_norm
            
            # 레이어 유형별 분류
            if 'embedding' in name:
                layer_type = 'Embedding'
            elif 'attn' in name:
                layer_type = 'Attention'  
            elif 'mlp' in name:
                layer_type = 'MLP'
            elif 'ln' in name:
                layer_type = 'LayerNorm'
            else:
                layer_type = 'Other'
            
            print(f"{layer_type:10} | {name:30} | Grad Norm: {grad_norm:.6f}")
    
    # 전체 그래디언트 노름
    total_grad_norm = torch.sqrt(sum(norm**2 for norm in gradient_norms.values()))
    print(f"\n전체 그래디언트 노름: {total_grad_norm:.6f}")
    
    # 그래디언트 클리핑 필요성 판단
    if total_grad_norm > 1.0:
        print("⚠️  그래디언트 클리핑 권장")
    else:
        print("✅ 그래디언트 안정적")

# 사용
model = GPT(GPTConfig())
analyze_gradients(model)
```

### 실습 3: 하이퍼파라미터 실험

```python
def hyperparameter_experiment():
    """다양한 하이퍼파라미터 조합 실험"""
    
    configs = [
        # 기본 설정
        {"lr": 1e-3, "weight_decay": 0.1, "dropout": 0.1, "name": "baseline"},
        
        # 학습률 변화
        {"lr": 5e-4, "weight_decay": 0.1, "dropout": 0.1, "name": "low_lr"},
        {"lr": 2e-3, "weight_decay": 0.1, "dropout": 0.1, "name": "high_lr"},
        
        # 정규화 강도 변화
        {"lr": 1e-3, "weight_decay": 0.01, "dropout": 0.05, "name": "low_reg"},
        {"lr": 1e-3, "weight_decay": 0.2, "dropout": 0.2, "name": "high_reg"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n=== {config['name']} 실험 ===")
        
        # 모델 초기화 (동일한 시드로)
        torch.manual_seed(42)
        model = GPT(GPTConfig())
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # 간단한 학습 (실제로는 더 긴 학습 필요)
        model.train()
        final_loss = None
        
        for step in range(100):  # 짧은 학습
            # 가상의 배치 데이터
            x = torch.randint(0, 1000, (4, 20))
            y = torch.randint(0, 1000, (4, 20))
            
            # 순전파
            logits, loss = model(x, y)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 가중치 업데이트
            optimizer.step()
            
            if step % 25 == 0:
                print(f"Step {step:3d}: Loss = {loss.item():.4f}")
            
            final_loss = loss.item()
        
        results.append({
            'name': config['name'],
            'final_loss': final_loss,
            'config': config
        })
    
    # 결과 비교
    print("\n=== 실험 결과 요약 ===")
    results.sort(key=lambda x: x['final_loss'])
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['name']:12} | 최종 손실: {result['final_loss']:.4f}")

hyperparameter_experiment()
```

---

## 마무리: 학습의 과학을 마스터했습니다

### 오늘 배운 핵심 내용

1. **Cross-Entropy Loss**: 예측 오차를 정확히 측정하는 과학적 방법
2. **Backpropagation**: 오차를 역방향으로 전파하여 모든 가중치 개선
3. **AdamW Optimizer**: 각 파라미터별 적응적 학습률과 weight decay
4. **Learning Rate Scheduling**: Warm-up으로 안정적 시작, 적절한 수렴 속도
5. **정규화 기법들**: Dropout, Weight Decay, LayerNorm의 삼중 방어

### 학습의 전체 그림

```
1. 순전파: 입력 → 예측
2. 손실 계산: 예측 vs 정답의 차이 측정
3. 역전파: 모든 파라미터의 개선 방향 계산
4. 가중치 업데이트: 똑똑한 옵티마이저로 개선
5. 정규화: 과적합 방지로 일반화 능력 확보
6. 반복: 수천 번 반복하여 점진적 개선

결과: 랜덤한 가중치 → 셰익스피어 스타일 텍스트 생성 능력
```

### 학습 성공의 핵심 지표들

```
✅ 건강한 학습의 신호:
- 훈련/검증 손실이 함께 감소
- 그래디언트 노름이 안정적 (0.1~2.0 범위)
- 학습률이 적절 (손실이 급격히 감소하지만 발산하지 않음)
- 검증 손실이 훈련 손실보다 약간만 높음

❌ 문제가 있는 학습의 신호:
- 손실이 NaN이나 무한대로 발산
- 검증 손실이 훈련 손실보다 훨씬 높음 (과적합)
- 그래디언트가 너무 크거나 0에 가까움
- 수백 이터레이션 동안 손실 변화 없음
```

### 다음 편 예고: 창작하는 AI의 비밀

다음 편에서는 학습된 모델이 어떻게 **새로운 텍스트를 창작**하는지 배웁니다:

- **Autoregressive Generation**: 한 단어씩 순차적으로 생성하는 원리
- **Temperature Sampling**: 창의성과 일관성의 균형 조절
- **Top-k & Top-p Sampling**: 현실적이면서도 다양한 텍스트 생성
- **Beam Search**: 더 나은 품질의 텍스트를 찾는 탐색 방법
- **생성 품질 평가**: 좋은 텍스트와 나쁜 텍스트를 구분하는 방법

**미리 생각해볼 질문:**
모델이 "ROMEO:"라는 입력에서 시작해서 어떻게 완전한 대사를 만들어낼까요? 각 단어를 선택할 때 어떤 전략을 사용해야 할까요?

### 실습 과제

다음 편까지 해볼 과제:

1. **학습 모니터링**: 실제 학습 과정에서 loss curve 관찰 및 분석
2. **하이퍼파라미터 실험**: learning rate, dropout 등을 바꿔가며 효과 확인
3. **그래디언트 분석**: 각 레이어의 그래디언트 크기 분포 관찰

이제 학습의 과학을 완전히 이해했으니, 창작하는 AI의 마지막 비밀을 파헤쳐봅시다! 🎨


