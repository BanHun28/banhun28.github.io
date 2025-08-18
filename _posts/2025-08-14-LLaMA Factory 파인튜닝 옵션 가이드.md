---
title: LLaMA Factory 파인튜닝 옵션 가이드
date: 2025-08-14 10:29:57 +0900
categories: [artificial intelligence, machine learning]
tags: [machine learning, deep learning, llm, finetuning, nlp, llamafactory, pytorch, transformers, huggingface, lora, qlora, tutorial, guide, ai]     # TAG names should always be lowercase
---

# LLaMA Factory 파인튜닝 옵션 가이드

## 📋 목차
1. [기본 설정 옵션](#-기본-설정-옵션)
2. [모델 관련 옵션](#-모델-관련-옵션)
3. [데이터셋 관련 옵션](#-데이터셋-관련-옵션)
4. [훈련 방법 옵션](#-훈련-방법-옵션)
5. [하이퍼파라미터 옵션](#️-하이퍼파라미터-옵션)
6. [양자화 옵션](#-양자화-옵션)
7. [LoRA 관련 옵션](#-lora-관련-옵션)
8. [분산 훈련 옵션](#-분산-훈련-옵션)
9. [로깅 및 모니터링 옵션](#-로깅-및-모니터링-옵션)
10. [최적화 관련 옵션](#-최적화-관련-옵션)

---

## 🔧 기본 설정 옵션

### stage (훈련 단계)
**설명**: 실행할 훈련 단계를 지정합니다.
```yaml
stage: sft  # 필수 옵션
```

**가능한 값**:
- `pt`: (Continued) Pre-training - 사전훈련 또는 지속적 사전훈련
- `sft`: Supervised Fine-Tuning - 지도 파인튜닝
- `rm`: Reward Modeling - 보상 모델 훈련 
- `ppo`: PPO Training - 강화학습 PPO 훈련
- `dpo`: DPO Training - Direct Preference Optimization
- `kto`: KTO Training - Knowledge Transfer from Oracle
- `orpo`: ORPO Training - Odds Ratio Preference Optimization

**실무 권장사항**:
- 대부분의 커스텀 작업: `sft`
- 강화학습 기반 정렬: `dpo` → `ppo` 순서로 진행

### do_train
**설명**: 훈련 모드 활성화 여부
```yaml
do_train: true  # 기본값: false
```

### do_eval
**설명**: 평가 모드 활성화 여부
```yaml
do_eval: true  # 기본값: false
```

### do_predict
**설명**: 예측/추론 모드 활성화 여부
```yaml
do_predict: true  # 기본값: false
```

---

## 🤖 모델 관련 옵션

### model_name_or_path
**설명**: 사용할 모델의 경로 또는 Hugging Face 모델 이름
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
# 또는 로컬 경로
model_name_or_path: /path/to/local/model
```

**지원 모델 예시**:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` 
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-2-9b-it`

### model_revision
**설명**: 특정 모델 버전/리비전 지정
```yaml
model_revision: main  # 기본값: main
```

### quantization_bit
**설명**: 양자화 비트 수 설정
```yaml
quantization_bit: 4  # 4, 8, 또는 None
```

### template
**설명**: 모델에 맞는 대화 템플릿 지정
```yaml
template: llama3  # 모델에 따라 필수
```

**주요 템플릿**:
- `llama3`: Llama 3 모델용
- `qwen`: Qwen 모델용
- `chatglm3`: ChatGLM 모델용
- `phi`: Phi 모델용
- `gemma`: Gemma 모델용

---

## 📊 데이터셋 관련 옵션

### dataset
**설명**: 사용할 데이터셋 이름(들)
```yaml
dataset: alpaca_en_demo  # 단일 데이터셋
# 또는
dataset: alpaca_en_demo,alpaca_zh_demo  # 다중 데이터셋
```

### dataset_dir
**설명**: 데이터셋 디렉토리 경로
```yaml
dataset_dir: data  # 기본값: data
```

### cutoff_len
**설명**: 최대 시퀀스 길이
```yaml
cutoff_len: 1024  # 기본값: 1024
```

**실무 가이드**:
- 일반적인 대화: 1024-2048
- 긴 문서 요약: 4096-8192
- 코드 생성: 2048-4096

### max_samples
**설명**: 사용할 최대 샘플 수
```yaml
max_samples: 100000  # 전체 데이터셋 크기 제한
```

### preprocessing_num_workers
**설명**: 데이터 전처리 워커 수
```yaml
preprocessing_num_workers: 16  # CPU 코어 수에 맞게 조정
```

### streaming
**설명**: 스트리밍 모드 활성화 (대용량 데이터셋)
```yaml
streaming: false  # 기본값: false
```

---

## 🎯 훈련 방법 옵션

### finetuning_type
**설명**: 파인튜닝 방법 선택
```yaml
finetuning_type: lora  # 필수 옵션
```

**가능한 값**:
- `lora`: LoRA (Low-Rank Adaptation)
- `qlora`: QLoRA (Quantized LoRA)
- `full`: Full-parameter fine-tuning
- `freeze`: Freeze tuning

**선택 가이드**:
```yaml
# 메모리 제약이 심한 경우
finetuning_type: qlora
quantization_bit: 4

# 성능 우선인 경우
finetuning_type: lora

# 최고 성능이 필요한 경우 (대용량 메모리 필요)
finetuning_type: full
```

---

## ⚙️ 하이퍼파라미터 옵션

### 학습률 관련

#### learning_rate
**설명**: 기본 학습률
```yaml
learning_rate: 5e-5  # 기본값: 5e-5
```

**권장 값**:
- LoRA: `5e-4` ~ `1e-4`
- Full fine-tuning: `5e-6` ~ `5e-5`
- QLoRA: `1e-4` ~ `2e-4`

#### lr_scheduler_type
**설명**: 학습률 스케줄러 타입
```yaml
lr_scheduler_type: cosine  # 기본값: linear
```

**옵션**:
- `linear`: 선형 감소
- `cosine`: 코사인 스케줄링 (권장)
- `polynomial`: 다항식 감소
- `constant`: 상수 유지

#### warmup_steps
**설명**: 워밍업 스텝 수
```yaml
warmup_steps: 100  # 기본값: 0
```

### 배치 관련

#### per_device_train_batch_size
**설명**: 디바이스당 훈련 배치 크기
```yaml
per_device_train_batch_size: 8  # 기본값: 8
```

**메모리별 권장값**:
- 8GB GPU: 1-2
- 16GB GPU: 2-4
- 24GB GPU: 4-8
- 40GB+ GPU: 8-16

#### gradient_accumulation_steps
**설명**: 그래디언트 누적 스텝
```yaml
gradient_accumulation_steps: 8  # 기본값: 8
```

**계산 공식**:
```
실제 배치 크기 = per_device_train_batch_size × gradient_accumulation_steps × GPU 수
```

#### dataloader_num_workers
**설명**: 데이터 로더 워커 수
```yaml
dataloader_num_workers: 4  # 기본값: 4
```

### 에포크 및 스텝

#### num_train_epochs
**설명**: 훈련 에포크 수
```yaml
num_train_epochs: 3.0  # 기본값: 3.0
```

#### max_steps
**설명**: 최대 훈련 스텝 (에포크 대신 사용)
```yaml
max_steps: 1000  # -1: 무제한
```

---

## 🔧 양자화 옵션

### 기본 양자화

#### quantization_bit
```yaml
quantization_bit: 4  # None, 4, 8
```

#### double_quantization
**설명**: 이중 양자화 활성화 (메모리 추가 절약)
```yaml
double_quantization: true  # 기본값: true
```

#### quant_type
**설명**: 양자화 타입
```yaml
quant_type: nf4  # nf4, fp4
```

### 고급 양자화 방법

#### GPTQ 양자화
```yaml
# GPTQ 설정 예시
gptq_bits: 4
gptq_group_size: 128
gptq_desc_act: false
```

#### AWQ 양자화
```yaml
# AWQ 설정 예시  
awq_bits: 4
awq_group_size: 128
```

---

## 🔗 LoRA 관련 옵션

### 기본 LoRA 설정

#### lora_rank
**설명**: LoRA 어댑터의 랭크 (차원)
```yaml
lora_rank: 8  # 기본값: 8
```

**가이드라인**:
- 작은 모델 (7B 이하): 4-8
- 중간 모델 (7B-30B): 8-16  
- 큰 모델 (30B+): 16-32

#### lora_alpha
**설명**: LoRA 스케일링 파라미터
```yaml
lora_alpha: 16  # 일반적으로 rank의 2배
```

#### lora_dropout
**설명**: LoRA 드롭아웃 비율
```yaml
lora_dropout: 0.1  # 기본값: 0.0
```

#### lora_target
**설명**: LoRA를 적용할 타겟 모듈
```yaml
lora_target: all  # 기본값: all
# 또는 특정 모듈 지정
lora_target: q_proj,v_proj,k_proj,o_proj
```

**모듈 옵션**:
- `all`: 모든 선형 레이어
- `q_proj,v_proj`: Attention의 Q, V 프로젝션
- `q_proj,v_proj,k_proj,o_proj`: 전체 Attention
- `gate_proj,up_proj,down_proj`: MLP 레이어

### LoRA 고급 옵션

#### use_rslora
**설명**: RS-LoRA 사용 여부
```yaml
use_rslora: false  # 기본값: false
```

#### loraplus_lr_ratio
**설명**: LoRA+ 학습률 비율
```yaml
loraplus_lr_ratio: 16.0  # LoRA+ 사용 시
```

#### pissa_init
**설명**: PiSSA 초기화 사용
```yaml
pissa_init: false  # 기본값: false  
```

---

## 🌐 분산 훈련 옵션

### 멀티 GPU 설정

#### ddp_backend
**설명**: 분산 백엔드
```yaml
ddp_backend: nccl  # nccl, gloo, mpi
```

#### ddp_timeout
**설명**: DDP 타임아웃 (초)
```yaml
ddp_timeout: 1800  # 기본값: 1800
```

#### fsdp
**설명**: FSDP (Fully Sharded Data Parallel) 설정
```yaml
fsdp: full_shard  # "", "full_shard", "shard_grad_op"
```

### DeepSpeed 설정

#### deepspeed
**설명**: DeepSpeed 설정 파일
```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

**ZeRO 단계**:
- ZeRO-1: 옵티마이저 상태 분할
- ZeRO-2: + 그래디언트 분할  
- ZeRO-3: + 모델 파라미터 분할

---

## 📊 로깅 및 모니터링 옵션

### 기본 로깅

#### output_dir
**설명**: 출력 디렉토리
```yaml
output_dir: saves/llama3-8b/lora/test
```

#### logging_steps
**설명**: 로깅 주기 (스텝)
```yaml
logging_steps: 10  # 기본값: 500
```

#### save_steps
**설명**: 체크포인트 저장 주기
```yaml
save_steps: 500  # 기본값: 500
```

#### eval_steps
**설명**: 평가 주기
```yaml
eval_steps: 500  # 기본값: 500
```

### 외부 모니터링 도구

#### report_to
**설명**: 로깅 백엔드
```yaml
report_to: none  # none, wandb, tensorboard
```

#### run_name
**설명**: 실험 이름
```yaml
run_name: llama3-sft-experiment
```

### TensorBoard
```yaml
report_to: tensorboard
logging_dir: ./logs
```

### Weights & Biases
```yaml
report_to: wandb
run_name: my_experiment
```

### SwanLab (권장)
```yaml
use_swanlab: true
swanlab_project: llamafactory
swanlab_run_name: test_run
swanlab_workspace: your_workspace  
swanlab_api_key: your_api_key
```

---

## ⚡ 최적화 관련 옵션

### 옵티마이저

#### optim
**설명**: 옵티마이저 타입
```yaml
optim: adamw_torch  # 기본값: adamw_torch
```

**옵션**:
- `adamw_torch`: AdamW (PyTorch)
- `adamw_hf`: AdamW (HuggingFace)
- `sgd`: SGD
- `adam_mini`: Adam-mini (메모리 효율적)

#### adam_beta1, adam_beta2
**설명**: Adam 베타 파라미터
```yaml
adam_beta1: 0.9   # 기본값: 0.9
adam_beta2: 0.999 # 기본값: 0.999
```

#### weight_decay
**설명**: 가중치 감쇠
```yaml
weight_decay: 0.01  # 기본값: 0.0
```

### 그래디언트 관련

#### max_grad_norm
**설명**: 그래디언트 클리핑
```yaml
max_grad_norm: 1.0  # 기본값: 1.0
```

#### gradient_checkpointing
**설명**: 그래디언트 체크포인팅 (메모리 절약)
```yaml
gradient_checkpointing: true  # 기본값: false
```

### 정밀도

#### fp16
**설명**: 16-bit 부동소수점 사용
```yaml
fp16: true  # 기본값: false
```

#### bf16
**설명**: Brain float 16 사용 (Ampere GPU 이상)
```yaml
bf16: false  # 기본값: false
```

### Flash Attention

#### flash_attn
**설명**: Flash Attention 사용
```yaml
flash_attn: auto  # auto, true, false
```

**권장**: 메모리 효율성을 위해 `auto` 사용

---

## 📋 실무 설정 예시

### 🚀 빠른 프로토타이핑용 설정
```yaml
# 기본 설정
stage: sft
do_train: true
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
template: llama3

# 데이터셋
dataset: alpaca_en_demo
cutoff_len: 1024

# 훈련 방법
finetuning_type: qlora
quantization_bit: 4

# 하이퍼파라미터 (빠른 실험용)
learning_rate: 1e-4
num_train_epochs: 1.0
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# LoRA 설정
lora_rank: 8
lora_alpha: 16
lora_target: all

# 로깅
output_dir: saves/quick-test
logging_steps: 10
save_steps: 50
```

### 🎯 프로덕션용 설정
```yaml
# 기본 설정
stage: sft
do_train: true
do_eval: true
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
template: llama3

# 데이터셋
dataset: your_production_dataset
cutoff_len: 2048
max_samples: 50000
preprocessing_num_workers: 16

# 훈련 방법  
finetuning_type: lora

# 하이퍼파라미터 (안정성 중심)
learning_rate: 5e-5
lr_scheduler_type: cosine
warmup_steps: 100
num_train_epochs: 3.0
per_device_train_batch_size: 8
gradient_accumulation_steps: 2

# LoRA 설정
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target: all

# 최적화
optim: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0
gradient_checkpointing: true

# 정밀도
fp16: true
flash_attn: auto

# 로깅 및 저장
output_dir: saves/production-model
logging_steps: 50
save_steps: 500
eval_steps: 500
report_to: tensorboard

# 모니터링
use_swanlab: true
swanlab_project: production
swanlab_run_name: final-model
```

### 💰 저비용 GPU 환경용 설정
```yaml
# 기본 설정
stage: sft  
do_train: true
model_name_or_path: microsoft/Phi-3-mini-4k-instruct
template: phi

# 데이터셋 (작게)
dataset: alpaca_en_demo
cutoff_len: 512
max_samples: 1000

# 메모리 최적화
finetuning_type: qlora
quantization_bit: 4
double_quantization: true

# 작은 배치
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true

# LoRA (작게)
lora_rank: 4
lora_alpha: 8
lora_target: q_proj,v_proj

# 기타
fp16: true
output_dir: saves/budget-model
```

## 🔍 트러블슈팅 가이드

### 메모리 부족 시
1. `quantization_bit: 4` 사용
2. `gradient_checkpointing: true` 활성화
3. `per_device_train_batch_size` 줄이기
4. `gradient_accumulation_steps` 늘리기
5. `lora_rank` 줄이기

### 학습이 불안정할 때
1. `learning_rate` 줄이기
2. `warmup_steps` 늘리기
3. `lr_scheduler_type: cosine` 사용
4. `weight_decay` 추가
5. `lora_dropout` 추가

### 속도가 느릴 때
1. `flash_attn: auto` 사용
2. `preprocessing_num_workers` 늘리기
3. `dataloader_num_workers` 늘리기
4. 데이터셋 전처리 사용

이 가이드를 통해 LLaMA Factory의 모든 주요 옵션을 이해하고, 상황에 맞는 최적의 설정을 구성하실 수 있습니다.

