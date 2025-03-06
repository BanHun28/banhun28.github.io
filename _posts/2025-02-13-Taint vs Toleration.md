---
title: Taint vs Toleration
date: 2025-02-13 12:43:09 +900
categories: [kubernetes, basic]
tags: [kubernetes, k8s, taint, toleration]     # TAG names should always be lowercase
image: assets/img/logos/kubernetes-logo.png
---

##  Taint와 Toleration

### 비유
- **Taint (테인트)**: 노드가 특정한 조건(드레스 코드)을 요구하는 '파티'라고 생각하세요. 조건에 맞지 않으면 입장할 수 없습니다.  
  예: "검은 드레스만 입은 사람만 입장 가능."
- **Toleration (톨러레이션)**: 파드가 "나는 이 조건을 맞출 수 있어요"라고 말하는 초대장 같은 개념입니다.  
  예: "검은 드레스를 입고 올게요."

### Taint와 Toleration의 동작 원리
1. **Taint**는 노드에 설정됩니다.
   ```bash
   kubectl taint nodes node1 key=value:NoSchedule
   ```
   - `node1` 노드에 `key=value`라는 Taint를 설정.
   - 톨러레이션이 없는 파드는 스케줄링되지 않음.

2. **Toleration**은 파드에 설정됩니다.
```yaml
tolerations:
- key: "key"
  operator: "Equal"
  value: "value"
  effect: "NoSchedule"
```
   - 해당 Taint를 가진 노드에 파드를 스케줄링할 수 있음.

### 기억하기 쉬운 포인트
- **Taint는 거절**: "이 조건에 맞지 않으면 들어오지 마!"
- **Toleration은 허락**: "나는 그 조건을 맞출 수 있어!"
- **Toleration은 Taint를 없애지 않음**: 단지 Taint를 "참을 수 있다"는 뜻입니다.

### 실무 팁
- **Taint의 사용 이유**: 특정 노드를 특정 워크로드(예: 중요한 파드, 테스트 파드)에만 사용하도록 제한.
- **Toleration만으로 충분하지 않을 때**: 노드를 특정 파드만 스케줄링하려면 **노드 셀렉터(nodeSelector)** 또는 **노드 어피니티(nodeAffinity)**를 추가로 사용.

### 예제: 실습 시나리오
1. **노드에 Taint 추가**:
```bash
kubectl taint nodes node1 environment=production:NoSchedule
```
2. **파드에 Toleration 설정**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  tolerations:
  - key: "environment"
    operator: "Equal"
    value: "production"
    effect: "NoSchedule"
  containers:
  - name: nginx
    image: nginx
```


