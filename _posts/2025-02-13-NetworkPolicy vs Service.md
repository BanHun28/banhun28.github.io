---
title: NetworkPolicy vs Service
data: 2025-02-13 12:50:42
categories: [kubernetes, basic]
tags: [kubernetes, k8s, network, network policy, service, policy]     # TAG names should always be lowercase
---

## 1. 기본 개념 정리

### Network Policy
- Kubernetes 클러스터 내에서 **Pod 간 트래픽을 제어**하기 위해 사용되는 리소스.
- **Ingress(들어오는 트래픽)**와 **Egress(나가는 트래픽)**를 세밀하게 제어.
- Pod Selector와 레이블을 활용해 특정 Pod 간 통신 허용/차단 가능.

### Service
- Kubernetes 클러스터 내에서 **Pod에 대한 안정적인 네트워크 접근을 제공**하는 리소스.
- Pod의 IP가 변경되더라도 Service를 통해 지속적인 연결 보장.
- 주요 타입:
  - ClusterIP: 클러스터 내부에서만 접근 가능.
  - NodePort: 외부에서 노드의 특정 포트를 통해 접근 가능.
  - LoadBalancer: 클라우드 제공사의 로드 밸런서를 활용해 외부 접근 가능.

---

## 2. Network Policy와 Service의 차이점과 공통점

| **구분**        | **Network Policy**                                   | **Service**                                        |
| --------------- | ---------------------------------------------------- | -------------------------------------------------- |
| **역할**        | Pod 간 트래픽 제어 (보안)                            | Pod에 대한 네트워크 접근 제공                      |
| **주요 기능**   | 특정 Pod에 대한 Ingress/Egress 트래픽을 허용/차단    | Pod의 동적 IP를 안정적으로 관리하고 외부 접근 제공 |
| **트래픽 방향** | Ingress(들어오는 트래픽), Egress(나가는 트래픽) 제어 | 주로 Ingress(들어오는 트래픽)                      |
| **적용 범위**   | Pod Selector 기반으로 특정 Pod에 적용                | Service에 연결된 모든 Pod에 적용                   |
| **보안 관점**   | 네트워크 레벨에서 세부적인 보안 제어 가능            | 보안보다는 네트워크 접근성 제공에 초점             |
| **외부 솔루션** | AWS Security Group, VPC 네트워크 ACL과 유사          | AWS ELB(Application/Network Load Balancer)와 유사  |

---

## 3. Network Policy의 세부 사항

### Network Policy가 없는 경우
- 기본적으로 모든 Pod 간 통신 허용.

### Network Policy가 있는 경우
- 명시적으로 허용된 트래픽만 통과, 나머지는 암묵적으로 차단.

### Ingress와 Egress
- **Ingress**: "어떤 트래픽이 들어올 수 있는가?" 제어.
- **Egress**: "어떤 트래픽이 나갈 수 있는가?" 제어.

---

## 4. Network Policy YAML 예제

### 특정 Pod으로 들어오는 트래픽(Ingress) 제어
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-web-to-db
  namespace: default
spec:
  podSelector:
    matchLabels:
      app: db
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: web
      ports:
        - protocol: TCP
          port: 3306
```
- **설명**: `app=db` Pod은 `app=web` Pod에서 오는 TCP 3306 포트 트래픽만 허용.

### 특정 Pod에서 나가는 트래픽(Egress) 제어
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-egress
  namespace: default
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Egress
  egress:
    - to:
        - ipBlock:
            cidr: 192.168.1.0/24
      ports:
        - protocol: TCP
          port: 443
```
- **설명**: `app=backend` Pod은 192.168.1.0/24 네트워크 대역으로 TCP 443 포트(HTTPS)만 나갈 수 있음.

---

## 5. AWS 솔루션과의 비교

### AWS Security Group
- **Ingress와 Egress 규칙**을 제공하며, Kubernetes Network Policy와 유사.
- Pod 수준이 아닌 **EC2 인스턴스 레벨**에서 적용.

### AWS ELB
- Kubernetes Service의 LoadBalancer 타입과 유사.
- 외부 트래픽을 클러스터 내부의 서비스로 전달.

### VPC Network ACL
- 서브넷 단위로 트래픽을 제어.
- Kubernetes Network Policy의 Pod Selector와는 적용 범위가 다름.

---

## 6. 요약
- **Network Policy**는 Pod 간의 트래픽을 세밀히 제어하여 보안을 강화.
- **Service**는 Pod에 대한 네트워크 접근성을 보장.
- Ingress와 Egress를 나누는 이유는 네트워크 트래픽을 양방향으로 관리하여 보안을 더욱 강화하기 위함.
- AWS와 비교하면, Network Policy는 Security Group이나 VPC Network ACL과 유사한 보안 제어 기능을 제공.

---

## 7. 활용 가이드
- **최소 권한 원칙(Principle of Least Privilege)**: Network Policy를 통해 필요한 통신만 허용.
- **Egress 제어로 데이터 유출 방지**: 외부 네트워크로 나가는 트래픽을 제한.
- **Service를 통해 안정적인 네트워크 제공**: Pod의 IP 변화에도 일관된 접근 보장.
