---
title: ServiceAccount vs User
data: 2025-02-13 12:24:06
categories: [kubernetes, authenticating]
tags: [kubernetes, k8s, serviceAccount, user, auth]     # TAG names should always be lowercase
---
## ServiceAccount와 User

###  비유
- **User (유저)**: 클러스터 외부에서 Kubernetes를 제어하는 '사람'입니다.  
  예: `kubectl` 명령어나 CI/CD 도구에서 사용.
- **ServiceAccount (서비스 어카운트)**: 클러스터 내부에서 동작하는 '로봇'입니다.  
  예: 파드가 Kubernetes API와 통신하거나 리소스를 조작할 때 사용.

### 공통점
1. **RBAC(Role-Based Access Control)을 통한 접근 제어**:
   - 유저와 서비스 어카운트 모두 역할(Role)을 기반으로 권한을 부여받음.
2. **토큰(Token)을 사용한 인증**:
   - 유저는 `kubeconfig` 파일에 토큰 저장.
   - 서비스 어카운트는 파드 실행 시 토큰이 자동으로 주입됨.

### 차이점

| **특징**           | **User (유저)**            | **ServiceAccount (서비스 어카운트)**     |
| ------------------ | -------------------------- | ---------------------------------------- |
| **사용 대상**      | 외부 클라이언트 (사람)     | 클러스터 내부 워크로드 (파드, 컨테이너)  |
| **권한 제어 범위** | 클러스터 외부에서 API 호출 | 클러스터 내부에서 리소스(API)와 상호작용 |
| **토큰 관리**      | kubeconfig에 저장          | 파드 내 자동 주입                        |
| **생성 방식**      | 수동 생성                  | 자동 생성 가능                           |

### AWS IAM Role과의 비교
- **공통점**:
  - 서비스 어카운트와 IAM Role은 모두 권한을 제어하며, 리소스와의 접근 관계를 정의.
  - RBAC (Kubernetes)와 IAM Policy (AWS)를 통해 권한 설정.
- **차이점**:
  - **ServiceAccount**는 Kubernetes 내부 리소스에 대한 권한을 제어.
  - **IAM Role**은 AWS 리소스(S3, DynamoDB 등)에 대한 권한을 제어.

### Kubernetes와 AWS 연계: IRSA
- Kubernetes의 **IAM Roles for Service Accounts (IRSA)**를 사용하면, Kubernetes 파드에서 AWS 리소스에 안전하게 접근 가능.
  ```yaml
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/my-iam-role
  ```

---

## 정리
- **Taint와 Toleration**: 노드의 조건(Taint)에 맞게 파드가 스케줄링되도록 Toleration 설정.
- **ServiceAccount와 User**: 
  - 유저는 외부에서 클러스터를 제어하는 사람.
  - 서비스 어카운트는 내부 워크로드(API 서버와 통신하는 파드 등)에 사용.
- **AWS IAM Role과 연계**: IRSA를 활용해 Kubernetes 파드에서 AWS 리소스 접근 권한을 안전하게 관리.
