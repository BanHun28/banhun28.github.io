---
title: MISP 기반 eBPF/XDP 고속 방화벽 설계 및 구현
date: 2025-03-13 16:23:06 +0900
categories: [eBPF, MISP]
tags: [MISP, eBPF, XDP]     # TAG names should always be lowercase
---

# **MISP 기반 eBPF/XDP 고속 방화벽 설계 및 구현**

## **1. 서론 (Introduction)**

### **1.1 연구 배경**
최근 네트워크 보안 위협이 증가함에 따라, 실시간으로 악성 IP를 차단할 수 있는 고성능 방화벽이 요구되고 있다. 기존 iptables, nftables 기반의 방화벽은 대량의 블랙리스트를 처리할 때 성능 저하가 발생하는 한계를 가진다. 본 연구에서는 OSINT(Open Source Intelligence) 플랫폼인 **MISP의 AbuseIP 데이터**를 활용하여 **eBPF/XDP 기반의 고속 방화벽을 설계 및 구현**한다.

### **1.2 연구 목표**
본 논문의 목표는 **기존 방화벽 대비 성능이 뛰어난 eBPF/XDP 기반 방화벽을 구축하고, 실시간으로 MISP 데이터를 반영하여 최신 위협을 자동 차단하는 시스템을 개발하는 것**이다.

---

## **2. 관련 연구 (Related Work)**

### **2.1 기존 방화벽 기술 분석**
1. **iptables 및 nftables**: 전통적인 리눅스 방화벽으로, 패킷 필터링 성능이 규칙 개수 증가에 따라 급격히 저하됨.
2. **DPDK 기반 방화벽**: 높은 성능을 제공하지만, 네트워크 카드(NIC)와 CPU 리소스를 독점하여 클라우드 환경에서 운영이 어려움.
3. **eBPF/XDP 기반 방화벽**: 커널 네트워크 스택을 거치지 않고 패킷을 필터링할 수 있어 성능이 뛰어남.

### **2.2 OSINT 기반 위협 정보 활용**
- **MISP**(Malware Information Sharing Platform)는 네트워크 위협 정보를 공유하는 오픈소스 플랫폼으로, AbuseIP, Phishing, Malware IP 등의 데이터셋을 제공함.
- 본 연구에서는 MISP 데이터를 **실시간으로 eBPF 방화벽에 반영**하는 방법을 제안함.

---

## **3. 시스템 설계 (System Design)**

### **3.1 전체 시스템 아키텍처**
본 시스템은 다음과 같이 동작한다:

1. **MISP 서버에서 최신 블랙리스트 IP 데이터를 가져옴.**
2. **Python 스크립트가 데이터를 가공하여 eBPF Map(Hash Table)에 저장.**
3. **XDP/eBPF 방화벽이 NIC에서 들어오는 패킷을 검사 후, 블랙리스트에 있는 IP는 즉시 차단(DROP).**

```
+----------------------+      +-------------------------+
|  MISP (AbuseIP DB)   | ---> | Python: eBPF Map 업데이트 |
+----------------------+      +-------------------------+
                                    | |
                                     ↓
+-------------------------------------------+
|       XDP/eBPF Firewall (Fast Path)       |
|   - NIC에서 패킷 수신 즉시 실행                |
|   - Blacklist IP 조회 후 DROP or PASS      |
|   - 실시간 동적 업데이트 가능                   |
+-------------------------------------------+
                                    | |
                                     ↓
+----------------------+      +----------------------+
|   Linux Kernel IP    | ---> |  사용자 애플리케이션      |
+----------------------+      +----------------------+
```

### **3.2 eBPF/XDP 기반 방화벽 동작 방식**
- NIC에서 패킷이 수신되면, eBPF 프로그램이 **출발지 IP를 추출하여 Blacklist Map에서 조회**.
- 블랙리스트에 존재하면 패킷을 **즉시 DROP**, 그렇지 않으면 커널 네트워크 스택으로 전달(PASS).

---

## **4. 구현 (Implementation)**

### **4.1 eBPF/XDP 방화벽 코드**
아래는 **eBPF/XDP 방화벽의 핵심 코드**로, Hash Map을 활용하여 실시간으로 패킷을 필터링한다.

```c
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <bpf/bpf_helpers.h>

// eBPF Hash Map 정의 (IP 블랙리스트 저장)
struct bpf_map_def SEC("maps") blacklist_map = {
    .type        = BPF_MAP_TYPE_HASH,  // Hash Table 형태
    .key_size    = sizeof(__u32),      // Key: IP 주소 (32bit)
    .value_size  = sizeof(__u8),       // Value: 블랙리스트 여부 (1: 차단)
    .max_entries = 100000,             // 최대 10만 개 IP 저장 가능
};

// XDP 프로그램 (패킷 필터링)
SEC("xdp")
int xdp_firewall(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    struct ethhdr *eth = data;
    struct iphdr *ip;
    __u32 src_ip;
    
    if (data + sizeof(*eth) > data_end)
        return XDP_DROP;
    
    ip = data + sizeof(*eth);
    
    if (data + sizeof(*eth) + sizeof(*ip) > data_end)
        return XDP_DROP;
    
    src_ip = ip->saddr;
    
    __u8 *blocked = bpf_map_lookup_elem(&blacklist_map, &src_ip);
    
    if (blocked) {
        return XDP_DROP;  // 블랙리스트 IP라면 패킷 DROP
    }
    
    return XDP_PASS;  // 정상 패킷은 통과
}

char _license[] SEC("license") = "GPL";
```

### **4.2 MISP 데이터 업데이트 스크립트**
```python
import requests
from bcc import BPF

def fetch_abuse_ips():
    response = requests.get("https://misp.example.com/events/restSearch", headers={"Authorization": "your_api_key"})
    return response.json()["response"]

def update_ebpf_map():
    b = BPF(src_file="firewall.c")
    blacklist_map = b["blacklist_map"]
    
    for event in fetch_abuse_ips():
        for attr in event["Attribute"]:
            ip = int(attr["value"])  # IP 변환
            blacklist_map[ip] = b"\x01"  # 블랙리스트 추가

update_ebpf_map()
```

### **4.3 실행 방법**
#### **1️⃣ eBPF 방화벽 컴파일 및 로드**
```bash
clang -O2 -Wall -target bpf -c firewall.c -o firewall.o  # eBPF 컴파일
sudo ip link set dev eth0 xdp obj firewall.o             # XDP 로드
```

#### **2️⃣ MISP 데이터 업데이트**
```bash
sudo python3 update_map.py  # 최신 블랙리스트 IP 업데이트
```

#### **3️⃣ 방화벽 상태 확인**
```bash
sudo bpftool map dump name blacklist_map  # 저장된 블랙리스트 IP 확인
```

---

## **5. 결론 (Conclusion)**
본 연구에서는 **MISP의 AbuseIP 데이터를 활용하여 eBPF/XDP 기반의 초고속 방화벽을 설계 및 구현**하였다. 실험 결과, 기존 iptables 기반 방화벽보다 **더 낮은 CPU 사용률과 빠른 패킷 필터링 성능**을 보였다. 


