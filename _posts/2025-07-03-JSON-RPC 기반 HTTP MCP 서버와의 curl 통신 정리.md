---
title: JSON-RPC 기반 HTTP MCP 서버와의 curl 통신 정리
date: 2025-07-03 12:38:22 +0900
categories: [mcp, json-rpc]
tags: [json-rpc, http, mcp, curl]     # TAG names should always be lowercase
---


# JSON-RPC 기반 HTTP MCP 서버와의 `curl` 통신 정리

최근 OP.GG에서 제공하는 [MCP API](https://mcp-api.op.gg/mcp)를 활용하면서 JSON-RPC 기반 MCP 서버와의 통신 방식에 대해 정리할 기회가 생겼습니다. 이 글에서는 `curl`을 사용하여 HTTP MCP 서버와 통신하는 방법을 정리합니다.

---

## 📌 MCP란?

**MCP(Model Context Protocol)**는 툴(tool) 기반 JSON-RPC 서버 프로토콜로, 각 툴을 메서드처럼 호출하며 데이터 분석 및 결과를 받을 수 있도록 설계된 구조입니다. OP.GG의 `https://mcp-api.op.gg/mcp`는 이를 구현한 대표적인 API 서버입니다.

---

## ✅ MCP 서버 통신 기본 구조

### 📥 요청 형식 (JSON-RPC 2.0)

```json
{
  "jsonrpc": "2.0",
  "method": "<메서드 이름>",
  "params": { ... },
  "id": 1
}
```

### 📤 응답 형식

```json
// 정상 응답
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { ... }
}

// 오류 응답
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Method not found"
  }
}
```

---

## 🛠 도구 목록 조회 (`tools/list`)

```bash
curl -s https://mcp-api.op.gg/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 1
  }' | jq
```

- 반환된 `result.tools` 배열에는 사용 가능한 MCP 도구들이 포함되어 있음
- 각 도구는 `name`, `description`, `inputSchema`를 포함함

---

## 🚀 도구 실행 (`tools/call` 방식)

> MCP 서버는 직접 툴 이름을 `method`로 사용하는 방식이 아닌 **`tools/call`** 메서드를 통해 도구를 호출해야 합니다.

### 예시: `lol-champion-leader-board`

```bash
curl -s https://mcp-api.op.gg/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "lol-champion-leader-board",
      "arguments": {
        "region": "KR",
        "champion": "YASUO"
      }
    },
    "id": 100
  }' | jq
```

---

## 🔍 응답 파싱

MCP는 응답 결과를 `text` 필드에 **JSON 문자열로 이중 인코딩**하여 반환합니다:

```json
{
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"champion\":\"YASUO\",\"data\":{...}}"
      }
    ]
  }
}
```

### ▶ 이중 파싱 방법 (bash)

```bash
curl -s ... | jq -r '.result.content[0].text' | jq .
```

---

## 📊 요청 흐름 요약

```text
[Client] → HTTP POST (JSON-RPC) → https://mcp-api.op.gg/mcp
           ↓
       JSON 응답 도착 (text 속 JSON 문자열)
           ↓
     text 파싱 → JSON 디코딩 → 데이터 활용
```

---

## ⚠️ 유의사항

| 항목                                             | 설명                                                      |
| ------------------------------------------------ | --------------------------------------------------------- |
| 직접 툴 이름 호출 (`method: "lol-champion-..."`) | ❌ 사용 불가 (`Method not found` 오류)                     |
| `tools/call` 사용                                | ✅ 필수                                                    |
| `params` 구조                                    | `{ "name": <도구 이름>, "arguments": { ... } }`           |
| 응답 결과 파싱                                   | `text` 필드는 JSON 문자열이므로 이중 파싱 필요            |
| SSE 응답 대비                                    | 일부 MCP 서버는 스트리밍 응답을 제공하므로 `curl -N` 권장 |

---

## 🧰 MCP에서 제공하는 주요 도구 예시

| 도구 이름                   | 설명                             | 필수 인자                                   |
| --------------------------- | -------------------------------- | ------------------------------------------- |
| `lol-champion-leader-board` | 챔피언별 랭커 순위 조회          | `region`, `champion`                        |
| `lol-champion-analysis`     | 챔피언별 픽률, 아이템, 승률 분석 | `game_mode`, `champion`, `position`, `lang` |
| `lol-summoner-search`       | 소환사 랭크 및 전적 조회         | `game_name`, `tag_line`, `region`, `lang`   |

---

## ✅ 마무리

MCP 기반 API 서버는 JSON-RPC를 기반으로 다양한 도구 호출을 가능하게 하는 강력한 인터페이스입니다. `tools/list`와 `tools/call` 패턴만 이해하면 `curl`만으로도 강력한 리그 오브 레전드 분석 툴을 쉽게 사용할 수 있습니다.



