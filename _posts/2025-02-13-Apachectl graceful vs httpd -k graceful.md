---
title: Apachectl graceful vs httpd -k graceful
date: 2025-02-13 14:26:23
categories: [linux, apache]
tags: [linux, apache, httpd, graceful, shell script]     # TAG names should always be lowercase
---


웹호스팅 고객사의 의뢰를 처리하던 중 언제나 그랬듯 `graceful`로 재시작을 마쳤다. 그런데 바로 전화벨이 울리고 해당 서버의 모든 웹 서비스가 순단되었다는 연락이 왔다. 고객의 컨디션을 보아하니 치명적으로 작용하진 않았었던 것 같아 다행이라고 생각했지만 이런 일이 반복된다면 문제가 생길 것 같았다.

다른 엔지니어들은 `apachectl graceful`을 통해 재시작을 진행했었고, 나는 이때에 `httpd -k graceful`로 진행했었다. 아래 명령어에는 무슨 차이가 있을까?

## 공식 문서 설명

### Httpd

`httpd`는 Apache HyperText Transfer Protocol (HTTP) 서버 프로그램이다. 독립 실행형 데몬 프로세스로 실행되도록 설계되었으며, 이를 실행하면 요청을 처리하는 여러 개의 자식 프로세스 또는 스레드 풀이 생성된다.

일반적으로 `httpd`를 직접 호출하지 않고, Unix 기반 시스템에서는 `apachectl`을 통해 실행하는 것이 권장된다.

### Apachectl

`apachectl`은 Apache HTTP 서버의 프론트엔드 역할을 한다. 관리자가 Apache `httpd` 데몬을 제어할 수 있도록 돕는다.

`apachectl` 스크립트는 두 가지 모드로 동작한다.
1. `httpd` 명령어의 단순한 프론트엔드 역할을 하며 필요한 환경 변수를 설정하고 명령어를 전달한다.
2. SysV init 스크립트처럼 동작하여 `start`, `restart`, `stop` 등의 단순한 명령어를 `httpd`에 전달한다.

`apachectl`은 성공 시 0을 반환하며, 오류 발생 시 0보다 큰 값을 반환한다.

## Apachectl 스크립트 분석

```sh
# /usr/local/apache/bin/apachectl
#!/bin/sh
HTTPD='/usr/local/apache/bin/httpd'

if test -f /usr/local/apache/bin/envvars; then
  . /usr/local/apache/bin/envvars
fi

ULIMIT_MAX_FILES="ulimit -S -n `ulimit -H -n`"

if [ "x$ULIMIT_MAX_FILES" != "x" ] ; then
    $ULIMIT_MAX_FILES
fi

case $ARGV in
start|stop|restart|graceful|graceful-stop)
    $HTTPD -k $ARGV
    ERROR=$?
    ;;
configtest)
    $HTTPD -t
    ERROR=$?
    ;;
esac

exit $ERROR
```

`apachectl`은 실행 전에 `ulimit`을 설정하여 파일 디스크립터 제한을 변경하는 기능을 수행한다. 대규모 트래픽을 처리할 경우, 파일 디스크립터 수가 부족하면 연결이 지연되거나 요청이 누락될 수 있다. 이를 방지하기 위해 소프트 리밋을 하드 리밋 수준으로 증가시키는 것이다.

## Ulimit 변화 확인

### 테스트 코드

```sh
cp apachectl apachectl_new
vi apachectl_new

echo "=== 적용 전."
echo "=== 적용 전: 소프트."
ulimit -aS
echo "=== 적용 전: 하드"
ulimit -aH

if [ "x$ULIMIT_MAX_FILES" != "x" ] ; then
    $ULIMIT_MAX_FILES
fi

echo "=== 적용 후."
echo "=== 적용 후: 소프트."
ulimit -aS
echo "=== 적용 후: 하드"
ulimit -aH

apachectl_new graceful
```

### 실행 결과

#### 적용 전
```
core file size          (blocks, -c) 0
data seg size           (kbytes, -d) unlimited
scheduling priority             (-e) 0
file size               (blocks, -f) unlimited
pending signals                 (-i) 14339
max locked memory       (kbytes, -l) 64
max memory size         (kbytes, -m) unlimited
open files                      (-n) 1024
pipe size            (512 bytes, -p) 8
```

#### 적용 후
```
core file size          (blocks, -c) 0
data seg size           (kbytes, -d) unlimited
scheduling priority             (-e) 0
file size               (blocks, -f) unlimited
pending signals                 (-i) 14339
max locked memory       (kbytes, -l) 64
max memory size         (kbytes, -m) unlimited
open files                      (-n) 262144
pipe size            (512 bytes, -p) 8
```

`apachectl`을 사용하면 `ulimit` 설정이 변경되어 파일 디스크립터 수가 증가하는 것을 확인할 수 있다. 즉, 트래픽이 많은 환경에서 서버가 안정적으로 파일 디스크립터를 활용할 수 있도록 도와준다. 파일 디스크립터는 소켓 파일도 포함되므로, 해당 설정 없이 트래픽이 많은 상태에서 `httpd`를 직접 사용해 `graceful`을 수행할 경우 일시적인 서비스 단절이 발생할 수 있다.

## 기타 고려 사항

- `graceful` 실행 시 리소스 부족, 네트워크 연결 시간 등의 문제로 인해 순단이 발생할 가능성이 있다.
- `apachectl`은 환경 변수 설정과 파일 디스크립터 수 조정을 포함하므로 더욱 안정적이다.
- 트래픽이 많은 서버에서는 반드시 `apachectl graceful`을 사용해야 한다.

## 결론

- `apachectl graceful`은 환경 변수 설정을 포함하여 실행되므로 안정적이다.
- `httpd -k graceful`을 직접 실행하면 파일 디스크립터 제한 설정이 반영되지 않아 순단이 발생할 수 있다.
- 따라서, `graceful` 재시작 시 `apachectl`을 사용하는 것이 바람직하다.

