# Multi-Node Setup - 빠른 시작 가이드

## 🚀 5분 안에 시작하기

### 장비 1 (Head Node - 192.168.100.1)
```bash
cd /Users/gotaejong/ExternHard/97_Workspace/localLLMService
./docker_run.sh --role head --this-ip 192.168.100.1
```

### 장비 2 (Worker Node - 192.168.100.2)
```bash
cd /Users/gotaejong/ExternHard/97_Workspace/localLLMService
./docker_run.sh --role worker --this-ip 192.168.100.2 --head-ip 192.168.100.1
```

---

## ✅ 실행 전 필수 확인

```bash
# 1. 네트워크 연결 확인 (장비 1에서)
ping -I 192.168.100.1 192.168.100.2

# 2. 인터페이스 이름 확인 (ib0, enpXs0 등)
ip addr show | grep 192.168.100

# 3. GPU 인식 확인
nvidia-smi

# 4. InfiniBand 드라이버 확인
ibstat
```

---

## 📊 수정된 주요 사항

| 항목 | 이전 | 수정됨 |
|------|------|--------|
| **네트워크** | Docker 브릿지 | `--network host`, `--ipc host` |
| **InfiniBand** | 없음 | `--device=/dev/infiniband` 추가 |
| **메모리** | 표준 | `--ulimit memlock=-1` (RDMA 핀 메모리) |
| **NCCL** | 기본값 | `NCCL_SOCKET_IFNAME` 자동 감지 |
| **노드 감지** | 없음 | IP 기반 자동 Head/Worker 구분 |

---

## 🔍 실행 중 확인사항

### 컨테이너 로그 확인
```bash
docker logs -f llama-server
```

### NCCL 디버그 정보 보기
```bash
# 이미 포함됨: NCCL_DEBUG=INFO
docker logs llama-server | grep NCCL
```

### 네트워크 연결 상태
```bash
# 컨테이너 내부에서
docker exec llama-server ip a

# InfiniBand 상태
docker exec llama-server ibstatus
```

---

## ⚠️ 일반적인 문제

| 문제 | 해결책 |
|------|--------|
| "192.168.100.x 찾을 수 없음" | `ip a` 확인, 네트워크 설정 검토 |
| "InfiniBand 디바이스 없음" | OFED 드라이버 설치: `mlnxofedinstall` |
| "NCCL_SOCKET_IFNAME 자동 감지 실패" | 수동으로 지정: 스크립트 수정 (`CONNECTX_INTERFACE="ib0"`) |
| "포트 8080 충돌" | `docker rm -f llama-server` 후 재시작 |

---

## 📝 명령어 참고

### 자동 감지 (권장)
```bash
./docker_run.sh
```
- IP 자동 감지 (192.168.100.x)
- 인터페이스 자동 감지
- 역할 자동 결정 (head/worker)

### 명시적 설정
```bash
./docker_run.sh --role head --this-ip 192.168.100.1 --head-ip 192.168.100.1
```

### 컨테이너 중지
```bash
docker stop llama-server
docker rm llama-server
```

---

## 📈 성능 체크

```bash
# 양쪽 GPU 메모리 확인
# 장비 1 & 2에서 각각:
nvidia-smi

# 컨테이너 메모리 사용
docker stats llama-server

# 네트워크 대역폭 (InfiniBand)
# 장비 1에서:
watch 'ethtool -S ib0 | grep -E "bytes|packets"'
```

---

더 자세한 설명은 **MULTI_NODE_SETUP.md** 참고
