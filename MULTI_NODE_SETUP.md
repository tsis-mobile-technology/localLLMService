# Multi-Node Distributed Inference Setup (ConnectX-7 RDMA)

이 가이드는 두 대의 ThinkStation PGX 워크스테이션을 ConnectX-7 네트워크로 연결하여 분산 추론 환경을 구성하는 방법을 설명합니다.

## 📋 사전 요구사항

### 1. 네트워크 설정
- **장비 1 (Head Node)**: IP `192.168.100.1` (DHCP 또는 정적 IP)
- **장비 2 (Worker Node)**: IP `192.168.100.2` (DHCP 또는 정적 IP)
- **인터페이스**: ConnectX-7 기반 (Mellanox/NVIDIA)
  - `ip a` 명령어로 확인: `192.168.100.x` 대역 인터페이스 이름 (예: `ib0`, `enpXs0`)

### 2. 호스트 소프트웨어 (양쪽 모두)
```bash
# NVIDIA OFED 드라이버 설치 확인
mlx_loader -v  # Mellanox driver loader
lspci | grep Mellanox  # ConnectX-7 인식 확인

# NVIDIA Container Toolkit 설치
sudo apt install nvidia-container-toolkit

# InfiniBand 유틸리티
sudo apt install infiniband-diags  # ibstat, ibstatus 등
```

### 3. 네트워크 연결 테스트
```bash
# 장비 1 (Head)에서:
ping -I 192.168.100.1 192.168.100.2

# 방화벽 확인 (Ray/NCCL 포트 개방)
sudo ufw allow 192.168.100.0/24/any
# 또는 방화벽 비활성화
sudo ufw disable
```

---

## 🚀 사용 방법

### 옵션 1: 자동 감지 (권장)

#### 장비 1 (Head Node - 192.168.100.1)
```bash
cd /Users/gotaejong/ExternHard/97_Workspace/localLLMService
./docker_run.sh
```

**또는 명시적 지정:**
```bash
./docker_run.sh --role head --this-ip 192.168.100.1
```

#### 장비 2 (Worker Node - 192.168.100.2)
```bash
./docker_run.sh
```

**또는 명시적 지정:**
```bash
./docker_run.sh --role worker --this-ip 192.168.100.2 --head-ip 192.168.100.1
```

### 옵션 2: 명시적 역할 지정

양쪽 장비에서 다음과 같이 실행:

**장비 1 (Head):**
```bash
./docker_run.sh --role head --head-ip 192.168.100.1
```

**장비 2 (Worker):**
```bash
./docker_run.sh --role worker --head-ip 192.168.100.1 --this-ip 192.168.100.2
```

---

## 🔧 스크립트 수정 사항 (v2.0 Multi-Node)

### 1. Docker 옵션 추가
| 옵션 | 목적 | 이유 |
|------|------|------|
| `--network host` | 호스트 네트워크 모드 | 컨테이너 간 초저지연 통신 |
| `--ipc host` | 호스트 IPC 공유 | 메모리 공유 메커니즘 |
| `--device=/dev/infiniband` | InfiniBand 디바이스 매핑 | ConnectX-7 하드웨어 접근 |
| `--ulimit memlock=-1` | 메모리 잠금 | RDMA Pinned Memory 지원 |

### 2. NCCL 환경 변수 (GPU 통신 설정)
```bash
NCCL_SOCKET_IFNAME=<ConnectX-7 인터페이스>  # 예: ib0
NCCL_IB_DISABLE=0                            # InfiniBand 활성화
NCCL_IB_HCA=mlx5                             # Mellanox 장치 지정
NCCL_ALGO=Ring                               # Ring AllReduce 알고리즘
NCCL_DEBUG=INFO                              # 디버그 로깅
```

### 3. 멀티 노드 환경 변수
```bash
NODE_ROLE=head|worker       # 이 노드의 역할
HEAD_NODE_IP=192.168.100.1  # 헤드 노드 주소
THIS_NODE_IP=192.168.100.x  # 이 노드의 주소
```

---

## 📊 실행 흐름

### 자동 감지 프로세스
1. **IP 감지**: `ip a` / `ifconfig`로 `192.168.100.x` 주소 확인
2. **인터페이스 감지**: ConnectX-7 인터페이스 이름 (ib0, enpXs0 등) 찾기
3. **역할 자동 결정**:
   - `192.168.100.1` → `head`
   - `192.168.100.2` → `worker`
   - 기타 → `worker`

### 컨테이너 실행 시 포함되는 항목
```
✅ 호스트 네트워크 + IPC
✅ InfiniBand 디바이스 바인딩
✅ RDMA 메모리 잠금
✅ NCCL 환경 변수 자동 구성
✅ 노드 역할 정보 컨테이너에 전달
```

---

## 🔍 문제 해결

### 1. ConnectX-7 인터페이스를 찾을 수 없는 경우
```bash
# 수동으로 인터페이스 확인
ip addr show
# 또는
ifconfig | grep 192.168.100

# 인터페이스 이름을 스크립트에 하드코딩
# docker_run.sh 수정:
# CONNECTX_INTERFACE="ib0"  # 자동 감지 대신 수동 입력
```

### 2. InfiniBand 드라이버 문제
```bash
# 드라이버 상태 확인
ibstat
ibstatus

# OFED 드라이버 재설치
sudo ./mlnxofedinstall --add-kernel-support
```

### 3. NCCL 통신 확인
```bash
# 컨테이너 내부에서
docker exec -it llama-server bash

# 환경 변수 확인
echo $NCCL_SOCKET_IFNAME
echo $NCCL_DEBUG

# NCCL 테스트 (설치된 경우)
ncclAllReduceTest -b 8M -e 8M
```

### 4. 포트 충돌
```bash
# 사용 중인 포트 확인
lsof -i :8080
ss -tlnp | grep 8080

# 컨테이너 정리
docker ps -a
docker rm -f llama-server
```

### 5. GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 멀티 노드에서 사용 가능한 총 메모리
# 예: 192GB (96GB x 2) = 장비 1 + 장비 2
```

---

## 📈 성능 최적화 팁

### 1. 대역폭 활용 확인
```bash
# ConnectX-7 대역폭 (200Gbps 이론)
ethtool -S ib0 | grep -i bytes

# 또는
ibstat -s
```

### 2. NCCL 알고리즘 튜닝
```bash
# Ring AllReduce (기본값) - 대기시간 최소
NCCL_ALGO=Ring

# Tree - 높은 대역폭 환경에서 더 효율적
NCCL_ALGO=Tree
```

### 3. 메모리 설정
```bash
# docker_run.sh의 shm_size 조정
# HIGH 프로필의 경우: 100g (현재값) → 필요시 더 크게

# 또는 호스트의 tmpfs 설정
mount -o remount,size=120G /dev/shm
```

---

## ✅ 검증 체크리스트

실행 전 다음을 확인하세요:

- [ ] 두 장비가 `192.168.100.x` 대역에서 Ping 통신 가능
- [ ] 양쪽 호스트의 방화벽 확인 (`ufw` 비활성화 또는 포트 개방)
- [ ] NVIDIA OFED 드라이버 설치 확인 (`mlx_loader -v`, `ibstat`)
- [ ] NVIDIA Container Toolkit 설치
- [ ] Docker 데몬 실행 중
- [ ] GPU 인식 확인 (`nvidia-smi`)
- [ ] 모델 파일 위치 확인 (`~/Programming/models/`)
- [ ] ConnectX 인터페이스 이름 확인 (`ip a` 또는 `ifconfig`)

---

## 🎯 다음 단계

### Ray Cluster 통합 (선택사항)
분산 처리 프레임워크(Ray)를 별도로 구성하면:
- 여러 워커 장비 확장 가능
- 동적 로드 밸런싱
- 자동 장애 조치

현재 버전은 **NCCL 기반 GPU 통신**에 최적화되어 있습니다.

---

## 📞 지원

문제가 발생하면:
1. `docker logs llama-server` 확인
2. 위 "문제 해결" 섹션 참고
3. NCCL 디버그 로그: `NCCL_DEBUG=INFO` (기본값)
