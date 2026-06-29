# docker_run.sh 수정 사항 (v1 → v2: Single-Node → Multi-Node)

## 🎯 목표
ConnectX-7 RDMA 네트워크를 통한 두 ThinkStation PGX 워크스테이션 간 분산 추론 환경 구성

**변경 대상:**
- 장비 1: 192.168.100.1 (Head Node)
- 장비 2: 192.168.100.2 (Worker Node)

---

## 📝 변경 사항 상세

### 1. 스크립트 헤더 (라인 1-30)
**변경 전:**
```bash
# LLaMA.cpp Interactive Docker Model Launcher (Multi-Hardware Support)
# Auto-detects hardware profile (LOW/MEDIUM/HIGH) and optimizes parameters.
```

**변경 후:**
```bash
# LLaMA.cpp Interactive Docker Model Launcher (Multi-Node Distributed Support)
# Auto-detects hardware profile (LOW/MEDIUM/HIGH) and optimizes parameters.
# Supports distributed inference via ConnectX-7 RDMA with Head/Worker topology.
# Usage: ./docker_run.sh [--role head|worker] [--head-ip 192.168.100.1]
```

**추가 변수:**
```bash
NODE_ROLE="auto"              # auto, head, or worker
HEAD_NODE_IP="192.168.100.1"  # Head node 주소
THIS_NODE_IP=""               # 이 노드의 IP (자동 감지)
CONNECTX_INTERFACE=""         # ConnectX-7 인터페이스 (자동 감지)
```

---

### 2. 새로운 함수들 (라인 212-290)

#### `parse_arguments()`
명령줄 인자 처리:
- `--role [head|worker]` - 노드 역할 지정
- `--head-ip [IP]` - Head 노드 IP 지정
- `--this-ip [IP]` - 이 노드의 IP 지정

#### `detect_node_ip()`
자동으로 192.168.100.x 주소 감지:
- `ip addr show` 또는 `ifconfig` 사용
- 실패 시 127.0.0.1로 폴백

#### `detect_connectx_interface()`
ConnectX-7 네트워크 인터페이스 자동 감지:
- 192.168.100.x가 할당된 인터페이스 찾기
- 일반 이름: ib0, ib1, enpXs0 등
- 실패 시 eth0로 폴백

#### `auto_detect_node_role()`
IP 주소 기반으로 역할 자동 결정:
- `192.168.100.1` → `head`
- 기타 IP → `worker`

---

### 3. Docker 옵션 추가 (라인 595-630)

#### 호스트 네트워킹
```bash
network_args+=(--network host)      # 호스트 네트워크 모드
network_args+=(--ipc host)          # 호스트 IPC 공유
```
**이유:** 컨테이너 간 초저지연, 메모리 공유

#### InfiniBand 지원
```bash
network_args+=(--device=/dev/infiniband)  # InfiniBand 디바이스 접근
```
**이유:** ConnectX-7 하드웨어 직접 접근

#### RDMA 메모리 잠금
```bash
network_args+=(--cap-add IPC_LOCK)        # IPC 잠금 권한
network_args+=(--ulimit memlock=-1:-1)    # 메모리 잠금 무제한
```
**이유:** RDMA는 메모리 페이지 잠금 필요

#### NCCL 환경 변수
```bash
nccl_args+=(-e "NCCL_SOCKET_IFNAME=${CONNECTX_INTERFACE}")  # InfiniBand 인터페이스
nccl_args+=(-e "NCCL_IB_DISABLE=0")                          # InfiniBand 활성화
nccl_args+=(-e "NCCL_IB_HCA=mlx5")                           # Mellanox 장치
nccl_args+=(-e "NCCL_ALGO=Ring")                             # Ring AllReduce
nccl_args+=(-e "NCCL_DEBUG=INFO")                            # 디버그 로깅
```
**이유:** GPU 간 통신을 일반 이더넷이 아닌 RDMA로 강제

#### 멀티 노드 환경 변수
```bash
-e "NODE_ROLE=${NODE_ROLE}"
-e "HEAD_NODE_IP=${HEAD_NODE_IP}"
-e "THIS_NODE_IP=${THIS_NODE_IP}"
```
**이유:** 컨테이너 내부에서 노드 역할 정보 필요

---

### 4. Docker Run 명령어 수정 (라인 639-668)

**변경 전:**
```bash
docker run -d --name "$CONTAINER_NAME" \
    ${gpu_args[@]+"${gpu_args[@]}"} \
    --cap-add IPC_LOCK \
    --cap-add SYS_ADMIN \
    --ulimit memlock=-1:-1 \
    # ... 기타 옵션
```

**변경 후:**
```bash
docker run -d --name "$CONTAINER_NAME" \
    ${gpu_args[@]+"${gpu_args[@]}"} \
    ${network_args[@]+"${network_args[@]}"}    # 호스트 네트, InfiniBand 추가
    ${nccl_args[@]+"${nccl_args[@]}"}          # NCCL 환경 변수 추가
    --cap-add SYS_ADMIN \
    --ulimit stack=67108864 \
    # ... 기타 옵션
    -e "NODE_ROLE=${NODE_ROLE}" \              # 멀티 노드 정보
    -e "HEAD_NODE_IP=${HEAD_NODE_IP}" \
    -e "THIS_NODE_IP=${THIS_NODE_IP}" \
    # ... 모델 인자
```

**변경 이유:**
- 중복 제거: `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1` 배열로 통합
- 네트워크 옵션 중앙화
- 노드 역할 정보 컨테이너에 전달

---

### 5. Multi-Instance 함수 업데이트 (라인 540-577)

Gemma 4 31B 멀티 인스턴스 모드도 동일하게 업데이트:
- `network_args`, `nccl_args` 추가
- 멀티 노드 환경 변수 전달
- 각 인스턴스에 노드 정보 주입

---

### 6. 메인 함수 초기화 (라인 800-820)

**변경 전:**
```bash
main() {
    check_whiptail
    check_prerequisites
    detect_hardware_profile
    # ...
}
```

**변경 후:**
```bash
main() {
    parse_arguments "$@"              # 명령줄 인자 파싱
    
    detect_node_ip                    # 이 노드의 IP 감지
    detect_connectx_interface         # ConnectX 인터페이스 감지
    auto_detect_node_role             # 역할 자동 결정
    
    # 멀티 노드 설정 출력
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Multi-Node Configuration:"
    echo "  Node Role:          $NODE_ROLE"
    echo "  This Node IP:       $THIS_NODE_IP"
    echo "  Head Node IP:       $HEAD_NODE_IP"
    echo "  ConnectX Interface: $CONNECTX_INTERFACE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    check_whiptail
    check_prerequisites
    detect_hardware_profile
    # ...
}
```

---

### 7. UI 업데이트 (라인 430-460)

**모델 선택 화면:**
```bash
# 이전: "Hardware: $hardware_info | GPU: ${GPU_COUNT}x..."

# 수정: 멀티 노드 정보 추가
node_info="🟢 HEAD Node (192.168.100.1)"  # 또는 "🔵 WORKER Node"
# 제목: "Multi-Node Distributed" (이전: "Auto-Optimized")
# 메뉴 정보: "Network: ${CONNECTX_INTERFACE} (192.168.100.x)" 추가
```

---

### 8. Entry Point 수정 (라인 900)

**변경 전:**
```bash
main
```

**변경 후:**
```bash
main "$@"  # 명령줄 인자 전달
```

---

## 🔄 동작 흐름 변화

### v1 (이전)
```
Script Start
  ↓
Hardware Detection
  ↓
UI Menu
  ↓
Model Selection
  ↓
Docker Run (단일 노드)
  ↓
Container Started
```

### v2 (수정됨)
```
Script Start + Arguments
  ↓
Detect Node IP (192.168.100.x)
  ↓
Detect ConnectX Interface (ib0, enpXs0, ...)
  ↓
Auto-Detect Node Role (Head/Worker)
  ↓
Print Multi-Node Config
  ↓
Hardware Detection
  ↓
UI Menu (Multi-Node 정보 표시)
  ↓
Model Selection
  ↓
Build Network Args + NCCL Args
  ↓
Docker Run (네트워크 옵션 포함)
  ↓
Container Started (멀티 노드 환경)
```

---

## 📊 호환성

| 항목 | 영향 범위 |
|------|----------|
| 기존 단일 노드 사용 | ✅ 호환 (명령줄 인자 무시) |
| 모델 선택 메뉴 | ✅ 기능 유지 (UI만 추가) |
| 하드웨어 감지 | ✅ 기능 유지 |
| liteLLM 통합 | ✅ 기능 유지 |
| Multi-Instance (Gemma 31B) | ✅ 멀티 노드 지원 추가 |

---

## ✅ 검증 항목

- [x] Bash 문법 검사 통과
- [x] 함수 이름 고유성 확인
- [x] 환경 변수 충돌 없음
- [x] 배열 문법 정확성
- [x] 호환성 유지 (기존 사용 시 자동 폴백)

---

## 📚 관련 문서

- **MULTI_NODE_SETUP.md** - 상세 설정 가이드
- **MULTI_NODE_QUICK_START.md** - 빠른 시작 가이드
- **README.md** - 기존 프로젝트 문서
