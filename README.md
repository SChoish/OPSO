# OPSO: Offline Pretraining with State-Only Imitation

OPSO는 오프라인 사전학습(Offline Pretraining)과 상태-전용 모방학습(State-Only Imitation)을 결합한 Mamba 기반 강화학습 시스템입니다. 시퀀스 상태 표현을 통해 효율적인 정책 학습과 온라인 적응을 달성합니다.

## 🚀 주요 기능

### 오프라인 사전학습 (Offline Pretraining)
- **Mamba 기반 인코더**: 시퀀스 상태 표현 학습
- **Q(z,z') 함수**: 상태-다음상태 간 Q-값 학습
- **V(z,z_g) 함수**: 상태-목표 간 가치 함수 학습
- **Expectile Regression**: 강건한 가치 함수 학습 (τ=0.9)
- **InfoNCE 손실**: 대조 학습을 통한 표현 학습
- **Trajectory Alignment**: 목표 지향적 궤적 정렬
- **보조 손실**: 상태 예측, 보상/완료 예측

### 상태-전용 모방학습 (State-Only Imitation)
- **Inverse Dynamics**: 상태 전이로부터 액션 복원
- **골-바이어스 탐색**: 효율적인 탐색 전략
- **Epsilon Decay**: 점진적 탐색 감소
- **액션 노이즈 스케줄**: 탐험과 활용의 균형

## 📁 프로젝트 구조

```
OPSO/
├── offline_agent.py          # 오프라인 학습 에이전트
├── online_agent.py           # 온라인 학습 에이전트
├── offline_trainer.py        # 오프라인 학습 트레이너
├── online_trainer.py         # 온라인 학습 트레이너
├── model.py                  # Mamba 인코더 모델
├── ogbench_utils.py          # OGBench 데이터셋 유틸리티
├── utils.py                  # 기타 유틸리티
└── datasets/                 # 데이터셋 저장소
```

## 🛠️ 설치 및 설정

### 1. 환경 설정
```bash
# Conda 환경 생성
conda create -n offrl python=3.10
conda activate offrl

# 필수 패키지 설치
pip install torch torchvision torchaudio
pip install ogbench
pip install matplotlib numpy
```

### 2. 데이터셋 다운로드
```bash
# OGBench 데이터셋 자동 다운로드 (첫 실행 시)
python offline_trainer.py --dataset antmaze-medium-navigate-v0
```

## 🎯 사용법

### 오프라인 학습
```bash
python offline_trainer.py \
    --dataset antmaze-medium-navigate-v0 \
    --max_trajectories 100 \
    --batch_size 16 \
    --num_epochs 200
```

### 온라인 학습
```bash
python online_trainer.py \
    --dataset antmaze-medium-navigate-v0 \
    --max_episodes 1000 \
    --batch_size 8
```

## 🔧 주요 하이퍼파라미터

### 오프라인 학습
- `context_length`: 100 (시퀀스 길이)
- `d_model`: 128 (인코더 차원)
- `hidden_dim`: 256 (은닉층 차원)
- `lr`: 0.0003 (학습률)

### 온라인 학습
- `epsilon_start`: 0.30 (초기 탐색률)
- `epsilon_end`: 0.05 (최종 탐색률)
- `action_noise_start`: 0.20 (초기 액션 노이즈)
- `action_noise_end`: 0.05 (최종 액션 노이즈)

## 🎨 상태-전용 모방학습 전략

OPSO의 핵심인 상태-전용 모방학습에서 사용되는 전략:

### 1. 골-바이어스 탐색
```python
# 골-바이어스 탐색: z' = z + α(zg - z)
alpha = np.random.uniform(0.2, 1.0)
z_prime = z + alpha * (goal_z - z)

# Inverse Dynamics로 액션 복원
action = inv_dynamics(z, z_prime, goal_z)
```

### 2. Expectile Regression 기반 가치 학습
```python
# 낙관적 가치 함수 학습 (τ=0.9)
def expectile_loss(pred, target, tau):
    diff = pred - target
    weight = torch.where(diff >= 0, tau, 1 - tau)
    return (weight * (diff ** 2)).mean()
```

### 3. Trajectory Alignment
```python
# 목표 지향적 궤적 정렬
d_z = zp - z  # 상태 전이 방향
goal_dir = goal_z - z  # 목표 방향
alignment_loss = -cosine_similarity(d_z, goal_dir)
```

## 📊 지원 데이터셋

- `antmaze-medium-navigate-v0`
- `antmaze-medium-explore-v0`
- `antmaze-medium-stitch-v0`
- `antmaze-large-navigate-v0`
- `antmaze-large-explore-v0`
- `antmaze-large-stitch-v0`
- `humanoidmaze-medium-navigate-v0`
- `humanoidmaze-medium-stitch-v0`
- `humanoidmaze-large-navigate-v0`
- `humanoidmaze-large-stitch-v0`

## 🔍 모니터링

학습 과정은 다음 메트릭으로 모니터링됩니다:

### 오프라인 사전학습
- Critic Loss (Q-function)
- V-function Loss (Expectile Regression)
- State Loss (Next State Prediction)
- Reward Loss (Reward/Done Prediction)
- InfoNCE Loss (Contrastive Learning)
- Alignment Loss (Trajectory Alignment)

### 상태-전용 모방학습
- Train Loss (Inverse Dynamics)
- Success Rate
- Episode Reward
- Epsilon Decay
- Action Noise Schedule
- Goal-biased Exploration Rate

## 📈 결과

학습 완료 후 다음 파일들이 생성됩니다:
- `best_model.pth`: 최고 성능 모델
- `training_curves.png`: 학습 곡선
- `online_training_curves.png`: 온라인 학습 곡선

## 🤝 기여

이 프로젝트에 기여하고 싶으시다면:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License

## 🔬 핵심 아이디어

OPSO는 다음과 같은 핵심 아이디어를 바탕으로 합니다:

1. **Offline Pretraining**: 오프라인 데이터로부터 강건한 상태 표현과 가치 함수 학습
2. **State-Only Imitation**: 액션 정보 없이 상태 전이만으로 정책 학습
3. **Expectile Regression**: 낙관적 가치 함수로 탐험 유도
4. **Trajectory Alignment**: 목표 지향적 궤적 정렬로 효율적 학습

## 🙏 감사의 말

- [OGBench](https://github.com/ogbench/ogbench) 데이터셋 제공
- [Mamba](https://github.com/state-spaces/mamba) 아키텍처 참고
- Expectile Regression과 State-Only Imitation 연구에 영감을 준 선행 연구들
