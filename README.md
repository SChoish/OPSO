# OPSO: Offline Pretraining with State-Only Imitation

Mamba 기반 **희소 목표 조건 강화학습(Sparse Goal-Conditioned RL)** 파이프라인입니다.  
오프라인 잠재 사전학습 후, **상태만 사용하는 모방(State-Only Imitation)**으로 온라인 제어를 합니다.  
이론적으로 **Implicit Latent QSS / Proto-D3G** 스타일에 맞춰 정리되어 있으며, full D3G(cycle consistency, 완전한 proposer 학습)는 구현하지 않습니다.

---

## 주요 기능

### 1. 오프라인 잠재 사전학습 (Offline Latent Pretraining)

- **Mamba 인코더**: 상태 시퀀스 → 잠재 벡터. 제어 경로는 `encode_last_valid`(마지막 유효 스텝) 사용. **오프라인에서만 학습**하며, 온라인에서는 고정·추론 전용으로 사용합니다.
- **Qzz(z, z', z_g)**: 목표 조건 **전이 가치(goal-conditioned latent transition-value)**, QSS 영감. Sigmoid → [0,1] 도달 가능성/성공 확률 해석.
- **Vz(z, z_g)**: 현재 잠재에서의 **도달 가능성 프록시(reachability proxy)**. Expectile regression으로 학습.
- **Latent proposer (proto-τ)**: (z, z_g) → 제안 다음 잠재. 예측 사전학습(MSE) + 정렬. Full D3G의 cycle consistency는 미구현.
- **GC_success_head**: 종료/성공 신호 BCE 예측.
- **InfoNCE**: 대조 학습. **Trajectory alignment**: 전이 방향과 목표 방향 정렬.

### 2. 온라인 제어 (Proto-D3G Style)

- **인코더**: 오프라인에서 사전학습된 인코더를 **그대로 고정(frozen)**하여 사용. `eval()` + `no_grad`로 추론만 수행하며, 온라인에서는 **잠재 공간(latent space)** 위의 control head만 학습합니다 (critic, Vz, proposer, inv_dynamics). 인코더 타깃 네트워크 없음.
- **제어 식**: 제안 잠재 `z' = τ(z, z_g)` (latent proposer), 액션 `a = I(z, z', z_g)` (inverse dynamics).
- **Inverse dynamics**: (z, z', z_g) → 행동. 제안된 다음 잠재를 행동으로 연결.
- **Target 네트워크**: `critic_target`, `vz_target` EMA로 TD 타깃 안정화.
- **탐험**: ε-greedy 시 **휴리스틱 목표 방향 보간**(goal-direction interpolation). QSS 이론적 τ 최적화는 아님.
- **Advantage-weighted inverse dynamics**: `exp(adv/temperature)` + 상한 cap으로 가중치 안정화.
- **선택 사항**: `student_checkpoint_path`로 오프라인 디스틸된 경량 인코더를 불러와 동일하게 고정·추론 전용으로 사용 가능. (워크플로: teacher 오프라인 사전학습 → student 오프라인 디스틸 → student frozen 배포.)

---

## 프로젝트 구조

```
OPSO/
├── model.py              # Mamba 인코더, GC_Qzz_net, GC_Vz_net, inv_dynamics_net
├── offline_agent.py      # 오프라인 에이전트 (encoder, Qzz, Vz, latent_proposer, GC_success_head)
├── online_agent.py       # 온라인 에이전트 (encoder 고정·추론 전용, latent-space만 학습)
├── offline_trainer.py    # 오프라인 학습 루프
├── online_trainer.py     # 온라인 학습 루프
├── utils/                # 유틸 모듈
│   ├── __init__.py       # 공통 (expectile_loss, last_valid_index_from_mask)
│   ├── ogbench_utils.py  # OGBench 데이터셋 다운로드/로드
│   └── d4rl_utils.py     # D4RL AntMaze 로드
├── config/
│   ├── default.yaml      # 기본 설정
│   ├── <데이터셋명>.yaml  # 데이터셋별 설정 (antmaze-medium-navigate-v0.yaml 등)
│   └── __init__.py       # get_offline_config(overrides, env), get_online_config(overrides, env)
├── docs/
│   ├── RECOMMENDATIONS.md              # 코드/알고리즘 제언
│   ├── PROTO_D3G_REFACTOR_SUMMARY.md   # Proto-D3G 리팩터 요약
│   └── REFACTOR_SECONDPASS_SUMMARY.md  # 마스크/패딩, V학습, InfoNCE 등 수정 요약
├── datasets/             # 데이터셋 저장
├── offline_checkpoints/  # 오프라인 체크포인트
└── online_checkpoints/   # 온라인 체크포인트
```

---

## 설치 및 설정

```bash
conda create -n offrl python=3.10
conda activate offrl

pip install -r requirements.txt
# 또는
pip install torch torchvision torchaudio mamba-ssm ogbench matplotlib numpy tqdm PyYAML
```

**실행**: 오프라인/온라인 학습은 `conda activate offrl` 한 뒤 같은 환경에서 실행하면 됩니다.

데이터셋은 첫 오프라인 학습 실행 시 자동 다운로드되며, `datasets/<name>/` 아래 `train.npz`, `val.npz`로 저장됩니다.

---

## 사용법

### 오프라인 학습

```bash
conda activate offrl
python offline_trainer.py \
    --dataset antmaze-medium-navigate-v0 \
    --max_trajectories 1000 \
    --batch_size 128 \
    --num_epochs 200
```

체크포인트: `offline_checkpoints/<dataset>/best_model.pth`, `best_offline_checkpoint_<dataset>.pth` (온라인 부트스트랩용).

#### D4RL AntMaze (OGBench 전 간단 테스트용)

데이터 소스를 `d4rl`로 두면 D4RL antmaze 데이터를 사용합니다. `pip install gym d4rl` 필요 (MuJoCo 의존성 있음).

```bash
conda activate offrl
python offline_trainer.py \
    --data d4rl \
    --dataset antmaze-medium-diverse-v0 \
    --max_trajectories 500 \
    --batch_size 128 \
    --num_epochs 100
```

지원 데이터셋: `antmaze-umaze-v0`, `antmaze-umaze-diverse-v0`, `antmaze-medium-play-v0`, `antmaze-medium-diverse-v0`, `antmaze-large-play-v0`, `antmaze-large-diverse-v0`.

### 온라인 학습

오프라인 체크포인트에서 encoder(고정), critic, vz, latent_proposer, inv_dynamics 등을 불러온 뒤, **인코더는 업데이트하지 않고** critic / Vz / proposer / inv_dynamics만 온라인으로 학습합니다.

```bash
conda activate offrl
python online_trainer.py \
    --dataset antmaze-medium-navigate-v0 \
    --max_episodes 1000 \
    --batch_size 32
```

- **인코더**: 항상 `eval()` + `no_grad`로 추론만 수행. 옵티마이저에 포함되지 않음.
- **선택**: `student_checkpoint_path`를 넘기면 해당 체크포인트의 encoder를 불러와 동일하게 고정 사용 (디스틸된 경량 인코더 배포용).

기본 경로:  
- 오프라인: `./offline_checkpoints/<dataset>/best_offline_checkpoint_<dataset>.pth`  
- 온라인: `./online_checkpoints/<dataset>/best_online_checkpoint_<dataset>.pth`

---

## 주요 하이퍼파라미터

| 구분 | 항목 | 설명 |
|------|------|------|
| 공통 | `context_length` | 시퀀스 길이 (기본 100) |
| 공통 | `d_model` | 잠재 차원 (기본 128) |
| 오프라인 | `expectile_tau` | V 학습 (τ<0.5 비관적, τ>0.5 낙관적; 기본 0.3) |
| 오프라인 | `beta_s`, `beta_r`, `beta_nce`, `beta_v`, `beta_a` | Proposer, success, InfoNCE, V, alignment 가중치 |
| 온라인 | `epsilon_start` / `epsilon_end` | 탐험률 스케줄 |
| 온라인 | `adv_weight_temperature`, `adv_weight_cap` | Advantage 가중치 안정화 (기본 1.0, 20.0) |
| 온라인 | `tau` | Critic/Vz target EMA 계수 (기본 0.01) |

---

## 지원 데이터셋 (OGBench)

- `antmaze-medium-navigate-v0`, `antmaze-medium-explore-v0`, `antmaze-medium-stitch-v0`
- `antmaze-large-navigate-v0`, `antmaze-large-explore-v0`, `antmaze-large-stitch-v0`
- `humanoidmaze-medium-navigate-v0`, `humanoidmaze-medium-stitch-v0`
- `humanoidmaze-large-navigate-v0`, `humanoidmaze-large-stitch-v0`

---

## 모니터링

- **오프라인**: total_loss, critic_loss, v_loss, state_loss(proposer MSE), reward_loss(GC_success_head), nce_loss, alignment_loss.
- **온라인**: critic_loss, action_loss, v_loss, proposer_loss, success rate, episode reward, epsilon/action_noise 스케줄.

학습 곡선: `offline_checkpoints/.../training_curves.png`, `online_checkpoints/.../online_training_curves_<dataset>.png`.

---

## 이론적 위치 (Proto-D3G)

- **Qzz**: 전이 가치(transition-value). Sigmoid → 도달 가능성/성공 확률 해석.
- **Vz**: 도달 가능성 프록시. Expectile로 학습.
- **Latent proposer**: proto-τ. 예측 사전학습 + 온라인 Q-max 항. Full D3G의 cycle consistency 또는 Q-guided forward model은 TODO.
- **제어**: `a = I(z, τ(z,z_g), z_g)`. 탐험은 휴리스틱 목표 방향 보간.
- **인코더**: 오프라인 전용. 온라인에서는 고정·추론 전용이며, 학습은 잠재 공간 위의 head만 수행.

자세한 리팩터 요약은 `docs/PROTO_D3G_REFACTOR_SUMMARY.md`, 마스크/패딩·V학습 등 수정 사항은 `docs/REFACTOR_SECONDPASS_SUMMARY.md`를 참고하세요.

---

## 라이선스

MIT License

---

## 참고

- [OGBench](https://github.com/ogbench/ogbench)
- [Mamba](https://github.com/state-spaces/mamba)
- Expectile regression, state-only imitation, goal-conditioned RL 관련 선행 연구
