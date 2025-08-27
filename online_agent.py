import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Mamba_Encoder, GC_Qzz_net, GC_Vz_net, inv_dynamics_net

class Online_Agent:
    def __init__(self, state_dim, hidden_dim, action_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p, device, batch_size, offline_checkpoint_path, online_checkpoint_path, **agent_kwargs):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.d_model = d_model
        self.d_state = d_state
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size

        # 정규화 통계 초기화
        self.normalize_stats = None

        self.gamma = agent_kwargs.get('gamma', 0.99)
        self.beta = agent_kwargs.get('beta', 50.0)
        self.lr = agent_kwargs.get('lr', 3e-4)
        self.tau = agent_kwargs.get('tau', 0.01)
        self.warmup_steps = agent_kwargs.get('warmup_steps', 1000)

        self.encoder = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        # Fix encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.critic = GC_Qzz_net(d_model, hidden_dim, double_q = True).to(device)
        self.vz = GC_Vz_net(d_model, hidden_dim).to(device)

        # next_state_estimator 추가
        self.next_state_estimator = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        ).to(device)

        self.inv_dynamics = inv_dynamics_net(d_model, hidden_dim, action_dim).to(device)

        self.optimizer = torch.optim.Adam(list(self.critic.parameters()) + list(self.vz.parameters()) + list(self.next_state_estimator.parameters()) + list(self.inv_dynamics.parameters()), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step / (self.warmup_steps * 1.0), 1.0))

        self.offline_checkpoint_path = offline_checkpoint_path
        self.online_checkpoint_path = online_checkpoint_path

        self.load_checkpoint()

    def normalize_observation(self, obs):
        """관측값 정규화"""
        if self.normalize_stats is None:
            return obs

        if isinstance(obs, torch.Tensor):
            mean = torch.FloatTensor(self.normalize_stats['mean']).to(obs.device)
            std = torch.FloatTensor(self.normalize_stats['std']).to(obs.device)
            return (obs - mean) / (std + 1e-8)
        else:
            # numpy array인 경우
            mean = self.normalize_stats['mean']
            std = self.normalize_stats['std']
            return (obs - mean) / (std + 1e-8)

    def select_action(self, state, goal, epsilon=0.05, action_noise=0.1):
        """Goal-biased 행동 선택 with 골-바이어스 탐색"""
        with torch.no_grad():
            # 정규화
            state_norm = self.normalize_observation(state)
            goal_norm = self.normalize_observation(goal)

            # 단일 관측을 시퀀스로 변환 (context_length에 맞게)
            if len(state_norm.shape) == 1:
                state_seq = np.tile(state_norm, (self.context_length, 1))
                goal_seq = np.tile(goal_norm, (self.context_length, 1))
                mask = np.zeros(self.context_length)
                mask[-1] = 1.0  # 마지막 스텝만 유효
            else:
                state_seq = state_norm
                goal_seq = goal_norm
                mask = np.ones(self.context_length)

            # 텐서로 변환
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)  # (1, L, state_dim)
            goal_tensor = torch.FloatTensor(goal_seq).unsqueeze(0).to(self.device)    # (1, L, state_dim)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)        # (1, L)

            # 인코딩
            z = self.encoder.encode_trajectory(state_tensor, mask_tensor)  # (1, d_model)
            goal_z = self.encoder.encode_trajectory(goal_tensor, mask_tensor)  # (1, d_model)

            # Epsilon-greedy 또는 골-바이어스 탐색
            if np.random.random() < epsilon:
                # 골-바이어스 탐색: z' = z + α(zg - z)
                alpha = np.random.uniform(0.2, 1.0)
                z_prime = z + alpha * (goal_z - z)  # goal 방향으로 탐색

                # z'에서 액션 복원
                action_pred = self.inv_dynamics(z, z_prime, goal_z)  # (1, action_dim)
                action = action_pred.squeeze(0).cpu().numpy()  # (action_dim,)
            else:
                # 정책 기반 행동 (next_state_estimator 사용)
                zp_estimated = self.next_state_estimator(z)  # (1, d_model)
                action_pred = self.inv_dynamics(z, zp_estimated, goal_z)  # (1, action_dim)
                action = action_pred.squeeze(0).cpu().numpy()  # (action_dim,)

            # 액션 노이즈 추가 (탐험)
            noise = np.random.normal(0, action_noise, self.action_dim)
            action = action + noise

            # 행동 범위 클리핑
            action = np.clip(action, -1, 1)

            return action.astype(np.float32)

    def load_checkpoint(self):
        if not os.path.exists(self.online_checkpoint_path):
            # 온라인 체크포인트가 없으면 오프라인 체크포인트 로드
            if os.path.exists(self.offline_checkpoint_path):
                print(f"오프라인 체크포인트 로드: {self.offline_checkpoint_path}")
                checkpoint = torch.load(self.offline_checkpoint_path, weights_only=False)
                # 안전한 state_dict 로딩 (예상치 못한 키 무시)
                self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                self.critic.load_state_dict(checkpoint['critic'])
                self.vz.load_state_dict(checkpoint['vz'])
                if 'next_state_estimator' in checkpoint:
                    self.next_state_estimator.load_state_dict(checkpoint['next_state_estimator'])

                # 옵티마이저와 스케줄러는 새로 초기화 (온라인 학습용)
                print("옵티마이저와 스케줄러는 새로 초기화됩니다.")

                # 정규화 통계 로드
                if 'normalize_stats' in checkpoint:
                    self.normalize_stats = checkpoint['normalize_stats']
                    print(f"정규화 통계 로드 완료: mean shape {self.normalize_stats['mean'].shape}, std shape {self.normalize_stats['std'].shape}")
                else:
                    print("경고: 체크포인트에 정규화 통계가 없습니다. 정규화 없이 진행합니다.")
            else:
                # 오프라인 체크포인트도 없으면 가장 마지막 체크포인트 찾기
                offline_dir = os.path.dirname(self.offline_checkpoint_path)
                print(f"오프라인 체크포인트 디렉토리 확인: {offline_dir}")

                if os.path.exists(offline_dir):
                    # 1. best_model.pth 찾기
                    best_model_path = os.path.join(offline_dir, 'best_model.pth')
                    if os.path.exists(best_model_path):
                        print(f"best_model.pth 로드: {best_model_path}")
                        checkpoint = torch.load(best_model_path, weights_only=False)
                        # 안전한 state_dict 로딩 (예상치 못한 키 무시)
                        self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                        self.critic.load_state_dict(checkpoint['critic'])
                        self.vz.load_state_dict(checkpoint['vz'])
                        if 'next_state_estimator' in checkpoint:
                            self.next_state_estimator.load_state_dict(checkpoint['next_state_estimator'])
                        # 옵티마이저와 스케줄러는 새로 초기화 (온라인 학습용)
                        print("옵티마이저와 스케줄러는 새로 초기화됩니다.")
                    else:
                        # 2. checkpoint_epoch_ 파일들 찾기
                        checkpoint_files = [f for f in os.listdir(offline_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
                        if checkpoint_files:
                            # 에포크 번호로 정렬하여 가장 마지막 것 선택
                            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                            latest_checkpoint = os.path.join(offline_dir, checkpoint_files[-1])
                            print(f"가장 마지막 체크포인트 로드: {latest_checkpoint}")
                            checkpoint = torch.load(latest_checkpoint, weights_only=False)
                            # 안전한 state_dict 로딩 (예상치 못한 키 무시)
                            self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                            self.critic.load_state_dict(checkpoint['critic'])
                            self.vz.load_state_dict(checkpoint['vz'])
                            if 'next_state_estimator' in checkpoint:
                                self.next_state_estimator.load_state_dict(checkpoint['next_state_estimator'])

                            # 옵티마이저와 스케줄러는 새로 초기화 (온라인 학습용)
                            print("옵티마이저와 스케줄러는 새로 초기화됩니다.")
                        else:
                            print("체크포인트를 찾을 수 없습니다. 랜덤 초기화로 시작합니다.")
                else:
                    print("오프라인 체크포인트 디렉토리를 찾을 수 없습니다. 랜덤 초기화로 시작합니다.")
        else:
            # 온라인 체크포인트가 있으면 로드
            print(f"온라인 체크포인트 로드: {self.online_checkpoint_path}")
            checkpoint = torch.load(self.online_checkpoint_path, weights_only=False)

            # 새로운 형식으로 로드 시도
            if 'encoder' in checkpoint:
                # 안전한 state_dict 로딩 (예상치 못한 키 무시)
                self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                self.critic.load_state_dict(checkpoint['critic'])
                self.vz.load_state_dict(checkpoint['vz'])
                if 'next_state_estimator' in checkpoint:
                    self.next_state_estimator.load_state_dict(checkpoint['next_state_estimator'])
                self.inv_dynamics.load_state_dict(checkpoint['inv_dynamics'])
                # 옵티마이저와 스케줄러는 온라인 체크포인트에서 로드
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                # 기존 형식으로 로드
                agent_state = checkpoint['agent_state']
                # 안전한 state_dict 로딩 (예상치 못한 키 무시)
                self.encoder.load_state_dict(agent_state['encoder'], strict=False)
                self.critic.load_state_dict(agent_state['critic'])
                self.vz.load_state_dict(agent_state['vz'])
                if 'next_state_estimator' in agent_state:
                    self.next_state_estimator.load_state_dict(agent_state['next_state_estimator'])
                self.inv_dynamics.load_state_dict(agent_state['inv_dynamics'])
                # 옵티마이저와 스케줄러는 온라인 체크포인트에서 로드
                self.optimizer.load_state_dict(agent_state['optimizer'])
                self.scheduler.load_state_dict(agent_state['scheduler'])

            # 정규화 통계 로드
            if 'normalize_stats' in checkpoint:
                self.normalize_stats = checkpoint['normalize_stats']
                print(f"정규화 통계 로드 완료: mean shape {self.normalize_stats['mean'].shape}, std shape {self.normalize_stats['std'].shape}")

            self.epoch = checkpoint.get('epoch', checkpoint.get('episode', 0))
            self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('best_success_rate', 0.0))

    def update(self, states, actions, next_states, rewards, dones, mask, goal_obs):
        """
        states: (B, K, state_dim)
        actions: (B, K, action_dim)
        next_states: (B, K, state_dim)
        goal_obs: (B, K, state_dim)
        rewards: (B, K)
        dones: (B, K)
        mask: (B, K)
        """

        # --- 타입/장치 정리 ---
        states = states.to(self.device)
        actions = actions[:, -1:].to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards[:, -1:].to(self.device)
        dones = dones[:, -1:].to(self.device)
        mask = mask[:, -1:].to(self.device)
        goal_obs = goal_obs.to(self.device)

        # --- 인코딩 ---
        z = self.encoder.encode_trajectory(states, mask)
        zp = self.encoder.encode_trajectory(next_states, mask)

        # Goal 인코딩 - 안전하게 처리
        if goal_obs.dim() == 2:  # (B, state_dim) -> (B, L, state_dim)로 변환
            goal_obs = goal_obs.unsqueeze(1).expand(-1, states.shape[1], -1)
            goal_mask = torch.ones_like(mask)  # 모든 스텝이 유효
        else:  # (B, L, state_dim)
            goal_mask = torch.ones_like(mask)  # 모든 스텝이 유효

        z_g = self.encoder.encode_trajectory(goal_obs, goal_mask)

        # --- 타깃 Q 계산 ---
        with torch.no_grad():
            v_target_next = self.vz(zp, z_g).detach()
            target_q = rewards + self.gamma * (1.0 - dones) * v_target_next

        # --- 현재 Q ---
        q_current1, q_current2 = self.critic(z, zp, z_g)

        # --- mask_last: (B,1) ---
        mask_last = None
        if mask is not None:
            mask_last = mask[:, -1:].to(self.device)  # 마지막 유효 스텝만

        # --- Critic loss (둘 다 (B,1)) ---
        if mask_last is not None:
            w = mask_last
            critic_l1 = F.smooth_l1_loss(q_current1, target_q, reduction='none')  # (B,1)
            critic_l2 = F.smooth_l1_loss(q_current2, target_q, reduction='none')  # (B,1)
            critic_loss = ((critic_l1 * w).sum() + (critic_l2 * w).sum()) / w.sum().clamp(min=1.0)
        else:
            critic_loss = F.smooth_l1_loss(q_current1, target_q) + F.smooth_l1_loss(q_current2, target_q)

        # --- 현재 V ---
        v_current = self.vz(z, z_g)  # (B,1)
        current_q = torch.min(q_current1, q_current2)
        v_loss = F.mse_loss(v_current, current_q)

        # --- next state estimation ---
        zp_estimated = self.next_state_estimator(z)  # (B, d_model)
        state_loss = F.mse_loss(zp_estimated, zp)   # (B, d_model) -> scalar

        # --- inverse dynamics (advantage-weighted) ---
        a_pred = self.inv_dynamics(z, zp_estimated, z_g)  # (B,action_dim)

        # advantage (detach & clip/normalize)
        qmin = torch.min(q_current1, q_current2)     # (B,1)
        v_pred = v_current                           # (B,1)
        adv = (qmin - v_pred).detach()               # (B,1)
        adv = adv.clamp(min=-5.0, max=5.0)
        w_adv = torch.exp(self.beta * adv)           # (B,1)
        if mask_last is not None:
            w_adv = w_adv * mask_last + (1 - (mask_last > 0).float()) * 0.0

        # actions 차원 맞추기: (B, 1, action_dim) -> (B, action_dim)
        actions_flat = actions.squeeze(1) if actions.dim() == 3 else actions
        a_mse = F.mse_loss(a_pred, actions_flat, reduction='none').mean(dim=1, keepdim=True)  # (B,1)
        if mask_last is not None:
            action_loss = (a_mse * w_adv).sum() / w_adv.sum().clamp(min=1.0)
        else:
            action_loss = (a_mse * w_adv).mean()

        # --- 옵티마이즈 ---
        self.optimizer.zero_grad()
        total_loss = critic_loss + action_loss + v_loss + state_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {
            'critic_loss': critic_loss.item(),
            'action_loss': action_loss.item(),
            'v_loss': v_loss.item(),
            'state_loss': state_loss.item(),
        }
