"""
Online control for sparse goal-conditioned RL (proto-D3G style).
Encoder is offline-pretrained only: frozen in online phase, inference-only. No encoder target;
online learning happens in latent space on top of frozen encoder (critic, Vz, proposer, inv_dynamics).
Control: proposer tau(z,z_g) -> desired next latent; action a = I(z, tau(z,z_g), z_g).
TODO (offline distillation): (a) pretrain teacher encoder offline (b) distill to lightweight student offline (c) deploy student frozen via student_checkpoint_path.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Mamba_Encoder, GC_Qzz_net, GC_Vz_net, inv_dynamics_net
from utils import expectile_loss, last_valid_index_from_mask

# Advantage weight cap and temperature for inv_dynamics weighting (avoid exp(beta*adv) overflow).
ADV_WEIGHT_TEMPERATURE = 1.0
ADV_WEIGHT_CAP = 20.0


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
        self.adv_temp = agent_kwargs.get('adv_weight_temperature', ADV_WEIGHT_TEMPERATURE)
        self.adv_cap = agent_kwargs.get('adv_weight_cap', ADV_WEIGHT_CAP)
        self.critic_lr = agent_kwargs.get('critic_lr', 3e-5)
        self.env_lr = agent_kwargs.get('env_lr', 3e-4)
        self.tau = agent_kwargs.get('tau', 0.01)
        self.lmbda = agent_kwargs.get('lmbda', 0.1)
        self.warmup_steps = agent_kwargs.get('warmup_steps', 1000)
        self.expectile_tau = agent_kwargs.get('expectile_tau', 0.9)
        # Encoder: offline-pretrained only; never trained or optimized online. Inference-only.
        self.encoder = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.critic = GC_Qzz_net(d_model, hidden_dim, double_q=True).to(device)
        self.critic_target = GC_Qzz_net(d_model, hidden_dim, double_q=True).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.vz = GC_Vz_net(d_model, hidden_dim).to(device)
        self.vz_target = GC_Vz_net(d_model, hidden_dim).to(device)
        self.vz_target.load_state_dict(self.vz.state_dict())

        # Latent proposer (tau): (z, z_g) -> proposed next latent. Control-side proto-tau.
        self.latent_proposer = nn.Sequential(
            nn.Linear(2*d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        ).to(device)

        self.inv_dynamics = inv_dynamics_net(d_model, hidden_dim, action_dim).to(device)

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.vz.parameters()), lr=self.critic_lr
        )
        self.env_optimizer = torch.optim.Adam(
            list(self.latent_proposer.parameters()) + list(self.inv_dynamics.parameters()), lr=self.env_lr
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lambda step: min(step / (self.warmup_steps * 1.0), 1.0))
        self.env_scheduler = torch.optim.lr_scheduler.LambdaLR(self.env_optimizer, lambda step: min(step / (self.warmup_steps * 1.0), 1.0))

        self.offline_checkpoint_path = offline_checkpoint_path
        self.online_checkpoint_path = online_checkpoint_path
        # Optional: load student encoder instead of teacher (e.g. after offline distillation). Student also frozen.
        self.student_checkpoint_path = agent_kwargs.get('student_checkpoint_path', None)

        self.load_checkpoint()
        self.encoder.eval()  # Keep encoder in eval (dropout off); never train online.

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

    def select_action(self, state, goal, epsilon=0.05, action_noise=0.1, eval_mode=False):
        """Action a = I(z, tau(z,z_g), z_g). Encoder used inference-only (no_grad)."""
        with torch.no_grad():
            self.encoder.eval()
            state_norm = self.normalize_observation(state)
            goal_norm = self.normalize_observation(goal)

            # Temporary single-state packing: zero-pad left, put obs only in last slot; until rolling history buffer.
            if len(state_norm.shape) == 1:
                state_seq = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
                state_seq[-1] = state_norm
                goal_seq = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
                goal_seq[-1] = goal_norm
                mask = np.zeros(self.context_length, dtype=np.float32)
                mask[-1] = 1.0
            else:
                state_seq = state_norm
                goal_seq = goal_norm
                mask = np.ones(self.context_length, dtype=np.float32)

            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            goal_tensor = torch.FloatTensor(goal_seq).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

            z = self.encoder.encode_last_valid(state_tensor, mask_tensor)
            goal_z = self.encoder.encode_last_valid(goal_tensor, mask_tensor)

            if eval_mode:
                epsilon = 0.0
                action_noise = 0.0

            if np.random.random() < epsilon:
                # Heuristic exploration: goal-direction interpolation (not full D3G tau optimization).
                alpha = np.random.uniform(0.2, 1.0)
                z_prime = z + alpha * (goal_z - z)
                action_pred = self.inv_dynamics(z, z_prime, goal_z)
                action = action_pred.squeeze(0).cpu().numpy()
            else:
                z_proposed = self.latent_proposer(torch.cat([z, goal_z], dim=1))
                action_pred = self.inv_dynamics(z, z_proposed, goal_z)
                action = action_pred.squeeze(0).cpu().numpy()

            action_noise = np.random.randn(self.action_dim) * action_noise
            action = action + action_noise
            action = np.clip(action, -1.0, 1.0)

            return action.astype(np.float32)

    def load_checkpoint(self):
        # Encoder: optional student (offline-distilled) else teacher from checkpoint. Always frozen and eval.
        encoder_loaded = False
        if getattr(self, 'student_checkpoint_path', None) and os.path.exists(self.student_checkpoint_path):
            ckpt = torch.load(self.student_checkpoint_path, weights_only=False)
            self.encoder.load_state_dict(ckpt.get('encoder', ckpt), strict=False)
            encoder_loaded = True
            print(f"Encoder loaded from student checkpoint (frozen): {self.student_checkpoint_path}")
        if not os.path.exists(self.online_checkpoint_path):
            if os.path.exists(self.offline_checkpoint_path):
                print(f"오프라인 체크포인트 로드: {self.offline_checkpoint_path}")
                checkpoint = torch.load(self.offline_checkpoint_path, weights_only=False)
                if not encoder_loaded:
                    self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                self.critic.load_state_dict(checkpoint.get('critic') or checkpoint.get('critic_target'))
                self.vz.load_state_dict(checkpoint.get('vz') or checkpoint.get('vz_target'))
                _lp = checkpoint.get('latent_proposer') or checkpoint.get('GC_next_state_estimator')
                if _lp:
                    self.latent_proposer.load_state_dict(_lp)
                if hasattr(self, 'critic_target'):
                    self.critic_target.load_state_dict(self.critic.state_dict())
                if hasattr(self, 'vz_target'):
                    self.vz_target.load_state_dict(self.vz.state_dict())

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
                        if not encoder_loaded:
                            self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                        self.critic.load_state_dict(checkpoint.get('critic') or checkpoint.get('critic_target'))
                        self.vz.load_state_dict(checkpoint.get('vz') or checkpoint.get('vz_target'))
                        _lp = checkpoint.get('latent_proposer') or checkpoint.get('GC_next_state_estimator')
                        if _lp:
                            self.latent_proposer.load_state_dict(_lp)
                        if hasattr(self, 'critic_target'):
                            self.critic_target.load_state_dict(self.critic.state_dict())
                        if hasattr(self, 'vz_target'):
                            self.vz_target.load_state_dict(self.vz.state_dict())
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
                            if not encoder_loaded:
                                self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
                            self.critic.load_state_dict(checkpoint.get('critic') or checkpoint.get('critic_target'))
                            self.vz.load_state_dict(checkpoint.get('vz') or checkpoint.get('vz_target'))
                            _lp = checkpoint.get('latent_proposer') or checkpoint.get('GC_next_state_estimator')
                            if _lp:
                                self.latent_proposer.load_state_dict(_lp)
                            if hasattr(self, 'critic_target'):
                                self.critic_target.load_state_dict(self.critic.state_dict())
                            if hasattr(self, 'vz_target'):
                                self.vz_target.load_state_dict(self.vz.state_dict())

                            # 옵티마이저와 스케줄러는 새로 초기화 (온라인 학습용)
                            print("옵티마이저와 스케줄러는 새로 초기화됩니다.")
                            

                        else:
                            print("체크포인트를 찾을 수 없습니다. 랜덤 초기화로 시작합니다.")
                else:
                    print("오프라인 체크포인트 디렉토리를 찾을 수 없습니다. 랜덤 초기화로 시작합니다.")
        else:
            print(f"온라인 체크포인트 로드: {self.online_checkpoint_path}")
            checkpoint = torch.load(self.online_checkpoint_path, weights_only=False)
            if not encoder_loaded:
                self.encoder.load_state_dict(checkpoint['encoder'], strict=False)
            self.critic.load_state_dict(checkpoint.get('critic') or checkpoint.get('critic_target'))
            self.vz.load_state_dict(checkpoint.get('vz') or checkpoint.get('vz_target'))
            if hasattr(self, 'critic_target'):
                self.critic_target.load_state_dict(checkpoint.get('critic_target') or checkpoint.get('critic') or self.critic.state_dict())
            if hasattr(self, 'vz_target'):
                self.vz_target.load_state_dict(checkpoint.get('vz_target') or checkpoint.get('vz') or self.vz.state_dict())
            _lp = checkpoint.get('latent_proposer') or checkpoint.get('GC_next_state_estimator')
            if _lp:
                self.latent_proposer.load_state_dict(_lp)
            self.inv_dynamics.load_state_dict(checkpoint['inv_dynamics'])
            # 옵티마이저와 스케줄러는 온라인 체크포인트에서 로드
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.env_optimizer.load_state_dict(checkpoint['env_optimizer'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
            self.env_scheduler.load_state_dict(checkpoint['env_scheduler'])

            # 정규화 통계 로드
            if 'normalize_stats' in checkpoint:
                self.normalize_stats = checkpoint['normalize_stats']
                print(f"정규화 통계 로드 완료: mean shape {self.normalize_stats['mean'].shape}, std shape {self.normalize_stats['std'].shape}")



            self.epoch = checkpoint.get('epoch', checkpoint.get('episode', 0))
            self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('best_success_rate', 0.0))

    def update(self, states, actions, next_states, rewards, dones, mask, goal_obs):
        """
        Right-aligned: valid on the right, padding on the left. Online learning is in latent space only:
        z, zp, z_g are computed with encoder in eval mode and no_grad (detached); only critic, Vz, proposer, inv_dynamics are updated.
        """
        device = self.device
        states = states.to(device)
        next_states = next_states.to(device)
        rewards_full = rewards.to(device)
        dones_full = dones.to(device)
        mask_full = mask.to(device)
        actions_full = actions.to(device)
        last_valid_idx = last_valid_index_from_mask(mask_full)
        rewards_last = rewards_full.gather(1, last_valid_idx.unsqueeze(1)).float()
        dones_last = dones_full.gather(1, last_valid_idx.unsqueeze(1)).float()
        actions_last = actions_full.gather(1, last_valid_idx.view(-1, 1, 1).expand(-1, 1, actions_full.size(-1))).squeeze(1)
        mask_last = (mask_full.sum(dim=1, keepdim=True) > 0).float()

        if goal_obs.dim() == 2:
            goal_obs_full = torch.zeros(states.size(0), states.size(1), goal_obs.size(-1), device=device, dtype=states.dtype)
            goal_obs_full[:, -1, :] = goal_obs.to(device)
            goal_mask_full = torch.zeros_like(mask_full)
            goal_mask_full[:, -1] = 1.0
        else:
            goal_obs_full = goal_obs.to(device)
            goal_mask_full = torch.ones_like(mask_full)

        # Encoder inference-only: eval + no_grad so z, zp, z_g are detached; online learning only in control heads.
        self.encoder.eval()
        with torch.no_grad():
            z   = self.encoder.encode_last_valid(states, mask_full)
            zp  = self.encoder.encode_last_valid(next_states, mask_full)
            z_g = self.encoder.encode_last_valid(goal_obs_full, goal_mask_full)
        # TODO: optional latent replay cache — store precomputed (z, zp, z_g) in replay to skip encoder forward at update time.

        with torch.no_grad():
            v_target_next = self.vz_target(zp, z_g)
            target_q = rewards_last + self.gamma * (1.0 - dones_last) * v_target_next

        q_current1, q_current2 = self.critic(z, zp, z_g)
        if mask_last.sum() > 0:
            w = mask_last
            critic_l1 = F.smooth_l1_loss(q_current1, target_q, reduction='none')
            critic_l2 = F.smooth_l1_loss(q_current2, target_q, reduction='none')
            td_loss = ((critic_l1 * w).sum() + (critic_l2 * w).sum()) / w.sum().clamp(min=1.0)
        else:
            td_loss = F.smooth_l1_loss(q_current1, target_q) + F.smooth_l1_loss(q_current2, target_q)

        v_current = self.vz(z, z_g)
        with torch.no_grad():
            q_for_v1, q_for_v2 = self.critic_target(z, zp, z_g)
            q_for_v = torch.min(q_for_v1, q_for_v2)
        v_loss = expectile_loss(v_current, q_for_v, self.expectile_tau)
        current_q = torch.min(q_current1, q_current2)

        self.critic_optimizer.zero_grad()
        (td_loss + v_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # EMA target updates
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.vz.parameters(), self.vz_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        # Proposer (proto-tau): predictive reg + Q-max. TODO: closer-to-D3G would add proposer_target + cycle consistency or Q-guided forward model.
        for p in self.critic.parameters():
            p.requires_grad = False
        zp_proposed = self.latent_proposer(torch.cat([z, z_g], dim=1))
        q_proposed1, q_proposed2 = self.critic(z, zp_proposed, z_g)
        q_proposed = torch.min(q_proposed1, q_proposed2)
        predictive_reg = F.mse_loss(zp_proposed, zp)
        q_max_term = -self.lmbda * q_proposed.mean()
        proposer_loss = predictive_reg + q_max_term
        for p in self.critic.parameters():
            p.requires_grad = True

        # Advantage-weighted inverse dynamics; stable weights exp(adv/temp) capped to avoid overflow.
        a_pred = self.inv_dynamics(z, zp, z_g)
        adv = (current_q - v_current).detach().clamp(min=-5.0, max=5.0)
        w_adv = torch.exp(adv / self.adv_temp).clamp(max=self.adv_cap)
        w_adv = w_adv * mask_last
        a_mse = F.mse_loss(a_pred, actions_last, reduction='none').mean(dim=1, keepdim=True)
        if w_adv.sum() > 1e-8:
            action_loss = (a_mse * w_adv).sum() / w_adv.sum().clamp(min=1e-8)
        else:
            action_loss = (a_mse * w_adv).mean()

        self.env_optimizer.zero_grad()
        (proposer_loss + action_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.latent_proposer.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.inv_dynamics.parameters(), max_norm=1.0)
        self.env_optimizer.step()
        self.env_scheduler.step()

        return {
            'critic_loss': td_loss.item(),
            'action_loss': action_loss.item(),
            'v_loss': v_loss.item(),
            'proposer_loss': proposer_loss.item(),
            'state_loss': proposer_loss.item(),  # alias for logging
        }