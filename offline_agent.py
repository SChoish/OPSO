import torch
from torch import nn
import torch.nn.functional as F
from model import Mamba_Encoder, GC_Qzz_net, GC_Vz_net


class InfoNCEManager:
    """InfoNCE loss와 메모리뱅크 관리를 위한 클래스"""

    def __init__(self, d_model, memory_bank_size=10000, temperature=0.1,
                 use_bidirectional=True, device="cuda"):
        self.d_model = d_model
        self.memory_bank_size = memory_bank_size
        self.temperature = temperature
        self.use_bidirectional = use_bidirectional
        self.device = device

        # Memory bank for InfoNCE
        self.memory_bank = torch.randn(memory_bank_size, d_model, device=device)
        self.memory_bank = F.normalize(self.memory_bank, p=2, dim=1)
        self.memory_ptr = 0

    def update_memory_bank(self, z, zp):
        """메모리뱅크 업데이트"""
        with torch.no_grad():
            # 현재 배치의 임베딩을 메모리뱅크에 추가
            batch_size = z.shape[0]
            embeddings = torch.cat([z, zp], dim=0)           # (2B,d)
            embeddings = F.normalize(embeddings, p=2, dim=1) # 정규화 후 저장
            for i in range(embeddings.shape[0]):
                self.memory_bank[self.memory_ptr] = embeddings[i]
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank_size

    def compute_infonce_loss(self, z, zp):
        """
        개선된 InfoNCE loss for contrastive learning
        - 양방향 InfoNCE (z→zp, zp→z)
        - 메모리뱅크 활용
        - 온도 파라미터 조절 가능
        """
        batch_size = z.shape[0]

        # L2 normalize embeddings
        z_norm = F.normalize(z, p=2, dim=1)
        zp_norm = F.normalize(zp, p=2, dim=1)

        if self.use_bidirectional:
            # 양방향 InfoNCE: z→zp와 zp→z의 평균
            loss_z_to_zp = self._compute_single_direction_nce(z_norm, zp_norm)
            loss_zp_to_z = self._compute_single_direction_nce(zp_norm, z_norm)
            nce_loss = (loss_z_to_zp + loss_zp_to_z) / 2.0
        else:
            # 단방향 InfoNCE: z→zp만
            nce_loss = self._compute_single_direction_nce(z_norm, zp_norm)

        return nce_loss

    def _compute_single_direction_nce(self, anchor, positive):
        """
        단방향 InfoNCE 계산
        anchor: anchor embeddings (B, d_model)
        positive: positive embeddings (B, d_model)
        """
        batch_size = anchor.shape[0]

        # In-batch negatives
        sim_matrix = torch.matmul(anchor, positive.T) / self.temperature  # (B, B)

        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=anchor.device)

        # 메모리뱅크에서 추가 negatives 가져오기
        if self.memory_bank_size > 0:
            # 메모리뱅크와의 유사도 계산
            bank = self.memory_bank.detach()
            memory_sim = torch.matmul(anchor, bank.T) / self.temperature  # (B, memory_size)

            # 배치 내 negatives와 메모리뱅크 negatives 결합
            all_negatives = torch.cat([sim_matrix, memory_sim], dim=1)  # (B, B + memory_size)

            # Positive는 여전히 대각선 (배치 내에서만)
            # 메모리뱅크는 모두 negative로 취급
            loss = F.cross_entropy(all_negatives, labels)
        else:
            # 메모리뱅크 없이 배치 내 negatives만 사용
            loss = F.cross_entropy(sim_matrix, labels)

        return loss

class Offline_Encoder:
    def __init__(self, state_dim, hidden_dim,
    context_length, d_model, d_state, d_conv, expand,
    tau = 0.01, gamma = 0.99, expectile_tau = 0.9,
    device = "cuda",
    encoder_lr = 3e-4, n_layers = 1, warmup_steps = 1000,
    drop_p = 0.1, beta_s = 1.0, beta_r = 1.0, beta_nce = 0.1, beta_v = 0.1, beta_a = 0.1,
    use_focal_loss = False, focal_alpha = 0.25, focal_gamma = 2.0,
    nce_temperature = 0.1, memory_bank_size = 10000, use_bidirectional_nce = True,
):

        # Environment Parameters
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Mamba Parameters
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers

        # Hyperparameters
        self.warmup_steps = warmup_steps
        self.encoder_lr = encoder_lr
        self.beta_s = beta_s
        self.beta_r = beta_r
        self.beta_nce = beta_nce
        self.beta_v = beta_v
        self.beta_a = beta_a
        self.tau = tau
        self.gamma = gamma
        self.expectile_tau = expectile_tau

        # Auxiliary Losses
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.nce_temperature = nce_temperature
        self.memory_bank_size = memory_bank_size
        self.use_bidirectional_nce = use_bidirectional_nce

        # Device
        self.device = device

        # Model
        self.encoder = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        self.encoder_target = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.critic = GC_Qzz_net(d_model, hidden_dim, double_q = True).to(device)
        self.critic_target = GC_Qzz_net(d_model, hidden_dim, double_q = True).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.vz = GC_Vz_net(d_model, hidden_dim).to(device)

        # Auxiliary Models
        self.next_state_estimator = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        ).to(self.device)

        self.next_state_estimator_target = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        ).to(self.device)
        self.next_state_estimator_target.load_state_dict(self.next_state_estimator.state_dict())

        self.rewards_estimator = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        # Target networks 고정
        for param in self.encoder_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        for param in self.next_state_estimator_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                          list(self.next_state_estimator.parameters()) +
                                          list(self.rewards_estimator.parameters()) +
                                          list(self.critic.parameters()) +
                                          list(self.vz.parameters()), lr=self.encoder_lr)

        # Warmup schedulers
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / warmup_steps)  # 1000 steps 동안 warmup
        )

        self.update_num = 0

        # EMA for pos_weight (안정적인 클래스 불균형 처리)
        self.pos_weight_ema = 1.0
        self.ema_momentum = 0.99

        # InfoNCE Manager
        self.infonce_manager = InfoNCEManager(
            d_model=d_model,
            memory_bank_size=memory_bank_size,
            temperature=nce_temperature,
            use_bidirectional=use_bidirectional_nce,
            device=device
        )



    def update(self, states, next_states, rewards, dones=None, mask=None, goal_obs=None):
        """
        states, next_states, goal_obs: (B, K, state_dim)  # K == context_length
        rewards, dones: (B, 1)
        """
        # --- 타입/장치 정리 ---
        rewards = rewards.float().to(self.device)
        if dones is None:
            # 희소 보상일 때 임시 대체 (가능하면 호출측에서 dones 제공 권장)
            dones = (rewards > 0.5).float()
        dones = dones.to(self.device)

        if goal_obs is None:
            goal_z = torch.zeros(states.shape[0], self.d_model, device=self.device)
            goal_z_target = torch.zeros(states.shape[0], self.d_model, device=self.device)
        else:
            goal_z = self.encoder.encode_last_valid(goal_obs, mask)  # (B, d)
            goal_z_target = self.encoder_target.encode_last_valid(goal_obs, mask)  # (B, d)

        # --- 인코딩 (현재 네트워크) ---
        z  = self.encoder.encode_trajectory(states, mask)        # (B, d)
        zp = self.encoder.encode_trajectory(next_states, mask)   # (B, d)

        # --- 인코딩 (타깃 네트워크, 그래프 미연결) ---
        with torch.no_grad():
            z_target = self.encoder_target.encode_trajectory(states, mask)        # z_{t}
            zp_target  = self.encoder_target.encode_trajectory(next_states, mask)        # z_{t+1}

        # --- 1-step latent consistency ---
        zp_estimated = self.next_state_estimator(z)

        # --- 타깃 Q 계산 ---
        with torch.no_grad():
            # 모델 기반 1-step 롤아웃
            v_target_next = self.vz(zp_target, goal_z_target)  # (B,1)
            target_q = rewards + self.gamma * (1.0 - dones) * v_target_next  # (B,1)

        # --- 현재 Q ---
        q_current1, q_current2 = self.critic(z, zp, goal_z)  # (B,1)

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

        v_current = self.vz(z, goal_z)  # (B,1)
        with torch.no_grad():
            q_target1, q_target2 = self.critic_target(z_target, zp_target, goal_z_target)  # (B,1)
            q_target_current = torch.min(q_target1, q_target2)  # (B,1)
        # --- V loss (둘 다 (B,1)) ---
        if mask_last is not None:
            w = mask_last
            v_err = self._expectile_loss(v_current, q_target_current, self.expectile_tau, reduction='none')  # (B,1)
            v_loss = (v_err * w).sum() / w.sum().clamp(min=1.0)
        else:
            v_loss = self._expectile_loss(v_current, q_target_current, self.expectile_tau)

        # 보상/종결 예측(여기서는 dones를 타깃으로 사용)
        reward_logits = self.rewards_estimator(zp)  # (B,1)

        # --- State loss (둘 다 (B,d)) -> (B,1)로 평균 후 mask_last 적용 ---
        state_err = F.mse_loss(zp_estimated, zp_target, reduction='none').mean(dim=1, keepdim=True)  # (B,1)
        if mask_last is not None:
            state_loss = (state_err * mask_last).sum() / mask_last.sum().clamp(min=1.0)
        else:
            state_loss = state_err.mean()

        # --- Reward(Done) BCE (둘 다 (B,1)) ---
        if self.use_focal_loss:
            raw_bce = self._focal_loss(reward_logits, dones, self.focal_alpha, self.focal_gamma, reduction='none')  # (B,1)
        else:
            # pos_weight_ema는 텐서 유지 권장
            self.pos_weight_ema = (
                self.ema_momentum * self.pos_weight_ema
                + (1 - self.ema_momentum) * self._compute_pos_weight(dones).detach()
            ).clamp_(0.1, 100.0)
            raw_bce = F.binary_cross_entropy_with_logits(
                reward_logits, dones, pos_weight=self.pos_weight_ema, reduction='none'
            )
        if mask_last is not None:
            reward_loss = (raw_bce * mask_last).sum() / mask_last.sum().clamp(min=1.0)
        else:
            reward_loss = raw_bce.mean()

        # InfoNCE (양방향 + 뱅크 결합은 내부 함수가 처리)
        nce_loss = 0.0
        if self.beta_nce > 0:
            nce_loss = self.infonce_manager.compute_infonce_loss(z, zp)

        # === Alignment (★ goal이 있을 때만) ===
        if goal_obs is not None:
            d_z     = zp - z               # (B,d)
            goal_dir= (goal_z - z)         # (B,d)
            # 안정화용 eps
            eps     = 1e-8
            num     = (d_z * goal_dir).sum(dim=1)
            den     = (d_z.norm(dim=1) * goal_dir.norm(dim=1)).clamp_min(eps)
            alignment_loss = -(num / den).mean()
        else:
            alignment_loss = torch.tensor(0.0, device=self.device)

        total_loss = critic_loss + self.beta_s * state_loss + self.beta_r * reward_loss + self.beta_nce * nce_loss + self.beta_v * v_loss + self.beta_a * alignment_loss

        # --- 옵티마이즈 ---
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.next_state_estimator.parameters()) +
            list(self.rewards_estimator.parameters()) +
            list(self.critic.parameters()) +
            list(self.vz.parameters()),
            max_norm=5.0
        )
        self.optimizer.step()
        self.scheduler.step()

        # --- (중요) backward 이후에 메모리뱅크 업데이트 ---
        # NCE 메모리뱅크: backward 전에 in-place 금지 → 여기서 갱신
        if self.beta_nce > 0:
            self.infonce_manager.update_memory_bank(z, zp)



        # --- 타깃 네트 EMA ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.next_state_estimator.parameters(), self.next_state_estimator_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.update_num += 1

        return {
            'total_loss': float(total_loss.detach().item()),
            'critic_loss': float(critic_loss.detach().item()),
            'state_loss': float(state_loss.detach().item()),
            'reward_loss': float(reward_loss.detach().item()),
            'nce_loss': float(nce_loss.detach().item()) if isinstance(nce_loss, torch.Tensor) else nce_loss,
        }

    def _compute_pos_weight(self, rewards):
        """
        클래스 불균형을 위한 positive weight 계산
        pos_weight = (negative_samples / positive_samples)
        """
        batch_size = rewards.shape[0]
        positive_count = rewards.sum().item()
        negative_count = batch_size - positive_count

        if positive_count == 0:
            return torch.tensor(1.0, device=rewards.device)

        pos_weight = negative_count / positive_count
        return torch.tensor(pos_weight, device=rewards.device)

    def _expectile_loss(self, pred, target, tau, reduction='mean'):
        """
        Expectile regression loss
        tau: expectile level (0 < tau < 1)
        - tau = 0.5: MSE loss (mean)
        - tau > 0.5: upper expectile (optimistic)
        - tau < 0.5: lower expectile (pessimistic)
        """
        diff = pred - target
        weight = torch.where(diff >= 0, tau, 1 - tau)
        loss = weight * (diff ** 2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def _focal_loss(self, logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance
        """
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def get_current_lr(self):
        """현재 learning rate 반환"""
        return {
            'lr': self.scheduler.get_last_lr()[0]
        }