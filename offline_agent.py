"""
Offline latent pretraining for sparse goal-conditioned RL.
Implicit latent QSS / proto-D3G: Qzz = transition-value, Vz = reachability proxy,
latent proposer = proto-tau (not full D3G cycle-consistent proposer).
"""
import torch
from torch import nn
import torch.nn.functional as F
from model import Mamba_Encoder, GC_Qzz_net, GC_Vz_net
from utils import expectile_loss, last_valid_index_from_mask


class InfoNCEManager:
    """InfoNCE loss와 메모리뱅크 관리를 위한 클래스"""

    def __init__(self, d_model, memory_bank_size=10000, temperature=0.1,
                 use_bidirectional=True, device="cuda"):
        self.d_model = d_model
        self.memory_bank_size = memory_bank_size
        self.temperature = temperature
        self.use_bidirectional = use_bidirectional
        self.device = device

        self.memory_bank = torch.randn(memory_bank_size, d_model, device=device)
        self.memory_bank = F.normalize(self.memory_bank, p=2, dim=1)
        self.memory_ptr = 0
        self.bank_count = 0  # number of slots ever written (capped at memory_bank_size); use this, not ptr, for "full"
        self.bank_full = False

    def update_memory_bank(self, z, zp):
        with torch.no_grad():
            embeddings = torch.cat([z, zp], dim=0)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            n = embeddings.shape[0]
            for i in range(n):
                self.memory_bank[self.memory_ptr] = embeddings[i]
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank_size
            self.bank_count = min(self.memory_bank_size, self.bank_count + n)
            self.bank_full = self.bank_count >= self.memory_bank_size

    def compute_infonce_loss(self, z, zp):
        """
        InfoNCE: use_bidirectional when warm; only use valid prefix of memory bank (bank_count), not ptr.
        """
        z_norm = F.normalize(z, p=2, dim=1)
        zp_norm = F.normalize(zp, p=2, dim=1)
        if self.use_bidirectional:
            loss_z_to_zp = self._compute_single_direction_nce(z_norm, zp_norm)
            loss_zp_to_z = self._compute_single_direction_nce(zp_norm, z_norm)
            nce_loss = (loss_z_to_zp + loss_zp_to_z) / 2.0
        else:
            nce_loss = self._compute_single_direction_nce(z_norm, zp_norm)
        return nce_loss

    def _compute_single_direction_nce(self, anchor, positive):
        batch_size = anchor.shape[0]
        sim_matrix = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(batch_size, device=anchor.device)
        n_valid = self.bank_count
        if n_valid > 0:
            bank = self.memory_bank[:n_valid].detach()
            memory_sim = (anchor @ bank.T) / self.temperature
            all_negatives = torch.cat([sim_matrix, memory_sim], dim=1)
            loss = F.cross_entropy(all_negatives, labels)
        else:
            loss = F.cross_entropy(sim_matrix, labels)
        return loss

class Offline_Encoder:
    def __init__(self, state_dim, hidden_dim,
    context_length, d_model, d_state, d_conv, expand,
    tau = 0.01, gamma = 0.99, expectile_tau = 0.3,
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

        # Latent proposer (proto-tau): (z, z_g) -> proposed next latent. Not full D3G cycle-consistent.
        self.latent_proposer = nn.Sequential(
            nn.Linear(2*d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        ).to(self.device)
        self.latent_proposer_target = nn.Sequential(
            nn.Linear(2*d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        ).to(self.device)
        self.latent_proposer_target.load_state_dict(self.latent_proposer.state_dict())

        # GC success/done head: BCE on dones (sparse success signal).
        self.GC_success_head = nn.Sequential(
            nn.Linear(2*d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        # Target networks fixed
        for param in self.encoder_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        for param in self.latent_proposer_target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                          list(self.latent_proposer.parameters()) +
                                          list(self.GC_success_head.parameters()) +
                                          list(self.critic.parameters()) +
                                          list(self.vz.parameters()), lr=self.encoder_lr)

        # Warmup schedulers
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / warmup_steps)  # 1000 steps 동안 warmup
        )

        self.update_num = 0

        # EMA for pos_weight (안정적인 클래스 불균형 처리)
        self.pos_weight_ema = torch.tensor(1.0, device=device)
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
        states, next_states: (B, K, state_dim). rewards, dones: (B, K).
        goal_obs: (B, D) single vector (preferred) or (B, K, D); if 2D, packed as last-slot-only inside.
        Right-aligned: valid on the right; we gather last-valid reward/done from mask.
        """
        if mask is None:
            mask = torch.ones(states.shape[0], states.shape[1], device=states.device, dtype=torch.float32)
        mask = mask.to(self.device)
        last_valid_idx = last_valid_index_from_mask(mask)  # (B,)
        rewards = rewards.float().to(self.device)
        if dones is None:
            dones = (rewards > 0.5).float().to(self.device)
        else:
            dones = dones.to(self.device)
        rewards_last = rewards.gather(1, last_valid_idx.unsqueeze(1))
        dones_last = dones.gather(1, last_valid_idx.unsqueeze(1))
        mask_last = (mask.sum(dim=1, keepdim=True) > 0).float().to(self.device)

        self.encoder_target.eval()
        self.critic_target.eval()
        self.latent_proposer_target.eval()

        if goal_obs is None:
            goal_z = torch.zeros(states.shape[0], self.d_model, device=self.device)
            goal_z_target = torch.zeros(states.shape[0], self.d_model, device=self.device)
        elif goal_obs.dim() == 2:
            # Goal as single vector (B, D): pack as state-like, only last slot valid
            B, L = states.size(0), states.size(1)
            goal_packed = torch.zeros(B, L, goal_obs.size(-1), device=self.device, dtype=goal_obs.dtype)
            goal_packed[:, -1, :] = goal_obs.to(self.device)
            goal_mask = torch.zeros(B, L, device=self.device)
            goal_mask[:, -1] = 1.0
            goal_z = self.encoder.encode_last_valid(goal_packed, goal_mask)
            goal_z_target = self.encoder_target.encode_last_valid(goal_packed, goal_mask)
        else:
            goal_z = self.encoder.encode_last_valid(goal_obs, mask)
            goal_z_target = self.encoder_target.encode_last_valid(goal_obs, mask)

        z  = self.encoder.encode_last_valid(states, mask)
        zp = self.encoder.encode_last_valid(next_states, mask)
        with torch.no_grad():
            z_target  = self.encoder_target.encode_last_valid(states, mask)
            zp_target = self.encoder_target.encode_last_valid(next_states, mask)

        zp_estimated = self.latent_proposer(torch.cat([z, goal_z], dim=1))

        # Success as absorbing: d_eff = max(done, reward) so target fits in [0,1] with sigmoid Q/V
        dones_eff = torch.maximum(dones_last, rewards_last)
        with torch.no_grad():
            v_target_next = self.vz(zp_target, goal_z_target)
            target_q = rewards_last + self.gamma * (1.0 - dones_eff) * v_target_next

        q_current1, q_current2 = self.critic(z, zp, goal_z)

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
        # --- V loss (둘 다 (B,1)); expectile_tau < 0.5 → 비관적, > 0.5 → 낙관적 ---
        if mask_last is not None:
            w = mask_last
            v_err = expectile_loss(v_current, q_target_current, self.expectile_tau, reduction='none')  # (B,1)
            v_loss = (v_err * w).sum() / w.sum().clamp(min=1.0)
        else:
            v_loss = expectile_loss(v_current, q_target_current, self.expectile_tau)

        # GC_success_head: predict goal-reaching (rewards_last), not trajectory boundary (dones_last)
        reward_logits = self.GC_success_head(torch.cat([zp, goal_z], dim=1))

        # Proposer: predictive pretraining (MSE to next latent). Not full cycle consistency.
        state_err = F.mse_loss(zp_estimated, zp_target, reduction='none').mean(dim=1, keepdim=True)
        if mask_last is not None:
            state_loss = (state_err * mask_last).sum() / mask_last.sum().clamp(min=1.0)
        else:
            state_loss = state_err.mean()

        if self.use_focal_loss:
            raw_bce = self._focal_loss(reward_logits, rewards_last, self.focal_alpha, self.focal_gamma, reduction='none')
        else:
            self.pos_weight_ema = (
                self.ema_momentum * self.pos_weight_ema.to(self.device)
                + (1 - self.ema_momentum) * self._compute_pos_weight(rewards_last).detach()
            ).clamp_(0.1, 100.0)
            raw_bce = F.binary_cross_entropy_with_logits(
                reward_logits, rewards_last, pos_weight=self.pos_weight_ema.to(self.device), reduction='none'
            )
        if mask_last is not None:
            reward_loss = (raw_bce * mask_last).sum() / mask_last.sum().clamp(min=1.0)
        else:
            reward_loss = raw_bce.mean()

        nce_loss = torch.tensor(0.0, device=self.device)
        if self.beta_nce > 0:
            min_bank_for_use = min(512, self.infonce_manager.memory_bank_size // 2)
            use_bank = self.infonce_manager.bank_count >= min_bank_for_use
            if use_bank:
                nce_loss = self.infonce_manager.compute_infonce_loss(z, zp)
            else:
                nce_loss = self.infonce_manager._compute_single_direction_nce(
                    F.normalize(z, p=2, dim=1), F.normalize(zp, p=2, dim=1))
        
        # --- Alignment ---
        if goal_obs is not None:
            eps = 1e-8
            cos = ( (zp - z) * (goal_z - z) ).sum(dim=1) / (
                (zp - z).norm(dim=1) * (goal_z - z).norm(dim=1) + eps )
            align = -cos.unsqueeze(1)  # (B,1)
            alignment_loss = (align * mask_last).sum() / mask_last.sum().clamp(min=1.0) if mask_last is not None else align.mean()
        else:
            alignment_loss = torch.tensor(0.0, device=self.device)

        total_loss = critic_loss + self.beta_s * state_loss + self.beta_r * reward_loss + self.beta_nce * nce_loss + self.beta_v * v_loss + self.beta_a * alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.latent_proposer.parameters()) +
            list(self.GC_success_head.parameters()) +
            list(self.critic.parameters()) +
            list(self.vz.parameters()),
            max_norm=5.0
        )
        self.optimizer.step()
        self.scheduler.step()

        if self.beta_nce > 0:
            self.infonce_manager.update_memory_bank(z, zp)

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.encoder.parameters(), self.encoder_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.latent_proposer.parameters(), self.latent_proposer_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        self.update_num += 1

        return {
            'total_loss': float(total_loss.detach().item()),
            'critic_loss': float(critic_loss.detach().item()),
            'v_loss': float(v_loss.detach().item()),
            'state_loss': float(state_loss.detach().item()),
            'reward_loss': float(reward_loss.detach().item()),
            'nce_loss': float(nce_loss.detach().item()),
            'alignment_loss': float(alignment_loss.detach().item()),
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