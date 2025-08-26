import torch
from torch import nn
import torch.nn.functional as F
from model import Mamba_Encoder, Qzz_net

class Offline_Encoder:
    def __init__(self, state_dim, hidden_dim, 
    context_length, d_model, d_state, d_conv, expand, 
    tau = 0.01, gamma = 0.99,
    device = "cuda",
    encoder_lr = 3e-4, n_layers = 1, warmup_steps = 1000,
    drop_p = 0.1, beta_s = 1.0, beta_r = 1.0, beta_nce = 0.1,
    use_focal_loss = False, focal_alpha = 0.25, focal_gamma = 2.0,
    nce_temperature = 0.1, memory_bank_size = 10000, use_bidirectional_nce = True,
    use_knn_target = True, knn_k = 5, knn_memory_size = 50000,
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
        self.warmup_steps = warmup_steps
        self.encoder_lr = encoder_lr
        self.beta_s = beta_s
        self.beta_r = beta_r
        self.beta_nce = beta_nce
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.nce_temperature = nce_temperature
        self.memory_bank_size = memory_bank_size
        self.use_bidirectional_nce = use_bidirectional_nce
        self.use_knn_target = use_knn_target
        self.knn_k = knn_k
        self.knn_memory_size = knn_memory_size

        # Hyperparameters
        self.tau = tau
        self.gamma = gamma
        
        # Device
        self.device = device

        # Model
        self.encoder = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        self.encoder_target = Mamba_Encoder(state_dim, context_length, d_model, d_state, d_conv, expand, n_layers, drop_p).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        
        self.critic = Qzz_net(d_model, hidden_dim).to(device)
        self.critic_target = Qzz_net(d_model, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

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
                                          list(self.critic.parameters()), lr=self.encoder_lr)
        
        # Warmup schedulers
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lambda step: min(1.0, step / warmup_steps)  # 1000 steps 동안 warmup
        )

        self.update_num = 0
        
        # EMA for pos_weight (안정적인 클래스 불균형 처리)
        self.pos_weight_ema = 1.0
        self.ema_momentum = 0.99
        
        # Memory bank for InfoNCE
        self.memory_bank = torch.randn(memory_bank_size, d_model, device=device)
        self.memory_bank = F.normalize(self.memory_bank, p=2, dim=1)
        self.memory_ptr = 0
        
        # KNN memory bank for target Q computation (paired banks)
        # knn_key_bank_norm: 검색용 (정규화된 z_{t+1})
        self.knn_key_bank_norm = torch.randn(knn_memory_size, d_model, device=device)
        self.knn_key_bank_norm = F.normalize(self.knn_key_bank_norm, p=2, dim=1)
        
        # knn_val_bank_raw: 후보값 입력용 (원본 z_{t+2})
        self.knn_val_bank_raw = torch.randn(knn_memory_size, d_model, device=device)
        
        self.knn_ptr = 0

    def update(self, states, next_states, next_next_states, rewards, dones=None, mask=None):
        """
        states, next_states, next_next_states: (B, K, state_dim)  # K == context_length
        rewards, dones: (B, 1)
        """
        # --- 타입/장치 정리 ---
        rewards = rewards.float().to(self.device)
        if dones is None:
            # 희소 보상일 때 임시 대체 (가능하면 호출측에서 dones 제공 권장)
            dones = (rewards > 0.5).float()
        dones = dones.to(self.device)

        # --- 인코딩 (현재 네트워크) ---
        z  = self.encoder.encode_trajectory(states, mask)        # (B, d)
        zp = self.encoder.encode_trajectory(next_states, mask)   # (B, d)

        # --- 인코딩 (타깃 네트워크, 그래프 미연결) ---
        with torch.no_grad():
            zp_target  = self.encoder_target.encode_trajectory(next_states, mask)        # z_{t+1}
            # 다음-다음 상태 윈도우는 호출측에서 그대로 전달 (길이 정합)
            zpp_target = self.encoder_target.encode_trajectory(next_next_states, mask)   # z_{t+2}

        # --- 1-step latent consistency ---
        zp_estimated = self.next_state_estimator(z)

        # --- 타깃 Q 계산 ---
        with torch.no_grad():
            # KNN 웜업 여부: knn_filled 없으면 0으로 초기화
            if not hasattr(self, "knn_filled"):
                self.knn_filled = 0

            knn_warmup_ok = (
                self.use_knn_target and
                (self.knn_filled >= max(1024, self.knn_k * 4))
            )

            if knn_warmup_ok:
                # 관측 페어 기반: Q(z_{t+1}, z_{t+2})의 max over KNN
                q_target_next = self._compute_knn_target_q(zp_target)  # (B,1)
            else:
                # 폴백: 모델 기반 1-step 롤아웃 (바이어스 있지만 웜업 초기에만)
                zpp_estimated_target = self.next_state_estimator_target(zp_target)
                q1_t, q2_t = self.critic_target(zp_target, zpp_estimated_target)
                q_target_next = torch.min(q1_t, q2_t)  # (B,1)

            target_q = rewards + self.gamma * (1.0 - dones) * q_target_next  # (B,1)

        # --- 현재 Q ---
        q1_current, q2_current = self.critic(z, zp)  # (B,1), (B,1)

        # --- 손실들 (mask 적용) ---
        if mask is not None:
            # mask로 가중 평균
            mask_sum = mask.sum()
            if mask_sum > 0:
                critic_loss = (F.smooth_l1_loss(q1_current, target_q, reduction='none') * mask).sum() / mask_sum + \
                             (F.smooth_l1_loss(q2_current, target_q, reduction='none') * mask).sum() / mask_sum
            else:
                critic_loss = torch.tensor(0.0, device=self.device)
        else:
            critic_loss = F.smooth_l1_loss(q1_current, target_q) + F.smooth_l1_loss(q2_current, target_q)

        # 보상/종결 예측(여기서는 dones를 타깃으로 사용)
        reward_logits = self.rewards_estimator(zp)  # (B,1)
        
        if mask is not None:
            mask_sum = mask.sum()
            if mask_sum > 0:
                state_loss = (F.mse_loss(zp_estimated, zp, reduction='none') * mask).sum() / mask_sum
            else:
                state_loss = torch.tensor(0.0, device=self.device)
        else:
            state_loss = F.mse_loss(zp_estimated, zp)

        if self.use_focal_loss:
            if mask is not None:
                mask_sum = mask.sum()
                if mask_sum > 0:
                    reward_loss = (self._focal_loss(reward_logits, dones, self.focal_alpha, self.focal_gamma, reduction='none') * mask).sum() / mask_sum
                else:
                    reward_loss = torch.tensor(0.0, device=self.device)
            else:
                reward_loss = self._focal_loss(reward_logits, dones, self.focal_alpha, self.focal_gamma)
        else:
            posw = self._compute_pos_weight(dones)
            # EMA로 부드럽게
            self.pos_weight_ema = self.ema_momentum * self.pos_weight_ema + (1 - self.ema_momentum) * posw
            # 과도한 스케일 방지(선택)
            self.pos_weight_ema = torch.clamp(self.pos_weight_ema, 0.1, 100.0)
            pos_weight_tensor = self.pos_weight_ema.detach().clone().to(dones.device)
            
            if mask is not None:
                mask_sum = mask.sum()
                if mask_sum > 0:
                    reward_loss = (F.binary_cross_entropy_with_logits(
                        reward_logits, dones, pos_weight=pos_weight_tensor, reduction='none'
                    ) * mask).sum() / mask_sum
                else:
                    reward_loss = torch.tensor(0.0, device=self.device)
            else:
                reward_loss = F.binary_cross_entropy_with_logits(
                    reward_logits, dones, pos_weight=pos_weight_tensor
                )

        # InfoNCE (양방향 + 뱅크 결합은 내부 함수가 처리)
        nce_loss = 0.0
        if self.beta_nce > 0:
            nce_loss = self._compute_infonce_loss(z, zp)

        total_loss = critic_loss + self.beta_s * state_loss + self.beta_r * reward_loss + self.beta_nce * nce_loss

        # --- 옵티마이즈 ---
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.next_state_estimator.parameters()) +
            list(self.rewards_estimator.parameters()) +
            list(self.critic.parameters()),
            max_norm=5.0
        )
        self.optimizer.step()
        self.scheduler.step()

        # --- (중요) backward 이후에 메모리/페어뱅크 업데이트 ---
        # NCE 메모리뱅크: backward 전에 in-place 금지 → 여기서 갱신
        if self.beta_nce > 0:
            with torch.no_grad():
                # 현재 배치의 임베딩을 메모리뱅크에 추가
                batch_size = z.shape[0]
                embeddings = torch.cat([z, zp], dim=0)           # (2B,d)
                embeddings = F.normalize(embeddings, p=2, dim=1) # 정규화 후 저장
                for i in range(embeddings.shape[0]):
                    self.memory_bank[self.memory_ptr] = embeddings[i]
                    self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank_size

        # KNN 페어뱅크 (관측 z_{t+1}, z_{t+2})
        if self.use_knn_target:
            with torch.no_grad():
                # 검색키는 정규화된 z_{t+1}, 후보값은 원본 z_{t+2}
                zp_norm = F.normalize(zp_target, p=2, dim=1)   # (B,d)
                B = zp_norm.size(0)
                end = min(self.knn_ptr + B, self.knn_memory_size)
                n1 = end - self.knn_ptr
                self.knn_key_bank_norm[self.knn_ptr:end] = zp_norm[:n1]
                self.knn_val_bank_raw[self.knn_ptr:end]  = zpp_target[:n1]
                if n1 < B:  # ring buffer wrap-around
                    n2 = B - n1
                    self.knn_key_bank_norm[0:n2] = zp_norm[n1:]
                    self.knn_val_bank_raw[0:n2]  = zpp_target[n1:]
                self.knn_ptr = (self.knn_ptr + B) % self.knn_memory_size
                # filled 카운터 업데이트
                if not hasattr(self, "knn_filled"):
                    self.knn_filled = 0
                self.knn_filled = min(self.knn_filled + B, self.knn_memory_size)

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
    
    def _update_memory_bank(self, z, zp):
        """메모리뱅크 업데이트"""
        with torch.no_grad():
            # 현재 배치의 임베딩을 메모리뱅크에 추가
            batch_size = z.shape[0]
            embeddings = torch.cat([z.detach(), zp.detach()], dim=0)  # (2B, d_model)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # 메모리뱅크에 순환적으로 저장 (inplace 방지)
            for i in range(embeddings.shape[0]):
                self.memory_bank[self.memory_ptr] = embeddings[i].clone()
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank_size
    
    def _compute_infonce_loss(self, z, zp):
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
        
        if self.use_bidirectional_nce:
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
        sim_matrix = torch.matmul(anchor, positive.T) / self.nce_temperature  # (B, B)
        
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=anchor.device)
        
        # 메모리뱅크에서 추가 negatives 가져오기
        if self.memory_bank_size > 0:
            # 메모리뱅크와의 유사도 계산
            memory_sim = torch.matmul(anchor, self.memory_bank.T) / self.nce_temperature  # (B, memory_size)
            
            # 배치 내 negatives와 메모리뱅크 negatives 결합
            all_negatives = torch.cat([sim_matrix, memory_sim], dim=1)  # (B, B + memory_size)
            
            # Positive는 여전히 대각선 (배치 내에서만)
            # 메모리뱅크는 모두 negative로 취급
            loss = F.cross_entropy(all_negatives, labels)
        else:
            # 메모리뱅크 없이 배치 내 negatives만 사용
            loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def _update_knn_memory(self, zp, zpp):
        """
        KNN 메모리뱅크 업데이트 (정합된 쌍 저장)
        zp: z_{t+1} (검색용 키)
        zpp: z_{t+2} (후보값)
        """
        with torch.no_grad():
            # zp를 정규화하여 검색용 키로 저장 (inplace 방지)
            zp_norm = F.normalize(zp.detach(), p=2, dim=1)
            batch_size = zp_norm.shape[0]
            
            for i in range(batch_size):
                # 검색용 키 (정규화된 z_{t+1}) - clone으로 inplace 방지
                self.knn_key_bank_norm[self.knn_ptr] = zp_norm[i].clone()
                # 후보값 (원본 z_{t+2}) - clone으로 inplace 방지
                self.knn_val_bank_raw[self.knn_ptr] = zpp[i].detach().clone()
                self.knn_ptr = (self.knn_ptr + 1) % self.knn_memory_size
    
    def _compute_knn_target_q(self, zp_target):
        """
        KNN 기반 타겟 Q 계산 (정합성 보장)
        - 검색: 정규화된 zp_target으로 유사한 z_{t+1} 찾기
        - 후보: 해당하는 원본 z_{t+2} 사용
        - 계산: Q(zp_target, z_{t+2}) 중 max
        """
        batch_size = zp_target.shape[0]
        zp_target_norm = F.normalize(zp_target, p=2, dim=1)
        
        # KNN 검색: 정규화된 zp_target과 유사한 K개 후보 찾기
        similarities = torch.matmul(zp_target_norm, self.knn_key_bank_norm.T)  # (B, knn_memory_size)
        _, knn_indices = torch.topk(similarities, self.knn_k, dim=1)  # (B, knn_k)
        
        # 각 후보에 대해 Q값 계산
        q_values_list = []
        for k in range(self.knn_k):
            # k번째 후보들 선택 (원본 z_{t+2} 사용)
            knn_candidates = self.knn_val_bank_raw[knn_indices[:, k]]  # (B, d_model)
            
            # Q(zp_target, z_{t+2}) 계산 (원본 분포로)
            q1_k, q2_k = self.critic_target(zp_target, knn_candidates)
            q_k = torch.min(q1_k, q2_k)  # Double Q
            q_values_list.append(q_k)
        
        # K개 후보 중 max Q값 선택
        q_values_stack = torch.stack(q_values_list, dim=1)  # (B, knn_k, 1)
        q_target = torch.max(q_values_stack, dim=1)[0]  # (B, 1)
        
        return q_target
    
    def get_current_lr(self):
        """현재 learning rate 반환"""
        return {
            'lr': self.scheduler.get_last_lr()[0]
        }