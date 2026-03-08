"""
오프라인 트레이너: OGBench/D4RL 데이터로 encoder, Qzz, Vz, latent_proposer, GC_success_head 학습.
설정은 config/ (YAML) + CLI overrides. V 학습에 IQL 스타일 expectile loss 사용.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from agents.offline_agent import Offline_Encoder
from utils.logging_utils import get_logger
from utils.ogbench_utils import OGBenchDataset, create_dataloader, download_ogbench_datasets
from utils import expectile_loss, last_valid_index_from_mask
try:
    from utils.d4rl_utils import create_dataloader_d4rl
except ImportError:
    create_dataloader_d4rl = None


class OfflineTrainer:
    def __init__(self, 
                 dataset_name: str,
                 hidden_dim: int = 256,
                 context_length: int = 100,
                 d_model: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_layers: int = 2,
                 device: str = "cuda",
                 batch_size: int = 32,
                 max_trajectories: int = 1000,
                 normalize: bool = True,
                 data_dir: str = './datasets',
                 save_dir: str = None,
                 **agent_kwargs):
        
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size
        
        # save_dir 자동 생성
        if save_dir is None:
            self.save_dir = f'./offline_checkpoints/{dataset_name}'
        else:
            self.save_dir = save_dir
        
        self.data_source = agent_kwargs.pop("data_source", "ogbench")
        context_stride = agent_kwargs.pop("context_stride", 1)
        seed = agent_kwargs.pop("seed", None)
        self.logger = get_logger("offline", None)  # 초기 로그만; save_dir은 아래에서 설정
        self.logger.info(f"데이터셋 로드 중: {dataset_name} (source: {self.data_source})")
        if self.data_source == "d4rl":
            if create_dataloader_d4rl is None:
                raise ImportError("D4RL 사용 시 gym, d4rl 필요. pip install gym d4rl")
            self.train_dataset, self.train_loader = create_dataloader_d4rl(
                dataset_name, "train", max_trajectories, batch_size,
                normalize, data_dir, context_length
            )
            self.val_dataset, self.val_loader = create_dataloader_d4rl(
                dataset_name, "val", max(1, max_trajectories // 4), batch_size,
                normalize, data_dir, context_length
            )
        else:
            self.train_dataset, self.train_loader = create_dataloader(
                dataset_name, "train", max_trajectories, batch_size,
                normalize, data_dir, context_length,
                context_stride=context_stride, seed=seed
            )
            n_val = max(1, (max_trajectories // 4) if max_trajectories > 0 else 500)
            self.val_dataset, self.val_loader = create_dataloader(
                dataset_name, "val", n_val, batch_size, normalize, data_dir, context_length,
                context_stride=context_stride, seed=seed
            )
        
        # 실제 state_dim 자동 감지
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            state_dim = sample['observations'].shape[-1]
            self.logger.info(f"데이터셋에서 자동 감지된 state_dim: {state_dim}")
        else:
            raise ValueError("데이터셋이 비어있습니다.")
        
        # Agent 초기화
        self.agent = Offline_Encoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            device=device,
            **agent_kwargs
        )
        
        # 정규화 통계 저장 (온라인 학습용)
        base = self.train_dataset.dataset if isinstance(self.train_dataset, torch.utils.data.Subset) else self.train_dataset
        self.normalize_stats = {"mean": base.state_mean.copy(), "std": base.state_std.copy()}
        
        # 훈련 통계
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.global_step = 0
        # max_updates mode: step indices at which we validated (for plot x-axis)
        self.val_steps = []

        # 상세 메트릭 트래킹
        self.train_metrics_history = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': [],
            'alignment_loss': [],
            'v_loss': []
        }
        self.val_metrics_history = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': [],
            'alignment_loss': [],
            'v_loss': []
        }
        
        # 저장 디렉토리 생성 후 로거에 파일 핸들러 추가
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = get_logger("offline", self.save_dir)

        self.logger.info("=" * 50)
        self.logger.info(
            f"훈련 설정 | dataset={dataset_name} | train={len(self.train_dataset)} val={len(self.val_dataset)} | "
            f"batch={batch_size} L={context_length} state_dim={state_dim} | device={device}"
        )
        self.logger.info("=" * 50)

    def _train_one_batch(self, batch):
        """Single gradient step: move batch to device, call agent.update, return metrics."""
        states = batch['observations'].to(self.device)
        next_states = batch['next_observations'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        mask = batch['mask'].to(self.device)
        goal = batch['goal'].to(self.device)
        return self.agent.update(states, next_states, rewards, dones, mask, goal)

    @staticmethod
    def _average_metric_buffer(buffer):
        """Given list of metric dicts (from batch updates), return one dict with mean per key. Empty buffer -> empty dict."""
        if not buffer:
            return {}
        keys = list(buffer[0].keys())
        return {k: np.mean([m[k] for m in buffer if k in m]) for k in keys}

    def validate(self):
        """검증"""
        self.agent.encoder.eval()
        self.agent.critic.eval()
        self.agent.latent_proposer.eval()
        self.agent.GC_success_head.eval()
        
        val_losses = []
        val_metrics = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': [],
            'alignment_loss': [],
            'v_loss': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 데이터를 장치로 이동
                states = batch['observations'].to(self.device)
                next_states = batch['next_observations'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                dones = batch['dones'].to(self.device)
                mask = batch['mask'].to(self.device)
                goal = batch['goal'].to(self.device)  # (B, obs_dim)
                metrics = self._compute_validation_metrics(states, next_states, rewards, dones, mask, goal)
                
                # 통계 수집
                val_losses.append(metrics['total_loss'])
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])
        
        # 검증 평균 계산
        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        # 검증 메트릭 히스토리 저장
        for key, value in avg_val_metrics.items():
            self.val_metrics_history[key].append(value)
        
        return avg_val_loss, avg_val_metrics
    
    @torch.no_grad()
    def _compute_validation_metrics(self, states, next_states, rewards, dones, mask, goal):
        """goal: (B, D) single vector; packed internally for encoder."""
        m = self.agent
        m.encoder.eval(); m.encoder_target.eval(); m.critic.eval(); m.critic_target.eval(); m.vz.eval()

        last_valid_idx = last_valid_index_from_mask(mask)
        rewards_last = rewards.gather(1, last_valid_idx.unsqueeze(1)).float().to(m.device)
        dones_last = dones.gather(1, last_valid_idx.unsqueeze(1)).float().to(m.device)
        mask_last = (mask.sum(dim=1, keepdim=True) > 0).float().to(m.device)

        if goal is None:
            goal_z = torch.zeros(states.size(0), m.d_model, device=m.device)
            goal_z_target = torch.zeros_like(goal_z)
        elif goal.dim() == 2:
            B, L = states.size(0), states.size(1)
            goal_packed = torch.zeros(B, L, goal.size(-1), device=m.device, dtype=goal.dtype)
            goal_packed[:, -1, :] = goal.to(m.device)
            goal_mask = torch.zeros(B, L, device=m.device)
            goal_mask[:, -1] = 1.0
            goal_z = m.encoder.encode_last_valid(goal_packed, goal_mask)
            goal_z_target = m.encoder_target.encode_last_valid(goal_packed, goal_mask)
        else:
            goal_z = m.encoder.encode_last_valid(goal, mask)
            goal_z_target = m.encoder_target.encode_last_valid(goal, mask)

        z   = m.encoder.encode_last_valid(states, mask)
        zp  = m.encoder.encode_last_valid(next_states, mask)
        z_t = m.encoder_target.encode_last_valid(states, mask)
        zp_t= m.encoder_target.encode_last_valid(next_states, mask)

        zp_hat = m.latent_proposer(torch.cat([z, goal_z], dim=1))
        state_err = F.mse_loss(zp_hat, zp_t, reduction='none').mean(dim=1, keepdim=True)

        dones_eff = torch.maximum(dones_last, rewards_last)
        v_next_t = m.vz(zp_t, goal_z_target)
        target_q = rewards_last + m.gamma * (1.0 - dones_eff) * v_next_t

        q1, q2 = m.critic(z, zp, goal_z)

        if mask_last.sum() > 0:
            wsum = mask_last.sum().clamp(min=1.0)
            critic_loss = ((F.smooth_l1_loss(q1, target_q, reduction='none') * mask_last).sum() +
                        (F.smooth_l1_loss(q2, target_q, reduction='none') * mask_last).sum()) / wsum
            state_loss  = (state_err * mask_last).sum() / wsum
        else:
            critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)
            state_loss  = state_err.mean()

        # V expectile (훈련과 동일: pred=V(z), target=min(Q_targ(z_t,zp_t)))
        v_pred = m.vz(z, goal_z)
        qt1, qt2 = m.critic_target(z_t, zp_t, goal_z_target)
        qmin_t = torch.min(qt1, qt2)
        v_err = expectile_loss(v_pred, qmin_t, m.expectile_tau, reduction='none')
        v_loss = (v_err * mask_last).sum() / mask_last.sum().clamp(min=1.0) if mask_last.sum() > 0 else v_err.mean()

        reward_logits = m.GC_success_head(torch.cat([zp, goal_z], dim=1))
        if m.use_focal_loss:
            raw_bce = m._focal_loss(reward_logits, rewards_last, m.focal_alpha, m.focal_gamma, reduction='none')
        else:
            posw = m.pos_weight_ema.to(m.device)
            raw_bce = F.binary_cross_entropy_with_logits(reward_logits, rewards_last, pos_weight=posw, reduction='none')
        reward_loss = (raw_bce * mask_last).sum() / mask_last.sum().clamp(min=1.0) if mask_last.sum() > 0 else raw_bce.mean()

        # Alignment (−cos, 마스크 가중; goal 없으면 0)
        if goal is not None:
            eps = 1e-8
            cos = ( (zp - z) * (goal_z - z) ).sum(dim=1) / (
                (zp - z).norm(dim=1) * (goal_z - z).norm(dim=1) + eps )
            align = -cos.unsqueeze(1)  # (B,1)
            alignment_loss = (align * mask_last).sum() / mask_last.sum().clamp(min=1.0) if mask_last.sum() > 0 else align.mean()
        else:
            alignment_loss = torch.tensor(0.0, device=m.device)

        # InfoNCE: 검증은 in-batch만 (은행 X)
        if m.beta_nce > 0:
            z_n = F.normalize(z, p=2, dim=1); zp_n = F.normalize(zp, p=2, dim=1)
            nce_loss = m.infonce_manager._compute_single_direction_nce(z_n, zp_n)
        else:
            nce_loss = torch.tensor(0.0, device=m.device)

        total_loss = (critic_loss +
                    m.beta_s * state_loss +
                    m.beta_r * reward_loss +
                    m.beta_nce * nce_loss +
                    m.beta_v * v_loss +
                    m.beta_a * alignment_loss)

        return {
            'total_loss': float(total_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'v_loss': float(v_loss.item()),
            'alignment_loss': float(alignment_loss.item()),
            'state_loss': float(state_loss.item()),
            'reward_loss': float(reward_loss.item()),
            'nce_loss': float(nce_loss.item()),
        }

    
    def save_checkpoint(self, is_best=False, step=None, only_best=False):
        """체크포인트 저장. step은 주기 저장 파일명용. only_best=True면 best만 저장 (검증 시)."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': getattr(self, 'global_step', 0),
            # 온라인 트레이너 호환 형식
            'encoder': self.agent.encoder.state_dict(),
            'critic': self.agent.critic.state_dict(),
            'vz': self.agent.vz.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'scheduler': self.agent.scheduler.state_dict(),
            'latent_proposer': self.agent.latent_proposer.state_dict(),
            'GC_success_head': self.agent.GC_success_head.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_steps': getattr(self, 'val_steps', []),
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'best_val_loss': self.best_val_loss,
            'dataset_name': self.dataset_name,
            'pos_weight_ema': self.agent.pos_weight_ema,
            # 정규화 통계 (온라인 학습용)
            'normalize_stats': self.normalize_stats,
            'state_dim': self.agent.state_dim,
            'context_length': self.agent.context_length
        }
        step = step if step is not None else self.global_step
        if not only_best:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_step_{step}.pth')
            torch.save(checkpoint, checkpoint_path)
        # 최고 성능 모델 (두 가지 형식으로 저장)
        if is_best:
            # 1. best_model.pth (기존 형식)
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            # 2. best_offline_checkpoint_{dataset_name}.pth (온라인 트레이너 호환 형식)
            best_checkpoint_path = os.path.join(self.save_dir, f'best_offline_checkpoint_{self.dataset_name}.pth')
            torch.save(checkpoint, best_checkpoint_path)
            
            self.logger.info(f"  >> 최고 성능 모델 저장: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.agent.encoder.load_state_dict(checkpoint['encoder'])
        self.agent.critic.load_state_dict(checkpoint['critic'])
        self.agent.vz.load_state_dict(checkpoint['vz'])
        lp_sd = checkpoint.get('latent_proposer') or checkpoint.get('GC_next_state_estimator')
        if lp_sd:
            self.agent.latent_proposer.load_state_dict(lp_sd)
        gs_sd = checkpoint.get('GC_success_head') or checkpoint.get('GC_rewards_estimator')
        if gs_sd:
            self.agent.GC_success_head.load_state_dict(gs_sd)
        # If checkpoint has no target nets, sync from current
        if not checkpoint.get('latent_proposer_target'):
            self.agent.latent_proposer_target.load_state_dict(self.agent.latent_proposer.state_dict())

        # 옵티마이저 및 스케줄러 복원 (구 형식 호환)
        opt_sd = checkpoint.get('optimizer') or checkpoint.get('optimizer_state_dict')
        sched_sd = checkpoint.get('scheduler') or checkpoint.get('scheduler_state_dict')
        if opt_sd:
            self.agent.optimizer.load_state_dict(opt_sd)
        if sched_sd:
            self.agent.scheduler.load_state_dict(sched_sd)
        
        # 훈련 통계 복원
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'critic_loss': [], 'state_loss': [], 'reward_loss': [], 'nce_loss': [], 'alignment_loss': [], 'v_loss': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'critic_loss': [], 'state_loss': [], 'reward_loss': [], 'nce_loss': [], 'alignment_loss': [], 'v_loss': []
        })
        self.best_val_loss = checkpoint['best_val_loss']
        self.agent.pos_weight_ema = checkpoint.get('pos_weight_ema', 1.0)
        self.global_step = checkpoint.get('global_step', 0)
        self.val_steps = checkpoint.get('val_steps', [])
        self.logger.info(f"체크포인트 로드 완료: epoch={self.epoch} global_step={self.global_step}")
    
    def plot_training_curves(self):
        """훈련 곡선 시각화. x축은 검증 시점(step)."""
        if len(self.train_losses) < 2:
            return
        val_steps = getattr(self, "val_steps", [])
        plt.figure(figsize=(20, 10))
        train_x = list(range(len(self.train_losses)))
        if val_steps and len(val_steps) == len(self.val_losses):
            val_x = val_steps
        else:
            val_x = list(range(len(self.val_losses)))
        xlabel = "Update step"
        # Loss 곡선
        plt.subplot(2, 4, 1)
        plt.plot(train_x, self.train_losses, label='Train Loss', alpha=0.7)
        if self.val_losses and val_x:
            plt.plot(val_x, self.val_losses, label='Val Loss', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Critic Loss
        plt.subplot(2, 4, 2)
        if self.train_metrics_history['critic_loss']:
            plt.plot(train_x, self.train_metrics_history['critic_loss'], label='Train Critic', alpha=0.7)
        if self.val_metrics_history['critic_loss'] and len(self.val_metrics_history['critic_loss']) == len(val_x):
            plt.plot(val_x, self.val_metrics_history['critic_loss'], label='Val Critic', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # State Loss
        plt.subplot(2, 4, 3)
        if self.train_metrics_history['state_loss']:
            plt.plot(train_x, self.train_metrics_history['state_loss'], label='Train State', alpha=0.7)
        if self.val_metrics_history['state_loss'] and len(self.val_metrics_history['state_loss']) == len(val_x):
            plt.plot(val_x, self.val_metrics_history['state_loss'], label='Val State', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('State Loss')
        plt.title('State Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # V Loss
        plt.subplot(2, 4, 4)
        if self.train_metrics_history['v_loss']:
            plt.plot(train_x, self.train_metrics_history['v_loss'], label='Train V', alpha=0.7)
        if self.val_metrics_history['v_loss'] and len(self.val_metrics_history['v_loss']) == len(val_x):
            plt.plot(val_x, self.val_metrics_history['v_loss'], label='Val V', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('V Loss')
        plt.title('V Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reward Loss
        plt.subplot(2, 4, 5)
        if self.train_metrics_history['reward_loss']:
            plt.plot(train_x, self.train_metrics_history['reward_loss'], label='Train Reward', alpha=0.7)
        if self.val_metrics_history['reward_loss'] and len(self.val_metrics_history['reward_loss']) == len(val_x):
            plt.plot(val_x, self.val_metrics_history['reward_loss'], label='Val Reward', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('Reward Loss')
        plt.title('Reward Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # NCE Loss
        plt.subplot(2, 4, 6)
        if self.train_metrics_history['nce_loss'] and any(x > 0 for x in self.train_metrics_history['nce_loss']):
            plt.plot(train_x, self.train_metrics_history['nce_loss'], label='Train NCE', alpha=0.7)
        if self.val_metrics_history['nce_loss'] and any(x > 0 for x in self.val_metrics_history['nce_loss']) and len(self.val_metrics_history['nce_loss']) == len(val_x):
            plt.plot(val_x, self.val_metrics_history['nce_loss'], label='Val NCE', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('NCE Loss')
        plt.title('InfoNCE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Alignment Loss
        plt.subplot(2, 4, 7)
        if self.train_metrics_history['alignment_loss']:
            plt.plot(train_x, self.train_metrics_history['alignment_loss'], label='Train Alignment', alpha=0.7)
        if self.val_metrics_history['alignment_loss'] and len(self.val_metrics_history['alignment_loss']) == len(val_x):
            plt.plot(val_x, self.val_metrics_history['alignment_loss'], label='Val Alignment', alpha=0.7, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel('Alignment Loss')
        plt.title('Alignment Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  >> 훈련 곡선 저장: {plot_path}")
    
    def train(self, max_updates: int = 1_000_000, save_every: int = 50_000, validate_every: int = 10_000):
        """Offline training loop. Train for max_updates gradient steps. validate_every/save_every are step intervals."""
        self._save_every = save_every
        self._validate_every = validate_every
        max_updates = max(1, max_updates)
        self.logger.info("=" * 50)
        self.logger.info(f"훈련 시작 | max_updates={max_updates} | save_every={save_every} | validate_every={validate_every} | dataset={self.dataset_name}")
        self.logger.info("=" * 50)
        start_time = time.time()
        global_step = getattr(self, 'global_step', 0)
        interval_buffer = []
        while global_step < max_updates:
            self.epoch += 1
            for batch in self.train_loader:
                self.agent.encoder.train()
                self.agent.critic.train()
                self.agent.latent_proposer.train()
                self.agent.GC_success_head.train()
                metrics = self._train_one_batch(batch)
                interval_buffer.append(metrics)
                global_step += 1
                self.global_step = global_step

                if global_step % 100 == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.logger.info("=" * 50)
                    self.logger.info(f"Step {global_step}/{max_updates} [{ts}]")
                    self.logger.info(f"Total Loss = {metrics['total_loss']:.4f}")
                    self.logger.info(f"Critic Loss = {metrics['critic_loss']:.4f}")
                    self.logger.info(f"State Loss = {metrics['state_loss']:.4f}")
                    self.logger.info(f"Reward Loss = {metrics['reward_loss']:.4f}")
                    self.logger.info(f"V Loss = {metrics.get('v_loss', 0):.4f}")
                    if metrics.get('nce_loss', 0) > 0:
                        self.logger.info(f"NCE Loss = {metrics['nce_loss']:.4f}")
                    if metrics.get('alignment_loss', 0) != 0:
                        self.logger.info(f"Alignment Loss = {metrics.get('alignment_loss', 0):.4f}")
                    self.logger.info("=" * 50)

                if global_step % validate_every == 0:
                    avg_train = self._average_metric_buffer(interval_buffer)
                    interval_buffer = []
                    val_loss, val_metrics = self.validate()
                    self.val_steps.append(global_step)
                    self.train_losses.append(avg_train.get('total_loss', 0.0))
                    self.val_losses.append(val_loss)
                    for k in self.train_metrics_history:
                        if k in avg_train:
                            self.train_metrics_history[k].append(avg_train[k])
                    for k, v in val_metrics.items():
                        if k in self.val_metrics_history:
                            self.val_metrics_history[k].append(v)
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.logger.info("=" * 50)
                    self.logger.info(f"Step {global_step}/{max_updates} [{ts}]")
                    self.logger.info(f"Total Loss = {avg_train.get('total_loss', 0):.4f}")
                    self.logger.info(f"Val Loss = {val_loss:.4f}")
                    self.logger.info(f"Critic Loss = {avg_train.get('critic_loss', 0):.4f}")
                    self.logger.info(f"State Loss = {avg_train.get('state_loss', 0):.4f}")
                    self.logger.info(f"Reward Loss = {avg_train.get('reward_loss', 0):.4f}")
                    self.logger.info(f"V Loss = {avg_train.get('v_loss', 0):.4f}")
                    self.logger.info(f"Alignment Loss = {avg_train.get('alignment_loss', 0):.4f}")
                    if avg_train.get('nce_loss', 0) > 0:
                        self.logger.info(f"NCE Loss = {avg_train['nce_loss']:.4f}")
                    self.logger.info(f"pos_weight_ema = {self.agent.pos_weight_ema:.4f}")
                    if is_best:
                        self.logger.info("★ best")
                    self.logger.info("=" * 50)
                    if is_best:
                        self.save_checkpoint(is_best=True, step=global_step, only_best=True)
                if global_step % save_every == 0:
                    self.save_checkpoint(is_best=False, step=global_step)
                if global_step % 50000 == 0:
                    self.plot_training_curves()
                if global_step >= max_updates:
                    break
            if global_step >= max_updates:
                break
        # Final validation/save if we did not land exactly on validate_every/save_every
        if global_step > 0:
            if global_step % validate_every != 0:
                if interval_buffer:
                    avg_train = self._average_metric_buffer(interval_buffer)
                    self.train_losses.append(avg_train.get('total_loss', 0.0))
                    for k in self.train_metrics_history:
                        if k in avg_train:
                            self.train_metrics_history[k].append(avg_train[k])
                val_loss, val_metrics = self.validate()
                self.val_steps.append(global_step)
                self.val_losses.append(val_loss)
                for k, v in val_metrics.items():
                    if k in self.val_metrics_history:
                        self.val_metrics_history[k].append(v)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True, step=global_step, only_best=True)
                ts = datetime.now().strftime("%H:%M:%S")
                self.logger.info("=" * 50)
                self.logger.info(f"Step {global_step}/{max_updates} [{ts}] (final val)")
                self.logger.info(f"Val Loss = {val_loss:.4f}")
                self.logger.info("=" * 50)
            if global_step % save_every != 0:
                self.save_checkpoint(is_best=False, step=global_step)
        total_time = time.time() - start_time
        self.logger.info("=" * 50)
        self.logger.info(
            f"훈련 완료 | updates={global_step} 시간={total_time/3600:.2f}h best_val_loss={self.best_val_loss:.4f} save_dir={self.save_dir}"
        )
        self.logger.info("=" * 50)
