import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from offline_agent import Offline_Encoder
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
        
        # 데이터셋 로드 (data_source: ogbench | d4rl)
        self.data_source = agent_kwargs.pop("data_source", "ogbench")
        print(f"데이터셋 로드 중: {dataset_name} (source: {self.data_source})")
        if self.data_source == "d4rl":
            if create_dataloader_d4rl is None:
                raise ImportError("D4RL 사용 시 d4rl_utils 필요. pip install gym d4rl")
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
                normalize, data_dir, context_length
            )
            self.val_dataset, self.val_loader = create_dataloader(
                dataset_name, "val", max_trajectories // 4, batch_size,
                normalize, data_dir, context_length
            )
        
        # 실제 state_dim 자동 감지
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            state_dim = sample['observations'].shape[-1]
            print(f"데이터셋에서 자동 감지된 state_dim: {state_dim}")
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
        
        # 저장 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 로깅 설정
        self._setup_logging()
        
        print(f"훈련 설정:")
        print(f"  데이터셋: {dataset_name}")
        print(f"  훈련 샘플: {len(self.train_dataset)}")
        print(f"  검증 샘플: {len(self.val_dataset)}")
        print(f"  배치 크기: {batch_size}")
        print(f"  컨텍스트 길이: {context_length}")
        print(f"  상태 차원: {state_dim}")
        print(f"  장치: {device}")
        
        # 로그에 설정 정보 기록
        config_info = {
            'dataset_name': dataset_name,
            'state_dim': state_dim,  # 데이터셋에서 자동 감지
            'hidden_dim': hidden_dim,
            'context_length': context_length,
            'd_model': d_model,
            'batch_size': batch_size,
            'max_trajectories': max_trajectories,
            'normalize': normalize,
            'device': device,
            'save_dir': self.save_dir,  # 자동 생성된 경로
            'data_dir': data_dir
        }
        self.logger.info(f"훈련 설정: {config_info}")
        self.logger.info(f"데이터셋: {dataset_name}, 훈련 샘플: {len(self.train_dataset)}, 검증 샘플: {len(self.val_dataset)}")
    
    def _setup_logging(self):
        """로깅 설정"""
        # 로그 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.save_dir, f"training_{timestamp}.log")
        
        # 로거 설정
        self.logger = logging.getLogger('OfflineTrainer')
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        print(f"로그 파일 저장 위치: {log_file}")
    
    def train_epoch(self):
        """한 에포크 훈련"""
        self.agent.encoder.train()
        self.agent.critic.train()
        self.agent.latent_proposer.train()
        self.agent.GC_success_head.train()
        
        epoch_losses = []
        epoch_metrics = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': [],
            'alignment_loss': [],
            'v_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 데이터를 장치로 이동
            states = batch['observations'].to(self.device)
            next_states = batch['next_observations'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            dones = batch['dones'].to(self.device)
            mask = batch['mask'].to(self.device)
            goal = batch['goal'].to(self.device)  # (B, obs_dim)
            goal_obs = goal.unsqueeze(1).expand(-1, states.size(1), -1)  # (B, K, obs_dim)
            metrics = self.agent.update(states, next_states, rewards, dones, mask, goal_obs)
            
            # 통계 수집
            epoch_losses.append(metrics['total_loss'])
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])
            
            # 진행률 업데이트
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'critic': f"{metrics['critic_loss']:.4f}",
                    'state': f"{metrics['state_loss']:.4f}",
                    'reward': f"{metrics['reward_loss']:.4f}",
                    'v': f"{metrics.get('v_loss', 0):.4f}",
                    'nce': f"{metrics.get('nce_loss', 0):.4f}",
                    'align': f"{metrics.get('alignment_loss', 0):.4f}"
                })
        
        # 에포크 평균 계산
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # 메트릭 히스토리 저장
        for key, value in avg_metrics.items():
            self.train_metrics_history[key].append(value)
        
        return avg_loss, avg_metrics
    
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
                goal = batch['goal'].to(self.device)
                goal_obs = goal.unsqueeze(1).expand(-1, states.size(1), -1)
                metrics = self._compute_validation_metrics(states, next_states, rewards, dones, mask, goal_obs)
                
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
    def _compute_validation_metrics(self, states, next_states, rewards, dones, mask, goal_obs):
        m = self.agent
        m.encoder.eval(); m.encoder_target.eval(); m.critic.eval(); m.critic_target.eval(); m.vz.eval()

        last_valid_idx = last_valid_index_from_mask(mask)
        rewards_last = rewards.gather(1, last_valid_idx.unsqueeze(1)).float().to(m.device)
        dones_last = dones.gather(1, last_valid_idx.unsqueeze(1)).float().to(m.device)
        mask_last = (mask.sum(dim=1, keepdim=True) > 0).float().to(m.device)

        if goal_obs is None:
            goal_z = torch.zeros(states.size(0), m.d_model, device=m.device)
            goal_z_target = torch.zeros_like(goal_z)
        else:
            goal_z        = m.encoder.encode_last_valid(goal_obs, mask)
            goal_z_target = m.encoder_target.encode_last_valid(goal_obs, mask)

        z   = m.encoder.encode_last_valid(states, mask)
        zp  = m.encoder.encode_last_valid(next_states, mask)
        z_t = m.encoder_target.encode_last_valid(states, mask)
        zp_t= m.encoder_target.encode_last_valid(next_states, mask)

        zp_hat = m.latent_proposer(torch.cat([z, goal_z], dim=1))
        state_err = F.mse_loss(zp_hat, zp_t, reduction='none').mean(dim=1, keepdim=True)

        v_next_t = m.vz(zp_t, goal_z_target)
        target_q = rewards_last + m.gamma * (1.0 - dones_last) * v_next_t

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
            raw_bce = m._focal_loss(reward_logits, dones_last, m.focal_alpha, m.focal_gamma, reduction='none')
        else:
            posw = m.pos_weight_ema.to(m.device)
            raw_bce = F.binary_cross_entropy_with_logits(reward_logits, dones_last, pos_weight=posw, reduction='none')
        reward_loss = (raw_bce * mask_last).sum() / mask_last.sum().clamp(min=1.0) if mask_last.sum() > 0 else raw_bce.mean()

        # Alignment (−cos, 마스크 가중; goal 없으면 0)
        if goal_obs is not None:
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

    
    def save_checkpoint(self, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
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
        
        # 일반 체크포인트
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 (두 가지 형식으로 저장)
        if is_best:
            # 1. best_model.pth (기존 형식)
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            # 2. best_offline_checkpoint_{dataset_name}.pth (온라인 트레이너 호환 형식)
            best_checkpoint_path = os.path.join(self.save_dir, f'best_offline_checkpoint_{self.dataset_name}.pth')
            torch.save(checkpoint, best_checkpoint_path)
            
            print(f"최고 성능 모델 저장:")
            print(f"  - {best_path}")
            print(f"  - {best_checkpoint_path}")
    
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
        
        print(f"체크포인트 로드 완료: epoch {self.epoch}")
    
    def plot_training_curves(self):
        """훈련 곡선 시각화"""
        if len(self.train_losses) < 2:
            return
        
        plt.figure(figsize=(20, 10))
        
        # Validation 에포크 인덱스 계산 (validate_every=10 기준)
        val_epochs = list(range(0, len(self.train_losses), 10))  # 0, 10, 20, 30, ...
        if len(self.train_losses) - 1 not in val_epochs:  # 마지막 에포크 추가
            val_epochs.append(len(self.train_losses) - 1)
        
        # Loss 곡선
        plt.subplot(2, 4, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        if self.val_losses:
            plt.plot(val_epochs, self.val_losses, label='Val Loss', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Critic Loss
        plt.subplot(2, 4, 2)
        if self.train_metrics_history['critic_loss']:
            plt.plot(self.train_metrics_history['critic_loss'], label='Train Critic', alpha=0.7)
        if self.val_metrics_history['critic_loss']:
            plt.plot(val_epochs, self.val_metrics_history['critic_loss'], label='Val Critic', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # State Loss
        plt.subplot(2, 4, 3)
        if self.train_metrics_history['state_loss']:
            plt.plot(self.train_metrics_history['state_loss'], label='Train State', alpha=0.7)
        if self.val_metrics_history['state_loss']:
            plt.plot(val_epochs, self.val_metrics_history['state_loss'], label='Val State', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('State Loss')
        plt.title('State Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # V Loss
        plt.subplot(2, 4, 4)
        if self.train_metrics_history['v_loss']:
            plt.plot(self.train_metrics_history['v_loss'], label='Train V', alpha=0.7)
        if self.val_metrics_history['v_loss']:
            plt.plot(val_epochs, self.val_metrics_history['v_loss'], label='Val V', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('V Loss')
        plt.title('V Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reward Loss
        plt.subplot(2, 4, 5)
        if self.train_metrics_history['reward_loss']:
            plt.plot(self.train_metrics_history['reward_loss'], label='Train Reward', alpha=0.7)
        if self.val_metrics_history['reward_loss']:
            plt.plot(val_epochs, self.val_metrics_history['reward_loss'], label='Val Reward', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Reward Loss')
        plt.title('Reward Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # NCE Loss
        plt.subplot(2, 4, 6)
        if self.train_metrics_history['nce_loss'] and any(x > 0 for x in self.train_metrics_history['nce_loss']):
            plt.plot(self.train_metrics_history['nce_loss'], label='Train NCE', alpha=0.7)
        if self.val_metrics_history['nce_loss'] and any(x > 0 for x in self.val_metrics_history['nce_loss']):
            plt.plot(val_epochs, self.val_metrics_history['nce_loss'], label='Val NCE', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('NCE Loss')
        plt.title('InfoNCE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Alignment Loss
        plt.subplot(2, 4, 7)
        if self.train_metrics_history['alignment_loss']:
            plt.plot(self.train_metrics_history['alignment_loss'], label='Train Alignment', alpha=0.7)
        if self.val_metrics_history['alignment_loss']:
            plt.plot(val_epochs, self.val_metrics_history['alignment_loss'], label='Val Alignment', alpha=0.7, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Alignment Loss')
        plt.title('Alignment Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"훈련 곡선 저장: {plot_path}")
    
    def train(self, num_epochs: int = 100, save_every: int = 10, validate_every: int = 5):
        """전체 훈련 루프"""
        print(f"훈련 시작: {num_epochs} 에포크")
        print(f"데이터셋: {self.dataset_name}")
        print(f"저장 주기: {save_every} 에포크")
        print(f"검증 주기: {validate_every} 에포크")
        
        # 로그 기록
        self.logger.info(f"=== 훈련 시작 ===")
        self.logger.info(f"에포크: {num_epochs}, 저장 주기: {save_every}, 검증 주기: {validate_every}")
        self.logger.info(f"데이터셋: {self.dataset_name}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # 훈련
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 검증
            if epoch % validate_every == 0 or epoch == num_epochs - 1:
                val_loss, val_metrics = self.validate()
                self.val_losses.append(val_loss)
                
                # 최고 성능 모델 저장
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"  Critic: {train_metrics['critic_loss']:.4f}, State: {train_metrics['state_loss']:.4f}, Reward: {train_metrics['reward_loss']:.4f}")
                print(f"  V: {train_metrics['v_loss']:.4f}, Alignment: {train_metrics['alignment_loss']:.4f}")
                if train_metrics['nce_loss'] > 0:
                    print(f"  NCE: {train_metrics['nce_loss']:.4f}")
                print(f"  Pos Weight EMA: {self.agent.pos_weight_ema:.4f}")
                
                # 로그 기록
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                self.logger.info(f"  Train - Critic: {train_metrics['critic_loss']:.4f}, State: {train_metrics['state_loss']:.4f}, Reward: {train_metrics['reward_loss']:.4f}")
                self.logger.info(f"  Train - V: {train_metrics['v_loss']:.4f}, Alignment: {train_metrics['alignment_loss']:.4f}")
                if train_metrics['nce_loss'] > 0:
                    self.logger.info(f"  Train - NCE: {train_metrics['nce_loss']:.4f}")
                self.logger.info(f"  Val - Critic: {val_metrics['critic_loss']:.4f}, State: {val_metrics['state_loss']:.4f}, Reward: {val_metrics['reward_loss']:.4f}")
                self.logger.info(f"  Val - V: {val_metrics['v_loss']:.4f}, Alignment: {val_metrics['alignment_loss']:.4f}")
                if val_metrics['nce_loss'] > 0:
                    self.logger.info(f"  Val - NCE: {val_metrics['nce_loss']:.4f}")
                self.logger.info(f"  Pos Weight EMA: {self.agent.pos_weight_ema:.4f}")
                
                if is_best:
                    print(f"  ★ 새로운 최고 성능!")
                    self.logger.info(f"  ★ 새로운 최고 성능! Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}")
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
            
            # 체크포인트 저장
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(is_best=(epoch % validate_every == 0 and val_loss < self.best_val_loss))
            
            # 훈련 곡선 업데이트
            if epoch % 10 == 0:
                self.plot_training_curves()
        
        # 최종 결과
        total_time = time.time() - start_time
        print(f"\n훈련 완료!")
        print(f"총 시간: {total_time/3600:.2f} 시간")
        print(f"최고 검증 손실: {self.best_val_loss:.4f}")
        print(f"최종 모델 저장 위치: {self.save_dir}")
        
        # 로그 기록
        self.logger.info(f"=== 훈련 완료 ===")
        self.logger.info(f"총 시간: {total_time/3600:.2f} 시간")
        self.logger.info(f"최고 검증 손실: {self.best_val_loss:.4f}")
        self.logger.info(f"최종 모델 저장 위치: {self.save_dir}")


def main():
    """메인 훈련 함수"""
    import argparse
    from config import get_offline_config

    parser = argparse.ArgumentParser(description='Offline RL Training')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                       help='Max trajectories (-1 for all, overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (overrides config)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Save directory (overrides config)')
    parser.add_argument('--env', type=str, default=None,
                       help='환경 설정 (config/<env>.yaml). 미지정 시 --dataset 값 또는 default.yaml')
    parser.add_argument('--data', dest='data_source', type=str, default=None,
                       help='데이터 소스: ogbench (기본) | d4rl (D4RL antmaze)')
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'env'}
    if 'dataset' in overrides:
        overrides['dataset_name'] = overrides.pop('dataset')
    if 'device' not in overrides:
        overrides['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = getattr(args, 'env', None) or overrides.get('dataset_name')
    config = get_offline_config(overrides, env=env)

    if config.get("data_source", "ogbench") == "ogbench":
        print("데이터셋 확인 중...")
        download_ogbench_datasets()

    print(f"오프라인 훈련 시작: {config['dataset_name']}")
    print(f"state_dim은 데이터셋에서 자동 감지됩니다")
    mt = config["max_trajectories"]
    print(f"max_trajectories: {mt} ({'전체 데이터' if mt == -1 else str(mt) + '개'})")

    trainer = OfflineTrainer(**{k: v for k, v in config.items() if k not in ("save_every", "validate_every", "num_epochs")})
    trainer.train(
        num_epochs=config['num_epochs'],
        save_every=config['save_every'],
        validate_every=config['validate_every']
    )


if __name__ == "__main__":
    main()
