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
from ogbench_utils import OGBenchDataset, create_dataloader, download_ogbench_datasets


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
        
        # 데이터셋 로드
        print(f"데이터셋 로드 중: {dataset_name}")
        self.train_dataset, self.train_loader = create_dataloader(
            dataset_name, 'train', max_trajectories, batch_size, 
            normalize, data_dir, context_length
        )
        
        self.val_dataset, self.val_loader = create_dataloader(
            dataset_name, 'val', max_trajectories // 4, batch_size, 
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
        self.normalize_stats = {
            'mean': self.train_dataset.state_mean.copy(),
            'std': self.train_dataset.state_std.copy()
        }
        
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
            'nce_loss': []
        }
        self.val_metrics_history = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': []
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
        self.agent.next_state_estimator.train()
        self.agent.rewards_estimator.train()
        
        epoch_losses = []
        epoch_metrics = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 데이터를 장치로 이동
            states = batch['observations'].to(self.device)  # (B, K, state_dim)
            next_states = batch['next_observations'].to(self.device)  # (B, K, state_dim)
            rewards = batch['rewards'].to(self.device)  # (B, K)
            dones = batch['dones'].to(self.device)  # (B, K)
            mask = batch['mask'].to(self.device)  # (B, K)
            goal_obs = batch['goal_obs'].to(self.device)  # (B, K, state_dim)
            
            # 마지막 스텝의 reward와 dones만 사용
            rewards_last = rewards[:, -1:].float()  # (B, 1)
            dones_last = dones[:, -1:].float()  # (B, 1)
            
            # Agent 업데이트 (전체 마스크 전달)
            metrics = self.agent.update(states, next_states, rewards_last, dones_last, mask, goal_obs)
            
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
                    'nce': f"{metrics.get('nce_loss', 0):.4f}"
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
        self.agent.next_state_estimator.eval()
        self.agent.rewards_estimator.eval()
        
        val_losses = []
        val_metrics = {
            'critic_loss': [],
            'state_loss': [],
            'reward_loss': [],
            'nce_loss': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 데이터를 장치로 이동
                states = batch['observations'].to(self.device)
                next_states = batch['next_observations'].to(self.device)
                rewards = batch['rewards'].to(self.device)
                dones = batch['dones'].to(self.device)
                mask = batch['mask'].to(self.device)
                goal_obs = batch['goal_obs'].to(self.device)
                
                # 마지막 스텝의 reward와 dones만 사용
                rewards_last = rewards[:, -1:].float()
                dones_last = dones[:, -1:].float()
                
                # 검증용 손실 계산 (gradient 없이, 전체 마스크 전달)
                metrics = self._compute_validation_metrics(states, next_states, rewards_last, dones_last, mask, goal_obs)
                
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
    
    def _compute_validation_metrics(self, states, next_states, rewards, dones, mask, goal_obs):
        """검증용 메트릭 계산 (gradient 없이)"""
        # Goal 인코딩
        if goal_obs is None:
            goal_z = torch.zeros(states.shape[0], self.agent.d_model, device=self.agent.device)
            goal_z_target = torch.zeros(states.shape[0], self.agent.d_model, device=self.agent.device)
        else:
            goal_z = self.agent.encoder.encode_last_valid(goal_obs, mask)  # (B, d)
            goal_z_target = self.agent.encoder_target.encode_last_valid(goal_obs, mask)  # (B, d)
        
        # 인코딩 (마스크 적용)
        z = self.agent.encoder.encode_trajectory(states, mask)
        zp = self.agent.encoder.encode_trajectory(next_states, mask)
        
        # 타겟 인코딩 (마스크 적용)
        z_target = self.agent.encoder_target.encode_trajectory(states, mask)
        zp_target = self.agent.encoder_target.encode_trajectory(next_states, mask)
        
        # 1-step latent consistency
        zp_estimated = self.agent.next_state_estimator(z)
        
        # 타겟 Q 계산
        v_target_next = self.agent.vz(zp_target, goal_z_target)  # (B,1)
        target_q = rewards + self.agent.gamma * (1.0 - dones) * v_target_next
        
        # 현재 Q
        q_current1, q_current2 = self.agent.critic(z, zp, goal_z)
        
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
        
        v_current = self.agent.vz(z, goal_z)
        v_loss = F.mse_loss(v_current, torch.min(q_current1, q_current2))

        # 보상/종결 예측
        reward_logits = self.agent.rewards_estimator(zp)
        
        # --- State loss (둘 다 (B,d)) -> (B,1)로 평균 후 mask_last 적용 ---
        state_err = F.mse_loss(zp_estimated, zp, reduction='none').mean(dim=1, keepdim=True)  # (B,1)
        if mask_last is not None:
            state_loss = (state_err * mask_last).sum() / mask_last.sum().clamp(min=1.0)
        else:
            state_loss = state_err.mean()
        
        # --- Reward(Done) BCE (둘 다 (B,1)) ---
        if self.agent.use_focal_loss:
            raw_bce = self.agent._focal_loss(reward_logits, dones, self.agent.focal_alpha, self.agent.focal_gamma, reduction='none')  # (B,1)
        else:
            posw = self.agent._compute_pos_weight(dones).to(self.device)
            raw_bce = F.binary_cross_entropy_with_logits(
                reward_logits, dones, pos_weight=posw, reduction='none'
            )
        if mask_last is not None:
            reward_loss = (raw_bce * mask_last).sum() / mask_last.sum().clamp(min=1.0)
        else:
            reward_loss = raw_bce.mean()
        
        # InfoNCE
        nce_loss = 0.0
        if self.agent.beta_nce > 0:
            nce_loss = self.agent.infonce_manager.compute_infonce_loss(z, zp)
        
        total_loss = critic_loss + self.agent.beta_s * state_loss + self.agent.beta_r * reward_loss + self.agent.beta_nce * nce_loss + self.agent.beta_v * v_loss
        
        return {
            'total_loss': float(total_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'v_loss': float(v_loss.item()), 
            'state_loss': float(state_loss.item()),
            'reward_loss': float(reward_loss.item()),
            'nce_loss': float(nce_loss.item()) if isinstance(nce_loss, torch.Tensor) else nce_loss,
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
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'scheduler_state_dict': self.agent.scheduler.state_dict(),
            'next_state_estimator': self.agent.next_state_estimator.state_dict(),
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
        
        # Agent 상태 복원 (새로운 형식)
        self.agent.encoder.load_state_dict(checkpoint['encoder'])
        self.agent.critic.load_state_dict(checkpoint['critic'])
        self.agent.vz.load_state_dict(checkpoint['vz'])
        
        # 옵티마이저 및 스케줄러 복원
        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.agent.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # 훈련 통계 복원
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'critic_loss': [], 'state_loss': [], 'reward_loss': [], 'nce_loss': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'critic_loss': [], 'state_loss': [], 'reward_loss': [], 'nce_loss': []
        })
        self.best_val_loss = checkpoint['best_val_loss']
        self.agent.pos_weight_ema = checkpoint.get('pos_weight_ema', 1.0)
        
        print(f"체크포인트 로드 완료: epoch {self.epoch}")
    
    def plot_training_curves(self):
        """훈련 곡선 시각화"""
        if len(self.train_losses) < 2:
            return
        
        plt.figure(figsize=(16, 10))
        
        # Loss 곡선
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Critic Loss
        plt.subplot(2, 3, 2)
        if self.train_metrics_history['critic_loss']:
            plt.plot(self.train_metrics_history['critic_loss'], label='Train Critic', alpha=0.7)
        if self.val_metrics_history['critic_loss']:
            plt.plot(self.val_metrics_history['critic_loss'], label='Val Critic', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # State Loss
        plt.subplot(2, 3, 3)
        if self.train_metrics_history['state_loss']:
            plt.plot(self.train_metrics_history['state_loss'], label='Train State', alpha=0.7)
        if self.val_metrics_history['state_loss']:
            plt.plot(self.val_metrics_history['state_loss'], label='Val State', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('State Loss')
        plt.title('State Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Reward Loss
        plt.subplot(2, 3, 4)
        if self.train_metrics_history['reward_loss']:
            plt.plot(self.train_metrics_history['reward_loss'], label='Train Reward', alpha=0.7)
        if self.val_metrics_history['reward_loss']:
            plt.plot(self.val_metrics_history['reward_loss'], label='Val Reward', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Reward Loss')
        plt.title('Reward Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # NCE Loss
        plt.subplot(2, 3, 5)
        if self.train_metrics_history['nce_loss'] and any(x > 0 for x in self.train_metrics_history['nce_loss']):
            plt.plot(self.train_metrics_history['nce_loss'], label='Train NCE', alpha=0.7)
        if self.val_metrics_history['nce_loss'] and any(x > 0 for x in self.val_metrics_history['nce_loss']):
            plt.plot(self.val_metrics_history['nce_loss'], label='Val NCE', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('NCE Loss')
        plt.title('InfoNCE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(2, 3, 6)
        lr_info = self.agent.get_current_lr()
        plt.axhline(y=lr_info['lr'], color='r', linestyle='--', alpha=0.7, label=f'Current LR: {lr_info["lr"]:.2e}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
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
                if train_metrics['nce_loss'] > 0:
                    print(f"  NCE: {train_metrics['nce_loss']:.4f}")
                print(f"  Pos Weight EMA: {self.agent.pos_weight_ema:.4f}")
                
                # 로그 기록
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                self.logger.info(f"  Train - Critic: {train_metrics['critic_loss']:.4f}, State: {train_metrics['state_loss']:.4f}, Reward: {train_metrics['reward_loss']:.4f}")
                if train_metrics['nce_loss'] > 0:
                    self.logger.info(f"  Train - NCE: {train_metrics['nce_loss']:.4f}")
                self.logger.info(f"  Val - Critic: {val_metrics['critic_loss']:.4f}, State: {val_metrics['state_loss']:.4f}, Reward: {val_metrics['reward_loss']:.4f}")
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
    
    parser = argparse.ArgumentParser(description='Offline RL Training')
    parser.add_argument('--dataset', type=str, default='antmaze-medium-navigate-v0',
                       help='Dataset name (default: antmaze-medium-navigate-v0)')
    parser.add_argument('--max_trajectories', type=int, default=-1,
                       help='Maximum number of trajectories (-1 for all, default: -1)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                       help='Data directory (default: ./datasets)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Save directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # 데이터셋 다운로드 (필요시)
    print("데이터셋 확인 중...")
    download_ogbench_datasets()
    
    # 훈련 설정
    config = {
        'dataset_name': args.dataset,
        'hidden_dim': 256,
        'context_length': 100,
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'n_layers': 2,
        'device': args.device,
        'batch_size': args.batch_size,
        'max_trajectories': args.max_trajectories,
        'normalize': True,
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        
        # Agent 하이퍼파라미터
        'tau': 0.01,
        'gamma': 0.99,
        'encoder_lr': 3e-4,
        'warmup_steps': 1000,
        'drop_p': 0.1,
        'beta_s': 1.0,
        'beta_r': 1.0,
        'beta_nce': 0.1,
        'beta_v': 0.1,
        'use_focal_loss': False,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'nce_temperature': 0.1,
        'memory_bank_size': 10000,
        'use_bidirectional_nce': True,
    }
    
    print(f"오프라인 훈련 시작: {args.dataset}")
    print(f"state_dim은 데이터셋에서 자동 감지됩니다")
    print(f"max_trajectories: {args.max_trajectories} ({'전체 데이터' if args.max_trajectories == -1 else f'{args.max_trajectories}개'})")
    
    # 트레이너 생성
    trainer = OfflineTrainer(**config)
    
    # 훈련 시작
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=20,
        validate_every=10
    )


if __name__ == "__main__":
    main()
