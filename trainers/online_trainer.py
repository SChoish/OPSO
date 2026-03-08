import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import ogbench

from agents.online_agent import Online_Agent
from utils.logging_utils import get_logger


class OnlineTrainer:
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
                 max_episodes: int = 1000,
                 max_steps_per_episode: int = 1000,
                 save_dir: str = None,
                 offline_checkpoint_path: str = None,
                 online_checkpoint_path: str = None,
                 **agent_kwargs):
        
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # save_dir 자동 생성
        if save_dir is None:
            self.save_dir = f'./online_checkpoints/{dataset_name}'
        else:
            self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = get_logger("online", self.save_dir)

        # 환경 생성 (헤드리스 모드)
        self.logger.info(f"환경 생성 중: {dataset_name}")
        self.env = ogbench.make_env_and_datasets(dataset_name, env_only=True)
        # 렌더링 비활성화
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # 환경에서 자동으로 가져온 차원 사용
        self.logger.info(f"환경 차원: state_dim={self.state_dim}, action_dim={self.action_dim}")
        
        # 에이전트 생성
        self.agent = Online_Agent(
            state_dim=self.state_dim,
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            context_length=context_length,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            device=device,
            batch_size=batch_size,
            offline_checkpoint_path=offline_checkpoint_path,
            online_checkpoint_path=online_checkpoint_path,
            **agent_kwargs
        )
        
        # 메트릭 히스토리
        self.train_metrics_history = {
            'critic_loss': [],
            'action_loss': [],
            'v_loss': [],
            'state_loss': [],
            'total_loss': [],
            'episode_reward': [],
            'episode_length': [],
            'success_rate': [],
            'latent_noise': [],
            'action_noise': []
        }
        
        self.val_metrics_history = {
            'success_rate': [],
            'episode_reward': [],
            'episode_length': []
        }
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.epoch = 0
        self.best_success_rate = 0.0
        self.eval_episodes = []

        # Proposer latent noise: z' = τ(z,z_g) + σξ (policy 구조 유지, 근처 탐색)
        self.latent_noise_start = float(agent_kwargs.get('latent_noise_start', 0.15))
        self.latent_noise_end = float(agent_kwargs.get('latent_noise_end', 0.02))
        self.latent_noise_decay_steps = min(200, max_episodes)
        
        # 액션 노이즈 스케줄
        self.action_noise_start = 0.20
        self.action_noise_end = 0.05
        self.action_noise_decay_steps = self.latent_noise_decay_steps
        
        # 설정 정보 로깅
        config_info = {
            'dataset_name': dataset_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': hidden_dim,
            'context_length': context_length,
            'd_model': d_model,
            'max_episodes': max_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'device': device,
            'batch_size': batch_size,
            'save_dir': self.save_dir,
            'latent_noise_start': self.latent_noise_start,
            'latent_noise_end': self.latent_noise_end,
            'latent_noise_decay_steps': self.latent_noise_decay_steps,
            **agent_kwargs
        }
        self.logger.info("=" * 50)
        self.logger.info(f"OnlineTrainer 초기화 완료: {config_info}")
        self.logger.info("=" * 50)
    
    def get_latent_noise(self, episode):
        """Proposer 출력 잠재 노이즈 σ (선형 감쇠). z' = τ(z,z_g) + σξ."""
        if episode < self.latent_noise_decay_steps:
            decay_rate = (self.latent_noise_start - self.latent_noise_end) / self.latent_noise_decay_steps
            sigma = self.latent_noise_start - decay_rate * episode
        else:
            sigma = self.latent_noise_end
        return max(sigma, self.latent_noise_end)
    
    def get_action_noise(self, episode):
        """액션 노이즈 값 계산 (선형 감쇠)"""
        if episode < self.action_noise_decay_steps:
            decay_rate = (self.action_noise_start - self.action_noise_end) / self.action_noise_decay_steps
            noise = self.action_noise_start - decay_rate * episode
        else:
            noise = self.action_noise_end
        return max(noise, self.action_noise_end)

    def collect_experience(self, num_episodes=10, episode=0):
        """경험 수집 (raw obs; agent가 select_action 내부에서만 정규화). Goal은 에피소드당 하나."""
        experiences = []
        current_latent_noise = self.get_latent_noise(episode)
        current_action_noise = self.get_action_noise(episode)

        for _ in range(num_episodes):
            task_id = np.random.randint(1, 6)
            ob, info = self.env.reset(options=dict(task_id=task_id, render_goal=False))
            goal = info['goal']

            episode_experience = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
                'goal': goal,
            }
            done = False
            step = 0

            while not done and step < self.max_steps_per_episode:
                action = self.agent.select_action(ob, goal, latent_noise_std=current_latent_noise, action_noise=current_action_noise)
                next_ob, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_experience['states'].append(ob.copy())
                episode_experience['actions'].append(action.copy())
                episode_experience['next_states'].append(next_ob.copy())
                episode_experience['rewards'].append(reward)
                episode_experience['dones'].append(float(terminated or truncated))

                ob = next_ob
                step += 1

            experiences.append(episode_experience)
        return experiences

    def train_episode(self, experiences, context_stride=1):
        """Timestep-centered windows: each transition t contributes one sample (last valid = s_t). Goal per sample (B, D)."""
        L = self.agent.context_length
        samples_states = []
        samples_next_states = []
        samples_actions = []
        samples_rewards = []
        samples_dones = []
        samples_masks = []
        samples_goals = []

        for exp in experiences:
            states_arr = np.array(exp['states'], dtype=np.float32)
            next_arr = np.array(exp['next_states'], dtype=np.float32)
            actions_arr = np.array(exp['actions'], dtype=np.float32)
            rewards_arr = np.array(exp['rewards'], dtype=np.float32)
            dones_arr = np.array(exp['dones'], dtype=np.float32)
            goal = np.asarray(exp['goal'], dtype=np.float32)
            T = len(exp['states'])
            if T < 2:
                continue
            # Transition indices t = 0..T-1; states[t]=s_t, next_states[t]=s_{t+1}; reward/done/action[t]=transition t. Include final transition.
            t_values = list(range(0, T, context_stride))
            if T > 1 and (T - 1) not in t_values:
                t_values.append(T - 1)
            for t in t_values:
                history_start = max(0, t + 1 - L)
                window_len = t + 1 - history_start
                pad_left = L - window_len
                # Invariant: last valid token of states = s_t, of next_states = s_{t+1}; reward/done/action at last valid = transition t.
                states_win = states_arr[history_start : t + 1]
                next_win = next_arr[history_start : t + 1]
                states_pad = np.zeros((L, self.state_dim), dtype=np.float32)
                states_pad[pad_left:] = states_win
                next_pad = np.zeros((L, self.state_dim), dtype=np.float32)
                next_pad[pad_left:] = next_win
                mask = np.zeros(L, dtype=np.float32)
                mask[pad_left:] = 1.0
                rewards_pad = np.zeros(L, dtype=np.float32)
                rewards_pad[L - 1] = rewards_arr[t]
                dones_pad = np.zeros(L, dtype=np.float32)
                dones_pad[L - 1] = float(dones_arr[t])
                actions_pad = np.zeros((L, self.action_dim), dtype=np.float32)
                actions_pad[L - 1] = actions_arr[t]
                samples_states.append(states_pad)
                samples_next_states.append(next_pad)
                samples_actions.append(actions_pad)
                samples_rewards.append(rewards_pad)
                samples_dones.append(dones_pad)
                samples_masks.append(mask)
                samples_goals.append(goal)

        if not samples_states:
            return {
                'total_loss': 0.0,
                'critic_loss': 0.0,
                'action_loss': 0.0,
                'v_loss': 0.0,
                'state_loss': 0.0,
            }

        episode_losses = []
        episode_metrics = {'critic_loss': [], 'action_loss': [], 'v_loss': [], 'state_loss': []}

        for start in range(0, len(samples_states), self.batch_size):
            end = min(start + self.batch_size, len(samples_states))
            batch_states = np.array(samples_states[start:end])
            batch_next = np.array(samples_next_states[start:end])
            batch_actions = np.array(samples_actions[start:end])
            batch_rewards = np.array(samples_rewards[start:end])
            batch_dones = np.array(samples_dones[start:end])
            batch_masks = np.array(samples_masks[start:end])
            batch_goals = np.array(samples_goals[start:end])

            states = torch.FloatTensor(batch_states).to(self.device)
            next_states = torch.FloatTensor(batch_next).to(self.device)
            actions = torch.FloatTensor(batch_actions).to(self.device)
            rewards = torch.FloatTensor(batch_rewards).to(self.device)
            dones = torch.FloatTensor(batch_dones).to(self.device)
            masks = torch.FloatTensor(batch_masks).to(self.device)
            goals = torch.FloatTensor(batch_goals).to(self.device)

            metrics = self.agent.update(states, actions, next_states, rewards, dones, masks, goals)
            episode_losses.append(metrics['critic_loss'] + metrics['action_loss'] + metrics['v_loss'])
            for key in episode_metrics:
                if key in metrics:
                    episode_metrics[key].append(metrics[key])

        return {
            'total_loss': np.mean(episode_losses),
            'critic_loss': np.mean(episode_metrics['critic_loss']),
            'action_loss': np.mean(episode_metrics['action_loss']),
            'v_loss': np.mean(episode_metrics['v_loss']),
            'state_loss': np.mean(episode_metrics['state_loss']),
        }

    def evaluate(self, num_episodes=5):
        """에이전트 평가. Raw ob/goal만 전달 (정규화는 agent.select_action 내부에서만)."""
        success_count = 0
        total_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            task_id = episode % 5 + 1
            ob, info = self.env.reset(options=dict(task_id=task_id, render_goal=False))
            goal = info['goal']
            episode_reward = 0
            step = 0
            done = False

            while not done and step < self.max_steps_per_episode:
                action = self.agent.select_action(ob, goal, eval_mode=True)
                ob, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
            
            # 성공 여부 확인
            success = info.get('success', 0)
            success_count += success
            
            total_rewards.append(episode_reward)
            episode_lengths.append(step)
        
        success_rate = success_count / num_episodes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        return {
            'success_rate': success_rate,
            'episode_reward': avg_reward,
            'episode_length': avg_length
        }

    def train(self, num_episodes_per_epoch=10, eval_interval=10):
        """메인 훈련 루프. 각 iteration마다 num_episodes_per_epoch개 env 에피소드 수집 후 train_episode(); episode = training iteration index."""
        self.logger.info("=" * 50)
        self.logger.info(f"온라인 훈련 시작 | max_episodes={self.max_episodes} eval_interval={eval_interval} dataset={self.dataset_name}")
        self.logger.info("=" * 50)

        for iteration in range(self.max_episodes):
            # iteration = training step index; each step: collect num_episodes_per_epoch env episodes, then train
            experiences = self.collect_experience(num_episodes_per_epoch, iteration)
            train_metrics = self.train_episode(experiences)

            for key, value in train_metrics.items():
                self.train_metrics_history[key].append(value)

            if iteration > 0 and iteration % 100 == 0:
                self.logger.info("====")
                self.logger.info(f"Episode {iteration} / {self.max_episodes}")
                self.logger.info(f"total_loss = {train_metrics['total_loss']:.4f}")
                if 'critic_loss' in train_metrics:
                    self.logger.info(f"critic_loss = {train_metrics['critic_loss']:.4f}")
                if 'action_loss' in train_metrics:
                    self.logger.info(f"action_loss = {train_metrics['action_loss']:.4f}")
                if 'v_loss' in train_metrics:
                    self.logger.info(f"v_loss = {train_metrics['v_loss']:.4f}")
                self.logger.info("====")

            if iteration % eval_interval == 0:
                self.eval_episodes.append(iteration)
                val_metrics = self.evaluate()
                for key, value in val_metrics.items():
                    self.val_metrics_history[key].append(value)
                current_latent_noise = self.get_latent_noise(iteration)
                current_action_noise = self.get_action_noise(iteration)
                self.logger.info("====")
                self.logger.info(f"Episode {iteration} / {self.max_episodes} (eval)")
                self.logger.info(f"total_loss = {train_metrics['total_loss']:.4f}")
                self.logger.info(f"success_rate = {val_metrics['success_rate']:.3f}")
                self.logger.info(f"episode_reward = {val_metrics['episode_reward']:.3f}")
                self.logger.info(f"latent_noise = {current_latent_noise:.4f}")
                self.logger.info(f"action_noise = {current_action_noise:.3f}")
                self.logger.info("====")
                
                self.train_metrics_history['latent_noise'].append(current_latent_noise)
                self.train_metrics_history['action_noise'].append(current_action_noise)
                
                # 훈련 곡선 업데이트 (10 에피소드마다)
                if iteration % 10 == 0:
                    self.plot_training_curves()
                
                # 최고 성공률 체크포인트 저장
                if val_metrics['success_rate'] > self.best_success_rate:
                    self.best_success_rate = val_metrics['success_rate']
                    self.save_checkpoint(iteration, is_best=True)
            
            # 정기 체크포인트 저장
            if iteration % 100 == 0:
                self.save_checkpoint(iteration)
        
        self.logger.info("=" * 50)
        self.logger.info(f"훈련 완료 | episodes={self.max_episodes} best_success_rate={self.best_success_rate:.3f} save_dir={self.save_dir}")
        self.logger.info("=" * 50)

    def save_checkpoint(self, episode, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'episode': episode,
            'epoch': episode,  # 호환성을 위해 epoch도 추가
            'eval_episodes': getattr(self, 'eval_episodes', []),
            # 오프라인 트레이너와 호환되는 형식
            'encoder': self.agent.encoder.state_dict(),
            'critic': self.agent.critic.state_dict(),
            'vz': self.agent.vz.state_dict(),
            'latent_proposer': self.agent.latent_proposer.state_dict(),
            'critic_target': self.agent.critic_target.state_dict(),
            'vz_target': self.agent.vz_target.state_dict(),
            'inv_dynamics': self.agent.inv_dynamics.state_dict(),
            'critic_optimizer': self.agent.critic_optimizer.state_dict(),
            'env_optimizer': self.agent.env_optimizer.state_dict(),
            'critic_scheduler': self.agent.critic_scheduler.state_dict(),
            'env_scheduler': self.agent.env_scheduler.state_dict(),
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'best_success_rate': self.best_success_rate,
            'dataset_name': self.dataset_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'context_length': self.agent.context_length,
            # 정규화 통계 (오프라인에서 로드한 것)
            'normalize_stats': self.agent.normalize_stats,

        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, f'best_online_checkpoint_{self.dataset_name}.pth')
        else:
            checkpoint_path = os.path.join(self.save_dir, f'online_checkpoint_{self.dataset_name}_ep{episode}.pth')
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"  >> 체크포인트 저장: {checkpoint_path}")

    def plot_training_curves(self):
        """훈련 곡선 시각화. Val 곡선 x축 = 실제 평가 시점 (eval_episodes)."""
        if len(self.train_metrics_history['total_loss']) < 2:
            return
        
        plt.figure(figsize=(20, 10))
        eval_eps = getattr(self, 'eval_episodes', [])
        if not eval_eps or len(eval_eps) != len(self.val_metrics_history.get('episode_reward', [])):
            eval_eps = list(range(len(self.val_metrics_history.get('episode_reward', []))))
        
        # Total Loss 곡선
        plt.subplot(2, 4, 1)
        has_train = len(self.train_metrics_history['total_loss']) > 0
        has_val = len(self.val_metrics_history.get('episode_reward', [])) > 0
        
        if has_train:
            plt.plot(self.train_metrics_history['total_loss'], label='Train Loss', alpha=0.7)
        if has_val and len(eval_eps) == len(self.val_metrics_history['episode_reward']):
            plt.plot(eval_eps, self.val_metrics_history['episode_reward'], label='Val Reward', alpha=0.7, marker='o')
        plt.xlabel('Episode')
        plt.ylabel('Loss/Reward')
        plt.title('Total Loss / Val Reward')
        if has_train or has_val:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Critic Loss
        plt.subplot(2, 4, 2)
        if self.train_metrics_history['critic_loss']:
            plt.plot(self.train_metrics_history['critic_loss'], label='Train Critic', alpha=0.7)
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss')
        plt.grid(True, alpha=0.3)
        
        # Action Loss
        plt.subplot(2, 4, 3)
        if self.train_metrics_history['action_loss']:
            plt.plot(self.train_metrics_history['action_loss'], label='Train Action', alpha=0.7)
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Action Loss')
        plt.title('Action Loss')
        plt.grid(True, alpha=0.3)
        
        # V Loss
        plt.subplot(2, 4, 4)
        if self.train_metrics_history['v_loss']:
            plt.plot(self.train_metrics_history['v_loss'], label='Train V', alpha=0.7)
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('V Loss')
        plt.title('V Loss')
        plt.grid(True, alpha=0.3)
        
        # State Loss
        plt.subplot(2, 4, 5)
        if self.train_metrics_history['state_loss']:
            plt.plot(self.train_metrics_history['state_loss'], label='Train State', alpha=0.7)
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('State Loss')
        plt.title('State Loss')
        plt.grid(True, alpha=0.3)
        
        # Exploration: latent_noise (proposer σ) + action_noise
        plt.subplot(2, 4, 6)
        has_latent = len(self.train_metrics_history.get('latent_noise', [])) > 0
        has_noise = len(self.train_metrics_history['action_noise']) > 0
        eps_x = eval_eps if len(eval_eps) == len(self.train_metrics_history.get('latent_noise', [])) else list(range(len(self.train_metrics_history.get('latent_noise', []))))
        if has_latent and eps_x:
            plt.plot(eps_x, self.train_metrics_history['latent_noise'], label='Latent noise σ', color='blue', alpha=0.7)
        if has_noise and eps_x and len(eps_x) == len(self.train_metrics_history.get('action_noise', [])):
            plt.plot(eps_x, self.train_metrics_history['action_noise'], label='Action noise', color='red', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.title('Exploration (latent σ + action)')
        if has_latent or has_noise:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Success Rate
        plt.subplot(2, 4, 7)
        if self.val_metrics_history['success_rate'] and len(eval_eps) == len(self.val_metrics_history['success_rate']):
            plt.plot(eval_eps, self.val_metrics_history['success_rate'], label='Val Success Rate', alpha=0.7, marker='o')
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Success Rate')
        plt.grid(True, alpha=0.3)
        
        # Episode Length
        plt.subplot(2, 4, 8)
        if self.val_metrics_history['episode_length'] and len(eval_eps) == len(self.val_metrics_history['episode_length']):
            plt.plot(eval_eps, self.val_metrics_history['episode_length'], label='Val Episode Length', alpha=0.7, marker='o')
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Length')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, f'online_training_curves_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  >> 훈련 곡선 저장: {plot_path}")
