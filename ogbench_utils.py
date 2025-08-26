import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ogbench
from dataclasses import dataclass
from typing import Optional
import os
import json


@dataclass
class Trajectory:
    """간단한 trajectory 구조체"""
    observations: np.ndarray      # (T, obs_dim)
    rewards: np.ndarray           # (T,)
    next_observations: np.ndarray # (T, obs_dim)
    next_next_observations: np.ndarray # (T, obs_dim) - 추가


def extract_trajectory_from_episode(episode_data) -> Optional[Trajectory]:
    """OGBench 데이터 구조에 맞게 trajectory 추출"""
    try:
        # OGBench 데이터는 이미 trajectory 형태로 되어 있음
        # episode_data는 딕셔너리이고 각 키에 전체 trajectory가 들어있음
        
        if not isinstance(episode_data, dict):
            print(f"  에피소드 데이터가 딕셔너리가 아님: {type(episode_data)}")
            return None
            
        print(f"  에피소드 데이터 키: {list(episode_data.keys())}")
        
        # observations, rewards, next_observations 확인
        if 'observations' not in episode_data:
            print(f"  observations 키가 없음")
            return None
            
        if 'next_observations' not in episode_data:
            print(f"  next_observations 키가 없음")
            return None
            
        if 'rewards' not in episode_data:
            print(f"  rewards 키가 없음")
            return None
            
        observations = np.array(episode_data['observations'])
        next_observations = np.array(episode_data['next_observations'])
        rewards = np.array(episode_data['rewards'])
        
        # next_next_observations 생성 (한 스텝씩 시프트)
        if len(observations) > 1:
            next_next_observations = np.vstack([
                next_observations[1:],  # 한 스텝 시프트
                next_observations[-1:]  # 마지막 스텝 복제
            ])
        else:
            next_next_observations = next_observations.copy()
        
        print(f"  observations shape: {observations.shape}")
        print(f"  next_observations shape: {next_observations.shape}")
        print(f"  next_next_observations shape: {next_next_observations.shape}")
        print(f"  rewards shape: {rewards.shape}")
        
        if len(observations) < 2:
            print(f"  observations가 너무 짧음: {len(observations)}")
            return None
            
        return Trajectory(observations, rewards, next_observations, next_next_observations)
        
    except Exception as e:
        print(f"  trajectory 추출 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_ogbench_datasets(dataset_names: list = None, save_dir: str = './datasets', force_redownload: bool = False):
    """
    OGBench 데이터셋들을 다운로드하고 지정된 폴더에 저장합니다.
    
    Args:
        dataset_names: 다운로드할 데이터셋 이름 리스트 (None이면 기본 데이터셋들)
        save_dir: 저장할 폴더 경로
        force_redownload: 이미 가공된 파일이 있어도 강제로 다시 다운로드할지 여부
    """
    if dataset_names is None:
        dataset_names = [
            # humanoidmaze 데이터셋들
            'humanoidmaze-medium-navigate-v0',
            'humanoidmaze-large-navigate-v0',
            'humanoidmaze-medium-stitch-v0',
            'humanoidmaze-large-stitch-v0',
            
            # antmaze 데이터셋들
            'antmaze-medium-navigate-v0',
            'antmaze-large-navigate-v0',
            'antmaze-medium-stitch-v0',
            'antmaze-large-stitch-v0',
            'antmaze-medium-explore-v0',
            'antmaze-large-explore-v0'
        ]
    
    # 저장 폴더 생성
    os.makedirs(save_dir, exist_ok=True)
    
    downloaded_datasets = {}
    
    for dataset_name in dataset_names:
        print(f"처리 중: {dataset_name}")
        
        # 이미 가공된 파일이 있는지 확인
        dataset_dir = os.path.join(save_dir, dataset_name)
        train_file = os.path.join(dataset_dir, 'train.npz')
        val_file = os.path.join(dataset_dir, 'val.npz')
        meta_file = os.path.join(dataset_dir, 'metadata.json')
        
        if not force_redownload and os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(meta_file):
            print(f"  이미 가공된 파일이 존재함 - 스킵")
            
            # 기존 파일 정보 로드
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                # 파일 크기 확인
                train_size = os.path.getsize(train_file)
                val_size = os.path.getsize(val_file)
                
                downloaded_datasets[dataset_name] = {
                    'train_file': train_file,
                    'val_file': val_file,
                    'meta_file': meta_file,
                    'train_size': metadata.get('train_size', 'unknown'),
                    'val_size': metadata.get('val_size', 'unknown'),
                    'status': 'already_exists'
                }
                
                print(f"  기존 파일 사용: 훈련 {metadata.get('train_size', 'unknown')}, 검증 {metadata.get('val_size', 'unknown')}")
                continue
                
            except Exception as e:
                print(f"  기존 파일 정보 로드 실패 - 다시 다운로드: {e}")
        
        try:
            # 데이터셋 로드 (자동 다운로드)
            env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)
            
            # 데이터셋 정보 저장
            dataset_info = {
                'name': dataset_name,
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'env_info': {
                    'observation_space': str(env.observation_space),
                    'action_space': str(env.action_space)
                }
            }
            
            # 실제 데이터 구조 확인
            print(f"  훈련 데이터 타입: {type(train_dataset)}")
            if isinstance(train_dataset, dict):
                print(f"  훈련 데이터 키: {list(train_dataset.keys())}")
                for key in train_dataset.keys():
                    print(f"    {key}: {type(train_dataset[key])}, shape: {train_dataset[key].shape if hasattr(train_dataset[key], 'shape') else 'N/A'}")
            elif hasattr(train_dataset, '__len__'):
                print(f"  훈련 데이터 길이: {len(train_dataset)}")
                if len(train_dataset) > 0:
                    print(f"  훈련 데이터 첫 번째 에피소드 타입: {type(train_dataset[0])}")
                    if hasattr(train_dataset[0], '__len__'):
                        print(f"  훈련 데이터 첫 번째 에피소드 길이: {len(train_dataset[0])}")
            
            # 검증 데이터 구조 확인
            print(f"  검증 데이터 타입: {type(val_dataset)}")
            if isinstance(val_dataset, dict):
                print(f"  검증 데이터 키: {list(val_dataset.keys())}")
                for key in val_dataset.keys():
                    print(f"    {key}: {type(val_dataset[key])}, shape: {val_dataset[key].shape if hasattr(val_dataset[key], 'shape') else 'N/A'}")
            elif hasattr(val_dataset, '__len__'):
                print(f"  검증 데이터 길이: {len(val_dataset)}")
                if len(val_dataset) > 0:
                    print(f"  검증 데이터 첫 번째 에피소드 타입: {type(val_dataset[0])}")
                    if hasattr(val_dataset[0], '__len__'):
                        print(f"  검증 데이터 첫 번째 에피소드 길이: {len(val_dataset[0])}")
            
            # 훈련 데이터 처리
            print(f"  훈련 데이터 처리 중... (길이: {len(train_dataset['observations'])})")
            train_data = []
            
            # OGBench 데이터는 이미 trajectory 형태로 되어 있음
            # 각 키에 전체 trajectory가 들어있음
            observations = np.array(train_dataset['observations'])
            next_observations = np.array(train_dataset['next_observations'])
            terminals = np.array(train_dataset['terminals'])
            
            # next_next_observations 생성 (한 스텝씩 시프트)
            next_next_observations = np.vstack([
                next_observations[1:],  # 한 스텝 시프트
                next_observations[-1:]  # 마지막 스텝 복제
            ])
            
            # rewards 생성: terminals=1이면 reward=1, 아니면 reward=0
            rewards = np.where(terminals == 1, 1.0, 0.0)
            
            print(f"  observations shape: {observations.shape}")
            print(f"  next_observations shape: {next_observations.shape}")
            print(f"  next_next_observations shape: {next_next_observations.shape}")
            print(f"  rewards shape: {rewards.shape}")
            print(f"  terminals shape: {terminals.shape}")
            
            # terminals 기반으로 trajectory 분할
            traj_starts = [0]
            for i in range(len(terminals)):
                if terminals[i]:  # episode가 끝나는 지점
                    traj_starts.append(i + 1)
            
            # 마지막 trajectory 시작점이 데이터 끝을 넘지 않도록 조정
            if traj_starts[-1] >= len(observations):
                traj_starts = traj_starts[:-1]
            
            print(f"  terminals 기반 trajectory 수: {len(traj_starts)}")
            
            for i in range(len(traj_starts)):
                start_idx = traj_starts[i]
                if i < len(traj_starts) - 1:
                    end_idx = traj_starts[i + 1]
                else:
                    end_idx = len(observations)
                
                # 최소 길이 체크 (너무 짧은 trajectory 제외)
                if end_idx - start_idx >= 10:  # 최소 10 스텝
                    traj = Trajectory(
                        observations[start_idx:end_idx],
                        rewards[start_idx:end_idx],
                        next_observations[start_idx:end_idx],
                        next_next_observations[start_idx:end_idx]
                    )
                    train_data.append(traj)
            
            # 검증 데이터 처리
            print(f"  검증 데이터 처리 중... (길이: {len(val_dataset['observations'])})")
            val_data = []
            
            val_observations = np.array(val_dataset['observations'])
            val_next_observations = np.array(val_dataset['next_observations'])
            val_terminals = np.array(val_dataset['terminals'])
            
            # val_next_next_observations 생성 (한 스텝씩 시프트)
            val_next_next_observations = np.vstack([
                val_next_observations[1:],  # 한 스텝 시프트
                val_next_observations[-1:]  # 마지막 스텝 복제
            ])
            
            # rewards 생성: terminals=1이면 reward=1, 아니면 reward=0
            val_rewards = np.where(val_terminals == 1, 1.0, 0.0)
            
            print(f"  val_observations shape: {val_observations.shape}")
            print(f"  val_next_observations shape: {val_next_observations.shape}")
            print(f"  val_next_next_observations shape: {val_next_next_observations.shape}")
            print(f"  val_rewards shape: {val_rewards.shape}")
            print(f"  val_terminals shape: {val_terminals.shape}")
            
            # terminals 기반으로 trajectory 분할
            val_traj_starts = [0]
            for i in range(len(val_terminals)):
                if val_terminals[i]:  # episode가 끝나는 지점
                    val_traj_starts.append(i + 1)
            
            # 마지막 trajectory 시작점이 데이터 끝을 넘지 않도록 조정
            if val_traj_starts[-1] >= len(val_observations):
                val_traj_starts = val_traj_starts[:-1]
            
            print(f"  검증 데이터 terminals 기반 trajectory 수: {len(val_traj_starts)}")
            
            for i in range(len(val_traj_starts)):
                start_idx = val_traj_starts[i]
                if i < len(val_traj_starts) - 1:
                    end_idx = val_traj_starts[i + 1]
                else:
                    end_idx = len(val_observations)
                
                # 최소 길이 체크 (너무 짧은 trajectory 제외)
                if end_idx - start_idx >= 10:  # 최소 10 스텝
                    traj = Trajectory(
                        val_observations[start_idx:end_idx],
                        val_rewards[start_idx:end_idx],
                        val_next_observations[start_idx:end_idx],
                        val_next_next_observations[start_idx:end_idx]
                    )
                    val_data.append(traj)
            
            # 데이터셋별 폴더 생성
            dataset_dir = os.path.join(save_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 훈련 데이터 저장
            train_file = os.path.join(dataset_dir, 'train.npz')
            if train_data:
                train_obs = np.concatenate([d.observations for d in train_data])
                train_rewards = np.concatenate([d.rewards for d in train_data])
                train_next_obs = np.concatenate([d.next_observations for d in train_data])
                train_next_next_obs = np.concatenate([d.next_next_observations for d in train_data])
                
                np.savez_compressed(train_file, 
                                  observations=train_obs,
                                  rewards=train_rewards,
                                  next_observations=train_next_obs,
                                  next_next_observations=train_next_next_obs)
            
            # 검증 데이터 저장
            val_file = os.path.join(dataset_dir, 'val.npz')
            if val_data:
                val_obs = np.concatenate([d.observations for d in val_data])
                val_rewards = np.concatenate([d.rewards for d in val_data])
                val_next_obs = np.concatenate([d.next_observations for d in val_data])
                val_next_next_obs = np.concatenate([d.next_next_observations for d in val_data])
                
                np.savez_compressed(val_file, 
                                  observations=val_obs,
                                  rewards=val_rewards,
                                  next_observations=val_next_obs,
                                  next_next_observations=val_next_next_obs)
            
            # 메타데이터 저장
            meta_file = os.path.join(dataset_dir, 'metadata.json')
            with open(meta_file, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            downloaded_datasets[dataset_name] = {
                'train_file': train_file,
                'val_file': val_file,
                'meta_file': meta_file,
                'train_size': len(train_data),
                'val_size': len(val_data)
            }
            
            print(f"  완료: {dataset_name} - 훈련: {len(train_data)}, 검증: {len(val_data)}")
            
        except Exception as e:
            print(f"  오류: {dataset_name} 다운로드 실패 - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 전체 요약 저장
    summary_file = os.path.join(save_dir, 'datasets_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(downloaded_datasets, f, indent=2)
    
    print(f"\n다운로드 완료! 총 {len(downloaded_datasets)}개 데이터셋")
    print(f"저장 위치: {save_dir}")
    print(f"요약 파일: {summary_file}")
    
    return downloaded_datasets


class OGBenchDataset(Dataset):
    """OGBench 데이터를 위한 간단한 PyTorch Dataset"""
    
    def __init__(self, dataset_name: str, dataset_type: str = 'train', 
                 max_trajectories: int = 100, normalize: bool = True,
                 data_dir: str = './datasets', context_length: int = 100):
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.data_dir = data_dir
        self.context_length = context_length
        self.trajectories = []  # 초기화 추가
        self.contexts = []  # context_length 길이의 데이터 조각들
        
        # 로컬 데이터에서 로드 시도
        if self._load_from_local():
            print(f"로컬 데이터에서 로드: {dataset_name}")
        else:
            # OGBench에서 직접 로드
            print(f"OGBench에서 직접 로드: {dataset_name}")
            self.env, self.train_dataset, self.val_dataset = ogbench.make_env_and_datasets(dataset_name)
            self._load_from_ogbench(dataset_type, max_trajectories)
        
        # context 생성
        self._create_contexts()
        
        # 정규화 통계 계산
        if normalize and len(self.contexts) > 0:
            if dataset_type == 'train':
                # 훈련 데이터: 실제 통계 계산
                self._compute_normalization_stats()
            else:
                # 검증 데이터: 훈련 데이터의 통계 로드 시도
                self._load_normalization_stats_from_train()
    
    def _load_from_local(self) -> bool:
        """로컬 저장된 데이터에서 로드"""
        try:
            dataset_dir = os.path.join(self.data_dir, self.dataset_name)
            if not os.path.exists(dataset_dir):
                return False
            
            # 메타데이터 확인
            meta_file = os.path.join(dataset_dir, 'metadata.json')
            if not os.path.exists(meta_file):
                return False
            
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.metadata = metadata
            
            # data.npz 파일에서 데이터 로드
            data_file = os.path.join(dataset_dir, 'data.npz')
            if not os.path.exists(data_file):
                return False
            
            # npz 파일 로드
            data = np.load(data_file)
            observations = data['observations']
            rewards = data['rewards']
            next_observations = data['next_observations']
            next_next_observations = data['next_next_observations']
            
            # terminals 정보로 trajectory 분할
            terminals = data.get('terminals', np.zeros(len(observations)))
            
            start_idx = 0
            for i in range(len(terminals)):
                if terminals[i] or i == len(terminals) - 1:  # episode 종료 또는 마지막 스텝
                    end_idx = i + 1
                    
                    if end_idx - start_idx > 10:  # 최소 길이 체크
                        traj = Trajectory(
                            observations[start_idx:end_idx],
                            rewards[start_idx:end_idx],
                            next_observations[start_idx:end_idx],
                            next_next_observations[start_idx:end_idx]
                        )
                        self.trajectories.append(traj)
                    
                    start_idx = end_idx
            
            print(f"  로컬에서 {len(self.trajectories)}개 trajectory 로드됨")
            return True
            
        except Exception as e:
            print(f"로컬 로드 실패: {e}")
            return False
    
    def _load_from_ogbench(self, dataset_type: str, max_trajectories: int):
        """OGBench에서 직접 데이터 로드"""
        if dataset_type == 'train':
            raw_data = self.train_dataset
        else:
            raw_data = self.val_dataset
            
        # Trajectory 추출
        self.trajectories = []
        
        # OGBench 데이터는 딕셔너리 형태로 되어 있음
        if isinstance(raw_data, dict):
            # 딕셔너리인 경우 이미 trajectory 형태
            observations = np.array(raw_data['observations'])
            next_observations = np.array(raw_data['next_observations'])
            terminals = np.array(raw_data['terminals'])
            
            # next_next_observations 생성 (한 스텝씩 시프트)
            next_next_observations = np.vstack([
                next_observations[1:],  # 한 스텝 시프트
                next_observations[-1:]  # 마지막 스텝 복제
            ])
            
            # rewards 생성: terminals=1이면 reward=1, 아니면 reward=0
            rewards = np.where(terminals == 1, 1.0, 0.0)
            
            print(f"  observations shape: {observations.shape}")
            print(f"  rewards shape: {rewards.shape}")
            print(f"  next_observations shape: {next_observations.shape}")
            print(f"  next_next_observations shape: {next_next_observations.shape}")
            print(f"  terminals shape: {terminals.shape}")
            
            # terminals 기반으로 trajectory 분할
            traj_starts = [0]
            for i in range(len(terminals)):
                if terminals[i]:  # episode가 끝나는 지점
                    traj_starts.append(i + 1)
            
            # 마지막 trajectory 시작점이 데이터 끝을 넘지 않도록 조정
            if traj_starts[-1] >= len(observations):
                traj_starts = traj_starts[:-1]
            
            # max_trajectories 제한 적용
            traj_starts = traj_starts[:max_trajectories]
            
            print(f"  OGBench에서 terminals 기반 {len(traj_starts)}개 trajectory 생성")
            
            for i in range(len(traj_starts)):
                start_idx = traj_starts[i]
                if i < len(traj_starts) - 1:
                    end_idx = traj_starts[i + 1]
                else:
                    end_idx = len(observations)
                
                # 최소 길이 체크 (너무 짧은 trajectory 제외)
                if end_idx - start_idx >= 10:  # 최소 10 스텝
                    traj = Trajectory(
                        observations[start_idx:end_idx],
                        rewards[start_idx:end_idx],
                        next_observations[start_idx:end_idx],
                        next_next_observations[start_idx:end_idx]
                    )
                    self.trajectories.append(traj)
        else:
            # 리스트인 경우 기존 방식 사용
            for episode_idx, episode_data in enumerate(raw_data[:max_trajectories]):
                traj = extract_trajectory_from_episode(episode_data)
                if traj:
                    self.trajectories.append(traj)
    
    def _create_contexts(self):
        """trajectory를 context_length 길이의 조각들로 분할"""
        self.contexts = []
        
        for traj in self.trajectories:
            traj_length = len(traj.observations)
            
            # context_length 단위로 분할
            for start_idx in range(0, traj_length, self.context_length):
                end_idx = start_idx + self.context_length
                
                # 패딩이 필요한 경우
                if end_idx > traj_length:
                    # 패딩 생성
                    pad_length = end_idx - traj_length
                    
                    # 마지막 상태로 패딩
                    last_obs = traj.observations[-1]
                    last_next_obs = traj.next_observations[-1]
                    
                    # 패딩된 데이터 생성
                    padded_obs = np.vstack([
                        traj.observations[start_idx:],
                        np.tile(last_obs, (pad_length, 1))
                    ])
                    padded_next_obs = np.vstack([
                        traj.next_observations[start_idx:],
                        np.tile(last_next_obs, (pad_length, 1))
                    ])
                    padded_next_next_obs = np.vstack([
                        traj.next_next_observations[start_idx:],
                        np.tile(traj.next_next_observations[-1], (pad_length, 1))
                    ])
                    padded_rewards = np.concatenate([
                        traj.rewards[start_idx:],
                        np.zeros(pad_length)
                    ])
                    
                    # dones와 mask 생성
                    # 실제 데이터 부분: mask=1, dones=0 (episode 중간)
                    # 패딩 부분: mask=0, dones=1 (episode 종결)
                    actual_length = traj_length - start_idx
                    dones = np.concatenate([
                        np.zeros(actual_length),  # 실제 데이터: 미종결
                        np.ones(pad_length)       # 패딩: 종결
                    ])
                    mask = np.concatenate([
                        np.ones(actual_length),   # 실제 데이터: 유효
                        np.zeros(pad_length)      # 패딩: 무효
                    ])
                else:
                    # 패딩 불필요
                    padded_obs = traj.observations[start_idx:end_idx]
                    padded_next_obs = traj.next_observations[start_idx:end_idx]
                    padded_next_next_obs = traj.next_next_observations[start_idx:end_idx]
                    padded_rewards = traj.rewards[start_idx:end_idx]
                    
                    # dones와 mask 생성 (모두 유효한 데이터)
                    dones = np.zeros(self.context_length)  # 모두 미종결
                    mask = np.ones(self.context_length)    # 모두 유효
                
                # goal_obs 만들기 (state-only라면 같은 차원)
                goal_indices = np.where(padded_rewards == 1)[0]
                if len(goal_indices) > 0:
                    g = padded_obs[goal_indices[0]]         # (obs_dim,)
                else:
                    g = padded_obs[-1]                     # 마지막 상태를 goal로

                goal_obs = np.tile(g, (self.context_length, 1))   # (L, obs_dim)
                
                # context 저장
                context = {
                    'observations': padded_obs,
                    'rewards': padded_rewards,
                    'next_observations': padded_next_obs,
                    'next_next_observations': padded_next_next_obs,
                    'dones': dones,
                    'mask': mask,
                    'goal_obs': goal_obs,            # ★ 추가
                }
                self.contexts.append(context)
        
        print(f"  {len(self.contexts)}개 context 생성 (context_length: {self.context_length})")
    
    def _compute_normalization_stats(self):
        """정규화를 위한 통계 계산 - 전체 데이터셋의 state(observations) 통계 사용"""
        if not self.contexts:
            return
            
        all_obs = []
        
        for context in self.contexts:
            all_obs.extend(context['observations'])
        
        all_obs = np.array(all_obs)
        
        # 전체 데이터셋의 state 통계 계산
        self.state_mean = np.mean(all_obs, axis=0)
        self.state_std = np.std(all_obs, axis=0)
        self.state_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)  # 0으로 나누기 방지
        
        print(f"  전체 데이터셋 state 통계 계산 완료: mean shape {self.state_mean.shape}, std shape {self.state_std.shape}")
    
    def _load_normalization_stats_from_train(self):
        """훈련 데이터의 정규화 통계를 로드"""
        try:
            # 훈련 데이터셋에서 통계 로드
            train_dataset = OGBenchDataset(
                self.dataset_name, 'train', 
                max_trajectories=1000,  # 충분한 데이터로 통계 계산
                normalize=False,  # 정규화 없이 원본 데이터로 통계 계산
                data_dir=self.data_dir, 
                context_length=self.context_length
            )
            
            if len(train_dataset.contexts) > 0:
                # 훈련 데이터의 통계 계산
                all_obs = []
                for context in train_dataset.contexts:
                    all_obs.extend(context['observations'])
                
                all_obs = np.array(all_obs)
                self.state_mean = np.mean(all_obs, axis=0)
                self.state_std = np.std(all_obs, axis=0)
                self.state_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)
                
                print(f"  훈련 데이터에서 정규화 통계 로드 완료: mean shape {self.state_mean.shape}, std shape {self.state_std.shape}")
            else:
                # 폴백: 기본값으로 초기화
                obs_dim = self.contexts[0]['observations'].shape[1] if self.contexts else 1
                self.state_mean = np.zeros(obs_dim)
                self.state_std = np.ones(obs_dim)
                print(f"  훈련 데이터 없음, 기본값으로 초기화: dim {obs_dim}")
                
        except Exception as e:
            # 폴백: 기본값으로 초기화
            obs_dim = self.contexts[0]['observations'].shape[1] if self.contexts else 1
            self.state_mean = np.zeros(obs_dim)
            self.state_std = np.ones(obs_dim)
            print(f"  정규화 통계 로드 실패, 기본값으로 초기화: {e}")
    
    def normalize_data(self, data: np.ndarray, data_type: str = 'state') -> np.ndarray:
        """데이터 정규화 - 모든 데이터를 state 통계로 정규화"""
        if not self.normalize:
            return data
            
        # 모든 데이터를 state 통계로 정규화
        return (data - self.state_mean) / self.state_std
    
    def denormalize_data(self, data: np.ndarray, data_type: str = 'state') -> np.ndarray:
        """데이터 역정규화 - state 통계로 역정규화"""
        if not self.normalize:
            return data
            
        # 모든 데이터를 state 통계로 역정규화
        return data * self.state_std + self.state_mean
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        """단일 context 반환"""
        context = self.contexts[idx]
        
        # 정규화 적용 - 모든 데이터를 state 통계로 정규화
        if self.normalize:
            obs = self.normalize_data(context['observations'])
            next_obs = self.normalize_data(context['next_observations'])
            next_next_obs = self.normalize_data(context['next_next_observations'])
            goal_obs = self.normalize_data(context['goal_obs'])
        else:
            obs = context['observations']
            next_obs = context['next_observations']
            next_next_obs = context['next_next_observations']
            goal_obs = context['goal_obs']
        
        return {
            'observations': torch.FloatTensor(obs),
            'rewards': torch.FloatTensor(context['rewards']),
            'next_observations': torch.FloatTensor(next_obs),
            'next_next_observations': torch.FloatTensor(next_next_obs),
            'dones': torch.FloatTensor(context['dones']),
            'mask': torch.FloatTensor(context['mask']),
            'goal_obs': torch.FloatTensor(goal_obs)
        }


def create_dataloader(dataset_name: str, dataset_type: str = 'train', 
                     max_trajectories: int = 100, batch_size: int = 32,
                     normalize: bool = True, data_dir: str = './datasets',
                     context_length: int = 100):
    """간단한 DataLoader 생성"""
    dataset = OGBenchDataset(dataset_name, dataset_type, max_trajectories, normalize, data_dir, context_length)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 사용 예시
if __name__ == "__main__":
    # 1. 데이터셋 다운로드
    print("=== OGBench 데이터셋 다운로드 ===")
    downloaded = download_ogbench_datasets()
    
    # 2. 로컬 데이터에서 DataLoader 생성
    print("\n=== 로컬 데이터로 DataLoader 생성 ===")
    dataset, dataloader = create_dataloader('humanoidmaze-large-navigate-v0', 'train', 50, 16, normalize=True, context_length=100)
    
    print(f"정규화 통계:")
    if dataset.normalize:
        print(f"  전체 데이터셋 state - mean: {dataset.state_mean[:5]}..., std: {dataset.state_std[:5]}...")
    
    for batch in dataloader:
        obs = batch['observations']        # (batch_size, context_length, obs_dim)
        rewards = batch['rewards']         # (batch_size, context_length)
        next_obs = batch['next_observations']  # (batch_size, context_length, obs_dim)
        next_next_obs = batch['next_next_observations']  # (batch_size, context_length, obs_dim)
        print(f"Batch shapes: obs {obs.shape}, rewards {rewards.shape}, next_obs {next_obs.shape}, next_next_obs {next_next_obs.shape}")
        
        # 정규화된 데이터 확인
        if dataset.normalize:
            print(f"정규화된 관찰값 범위: [{obs.min():.3f}, {obs.max():.3f}]")
            print(f"정규화된 다음 관찰값 범위: [{next_obs.min():.3f}, {next_obs.max():.3f}]")
            print(f"정규화된 다음-다음 관찰값 범위: [{next_next_obs.min():.3f}, {next_next_obs.max():.3f}]")
        break
