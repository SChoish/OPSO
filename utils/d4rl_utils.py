"""
D4RL AntMaze 데이터 로드. OGBench와 동일한 batch 형식: left-pad, right-aligned valid,
observations, next_observations, rewards, dones, mask, goal (단일 벡터), last_valid_idx.
사용: create_dataloader_d4rl('antmaze-medium-diverse-v0', 'train', max_trajectories, batch_size, ...)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import os
import json


@dataclass
class Trajectory:
    observations: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray


# D4RL antmaze 이름 (env.get_dataset() 지원)
D4RL_ANTMAZE_NAMES = [
    "antmaze-umaze-v0",
    "antmaze-umaze-diverse-v0",
    "antmaze-medium-play-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-large-play-v0",
    "antmaze-large-diverse-v0",
]


def load_d4rl_dataset(dataset_name: str):
    """D4RL 데이터셋 로드. gym + d4rl 필요."""
    try:
        import gym
        import d4rl
    except ImportError as e:
        raise ImportError("D4RL 사용 시 gym, d4rl 설치 필요: pip install gym d4rl") from e
    env = gym.make(dataset_name)
    data = env.get_dataset()
    return env, data


def trajectories_from_d4rl(data: dict, max_trajectories: int = -1) -> list:
    """D4RL dataset dict -> list of Trajectory. terminals로 에피소드 분할."""
    obs = np.array(data["observations"], dtype=np.float32)
    rewards = np.array(data["rewards"], dtype=np.float32)
    terminals = np.array(data["terminals"], dtype=np.float32)
    if "timeouts" in data:
        timeouts = np.array(data["timeouts"])
        dones = np.logical_or(terminals, timeouts)
    else:
        dones = terminals
    n = len(obs)
    next_obs = np.roll(obs, -1, axis=0)
    next_obs[-1] = obs[-1]

    traj_starts = [0]
    for i in range(n - 1):
        if dones[i]:
            traj_starts.append(i + 1)
    if traj_starts[-1] >= n:
        traj_starts = traj_starts[:-1]

    trajectories = []
    for i in range(len(traj_starts)):
        start = traj_starts[i]
        end = traj_starts[i + 1] if i + 1 < len(traj_starts) else n
        if end - start < 10:
            continue
        trajectories.append(
            Trajectory(
                observations=obs[start:end],
                rewards=rewards[start:end],
                next_observations=next_obs[start:end],
            )
        )
        if max_trajectories > 0 and len(trajectories) >= max_trajectories:
            break
    return trajectories


class D4RLDataset(Dataset):
    """D4RL AntMaze. OGBenchDataset와 동일한 __getitem__ 형식."""

    def __init__(
        self,
        dataset_name: str,
        dataset_type: str = "train",
        max_trajectories: int = 500,
        normalize: bool = True,
        data_dir: str = "./datasets",
        context_length: int = 100,
    ):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.normalize = normalize
        self.data_dir = data_dir
        self.context_length = context_length
        self.max_trajectories = max_trajectories
        self.trajectories = []
        self.contexts = []

        env, data = load_d4rl_dataset(dataset_name)
        self.obs_dim = data["observations"].shape[1]
        env.close()

        self.trajectories = trajectories_from_d4rl(
            data, 999999 if max_trajectories <= 0 else max_trajectories
        )
        if not self.trajectories:
            raise ValueError(f"D4RL에서 trajectory를 못 찾음: {dataset_name}")
        print(f"  D4RL {dataset_name}: {len(self.trajectories)} trajectories")
        self._create_contexts()
        if normalize and self.contexts:
            self._compute_normalization_stats()
        else:
            obs_dim = self.contexts[0]["observations"].shape[1] if self.contexts else self.obs_dim
            self.state_mean = np.zeros(obs_dim, dtype=np.float32)
            self.state_std = np.ones(obs_dim, dtype=np.float32)

    def _create_contexts(self):
        """Left-pad, right-aligned valid (OGBench 규약). goal은 단일 벡터."""
        L = self.context_length
        self.contexts = []
        for traj in self.trajectories:
            traj_len = len(traj.observations)
            for start_idx in range(0, traj_len, L):
                end_idx = min(start_idx + L, traj_len)
                actual_len = end_idx - start_idx
                pad_left = L - actual_len

                obs_slice = traj.observations[start_idx:end_idx]
                next_slice = traj.next_observations[start_idx:end_idx]
                rewards_slice = traj.rewards[start_idx:end_idx]

                padded_obs = np.zeros((L, self.obs_dim), dtype=np.float32)
                padded_obs[pad_left:] = obs_slice
                padded_next = np.zeros((L, self.obs_dim), dtype=np.float32)
                padded_next[pad_left:] = next_slice
                padded_rewards = np.zeros(L, dtype=np.float32)
                padded_rewards[pad_left:] = rewards_slice

                mask = np.zeros(L, dtype=np.float32)
                mask[pad_left:] = 1.0
                dones = np.zeros(L, dtype=np.float32)
                if end_idx == traj_len:
                    dones[L - 1] = 1.0

                goal_idx = np.where(rewards_slice >= 0.99)[0]
                if len(goal_idx) > 0:
                    goal = obs_slice[goal_idx[0]]
                else:
                    goal = obs_slice[-1]

                self.contexts.append({
                    "observations": padded_obs,
                    "rewards": padded_rewards,
                    "next_observations": padded_next,
                    "dones": dones,
                    "mask": mask,
                    "goal": goal,
                    "last_valid_idx": L - 1,
                })
        print(f"  {len(self.contexts)}개 context 생성 (context_length: {L}, left-pad right-align)")

    def _compute_normalization_stats(self):
        all_obs = np.vstack([t.observations for t in self.trajectories])
        self.state_mean = np.mean(all_obs, axis=0).astype(np.float32)
        self.state_std = np.std(all_obs, axis=0).astype(np.float32)
        self.state_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return x
        return (x - self.state_mean) / self.state_std

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        ctx = self.contexts[idx]
        if self.normalize:
            obs = self._norm(ctx["observations"])
            next_obs = self._norm(ctx["next_observations"])
            goal = self._norm(ctx["goal"])
        else:
            obs = ctx["observations"]
            next_obs = ctx["next_observations"]
            goal = ctx["goal"]
        return {
            "observations": torch.FloatTensor(obs),
            "rewards": torch.FloatTensor(ctx["rewards"]),
            "next_observations": torch.FloatTensor(next_obs),
            "dones": torch.FloatTensor(ctx["dones"]),
            "mask": torch.FloatTensor(ctx["mask"]),
            "goal": torch.FloatTensor(goal),
            "last_valid_idx": ctx["last_valid_idx"],
        }


def create_dataloader_d4rl(
    dataset_name: str,
    dataset_type: str = "train",
    max_trajectories: int = 500,
    batch_size: int = 32,
    normalize: bool = True,
    data_dir: str = "./datasets",
    context_length: int = 100,
):
    """D4RL용 DataLoader. OGBench create_dataloader와 동일하게 (dataset, loader) 반환. val이면 전체 로드 후 뒤 10%를 검증용."""
    dataset = D4RLDataset(
        dataset_name,
        dataset_type="train",
        max_trajectories=max_trajectories if max_trajectories > 0 else 999999,  # -1 = 전체
        normalize=normalize,
        data_dir=data_dir,
        context_length=context_length,
    )
    n = len(dataset)
    if dataset_type == "val" and n >= 10:
        val_size = max(1, n // 10)
        subset = torch.utils.data.Subset(dataset, range(n - val_size, n))
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        return dataset, loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader
