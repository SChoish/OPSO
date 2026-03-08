"""
OGBench preprocessing: state-only sparse goal-conditioned dataset.

- Trajectories from OGBench using `terminals` (boundary only; not success labels). Actions dropped.
- Contexts are current-timestep-centered: for each t (with stride), last valid token = s_t; history window left-padded, right-aligned.
- Goal is sampled from the FUTURE of current t ([t, T) or [t+1, T)), so goal is never in the past relative to current state.
- Sparse rewards relabeled w.r.t. chosen goal; primary control target = last valid transition.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ogbench
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os
import json

def get_goal_reached_config(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Dataset-specific default config for goal_reached. Returns None if dataset is not supported.
    antmaze-* / humanoidmaze-*: position (dims 0,1) distance <= threshold.
    """
    if "antmaze-" in dataset_name:
        return {"pos_dims": (0, 1), "distance_threshold": 0.5}
    if "humanoidmaze-" in dataset_name:
        return {"pos_dims": (0, 1), "distance_threshold": 0.5}
    return None


def goal_reached(
    next_obs: np.ndarray,
    goal: np.ndarray,
    dataset_name: str,
    threshold_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Pluggable: True iff next_obs is considered to have reached goal.
    Uses dataset-specific defaults via get_goal_reached_config when threshold_config is None.
    Raises if dataset unsupported and no threshold_config provided.
    """
    cfg = threshold_config
    if cfg is None:
        cfg = get_goal_reached_config(dataset_name)
    if cfg is None or (isinstance(cfg, dict) and not cfg and get_goal_reached_config(dataset_name) is None):
        raise ValueError(
            f"goal_reached: dataset '{dataset_name}' has no default config and no threshold_config provided. "
            "Pass threshold_config (pos_dims, distance_threshold) or use a supported dataset (antmaze-*, humanoidmaze-*)."
        )
    cfg = cfg or {}
    pos_dims = cfg.get("pos_dims", (0, 1))
    threshold = float(cfg.get("distance_threshold", 0.5))
    if next_obs.ndim == 2:
        next_obs = next_obs.reshape(-1)
    if goal.ndim == 2:
        goal = goal.reshape(-1)
    pos_next = next_obs[pos_dims[0] : pos_dims[1] + 1]
    pos_goal = goal[pos_dims[0] : pos_dims[1] + 1]
    dist = np.linalg.norm(pos_next - pos_goal)
    return dist <= threshold


def compute_sparse_rewards_for_window(
    next_observations: np.ndarray,
    goal: np.ndarray,
    dataset_name: str,
    threshold_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Binary rewards per step: 1 iff next_obs reaches goal. Primary control target is the last valid step."""
    n = len(next_observations)
    rewards = np.zeros(n, dtype=np.float32)
    for i in range(n):
        rewards[i] = 1.0 if goal_reached(next_observations[i], goal, dataset_name, threshold_config) else 0.0
    return rewards


def sample_goal_from_future(
    trajectory_observations: np.ndarray,
    current_t: int,
    traj_len: int,
    strategy: str = "future",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample goal from the FUTURE of current timestep t (so goal is not in the past relative to current state).
    future: u in [t, traj_len), goal = observations[u]. strict_future: u in [t+1, traj_len). final: goal = observations[traj_len-1].
    """
    rng = rng or np.random.default_rng()
    if strategy == "strict_future":
        if current_t + 1 >= traj_len:
            return trajectory_observations[traj_len - 1].copy()
        u = rng.integers(current_t + 1, traj_len)
        return trajectory_observations[u].copy()
    if strategy == "final":
        return trajectory_observations[traj_len - 1].copy()
    # future: [t, traj_len)
    if current_t >= traj_len:
        return trajectory_observations[traj_len - 1].copy()
    u = rng.integers(current_t, traj_len)
    return trajectory_observations[u].copy()


@dataclass
class RawTrajectory:
    """State-only trajectory from OGBench. No rewards here; rewards are relabeled per context using a chosen goal."""
    observations: np.ndarray       # (T, obs_dim)
    next_observations: np.ndarray  # (T, obs_dim)
    masks: Optional[np.ndarray] = None  # (T,) optional; 1=valid, 0=invalid. Not used as success/reward.


def _reconstruct_trajectories_from_terminals(
    observations: np.ndarray,
    next_observations: np.ndarray,
    terminals: np.ndarray,
    masks: Optional[np.ndarray] = None,
    min_length: int = 10,
) -> List[RawTrajectory]:
    """
    Reconstruct trajectories using terminals only. Terminals = trajectory boundary; do NOT use as success/reward.
    Drops actions; state-only. Optionally pass masks from raw dataset (not confused with rewards).
    """
    traj_starts = [0]
    for i in range(len(terminals)):
        if terminals[i]:
            traj_starts.append(i + 1)
    if traj_starts[-1] >= len(observations):
        traj_starts = traj_starts[:-1]
    out = []
    for i in range(len(traj_starts)):
        start_idx = traj_starts[i]
        end_idx = traj_starts[i + 1] if i + 1 < len(traj_starts) else len(observations)
        if end_idx - start_idx < min_length:
            continue
        obs = observations[start_idx:end_idx]
        next_obs = next_observations[start_idx:end_idx]
        m = masks[start_idx:end_idx].copy() if masks is not None else None
        out.append(RawTrajectory(observations=obs, next_observations=next_obs, masks=m))
    return out


def download_ogbench_datasets(
    dataset_names: list = None,
    save_dir: str = "./datasets",
    force_redownload: bool = False,
    compact_dataset: bool = False,
):
    """
    Download OGBench datasets and save state-only flat arrays. Trajectories reconstructed by terminals (boundary only).
    No rewards in saved files; sparse rewards are relabeled at context-creation time.
    """
    if dataset_names is None:
        dataset_names = [
            "humanoidmaze-medium-navigate-v0", "humanoidmaze-large-navigate-v0",
            "humanoidmaze-medium-stitch-v0", "humanoidmaze-large-stitch-v0",
            "antmaze-medium-navigate-v0", "antmaze-large-navigate-v0",
            "antmaze-medium-stitch-v0", "antmaze-large-stitch-v0",
            "antmaze-medium-explore-v0", "antmaze-large-explore-v0",
        ]
    os.makedirs(save_dir, exist_ok=True)
    downloaded_datasets = {}
    for dataset_name in dataset_names:
        dataset_dir = os.path.join(save_dir, dataset_name)
        train_file = os.path.join(dataset_dir, "train.npz")
        val_file = os.path.join(dataset_dir, "val.npz")
        meta_file = os.path.join(dataset_dir, "metadata.json")
        if not force_redownload and os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    metadata = json.load(f)
                downloaded_datasets[dataset_name] = {
                    "train_file": train_file, "val_file": val_file, "meta_file": meta_file,
                    "train_size": metadata.get("train_size", "unknown"),
                    "val_size": metadata.get("val_size", "unknown"),
                    "status": "already_exists",
                }
                continue
            except Exception:
                pass
        try:
            # Raw OGBench: compact_dataset=False gives flat arrays (observations, actions, next_observations, terminals; masks if present)
            try:
                env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, compact_dataset=compact_dataset)
            except TypeError:
                env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_info = {
                "name": dataset_name,
                "train_size": 0,
                "val_size": 0,
                "env_info": {"observation_space": str(env.observation_space), "action_space": str(env.action_space)},
            }
            for label, raw_data in [("train", train_dataset), ("val", val_dataset)]:
                if not isinstance(raw_data, dict) or "observations" not in raw_data or "next_observations" not in raw_data or "terminals" not in raw_data:
                    continue
                observations = np.array(raw_data["observations"])
                next_observations = np.array(raw_data["next_observations"])
                terminals = np.array(raw_data["terminals"])
                masks = np.array(raw_data["masks"]) if "masks" in raw_data else None
                trajs = _reconstruct_trajectories_from_terminals(observations, next_observations, terminals, masks)
                if not trajs:
                    continue
                if label == "train":
                    dataset_info["train_size"] = len(trajs)
                else:
                    dataset_info["val_size"] = len(trajs)
                lengths = np.array([len(t.observations) for t in trajs])
                flat_obs = np.concatenate([t.observations for t in trajs])
                flat_next = np.concatenate([t.next_observations for t in trajs])
                fpath = os.path.join(dataset_dir, "train.npz" if label == "train" else "val.npz")
                np.savez_compressed(fpath, observations=flat_obs, next_observations=flat_next, trajectory_lengths=lengths)
            meta_file = os.path.join(dataset_dir, "metadata.json")
            with open(meta_file, "w") as f:
                json.dump(dataset_info, f, indent=2)
            downloaded_datasets[dataset_name] = {
                "train_file": train_file, "val_file": val_file, "meta_file": meta_file,
                "train_size": dataset_info["train_size"], "val_size": dataset_info["val_size"],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    summary_file = os.path.join(save_dir, "datasets_summary.json")
    with open(summary_file, "w") as f:
        json.dump(downloaded_datasets, f, indent=2)
    return downloaded_datasets


class OGBenchDataset(Dataset):
    """
    State-only sparse goal-conditioned dataset from OGBench.
    Trajectories reconstructed from raw OGBench using terminals (boundary only). Sparse rewards relabeled per context from chosen goal.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_type: str = "train",
        max_trajectories: int = 100,
        normalize: bool = True,
        data_dir: str = "./datasets",
        context_length: int = 100,
        goal_sampling_strategy: str = "future",
        goal_reached_threshold_config: Optional[Dict[str, Any]] = None,
        compact_dataset: bool = False,
        context_stride: int = 1,
        seed: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.normalize = normalize
        self.data_dir = data_dir
        self.context_length = context_length
        self.max_trajectories = max_trajectories
        self.goal_sampling_strategy = goal_sampling_strategy
        self.goal_reached_threshold_config = goal_reached_threshold_config or {}
        self.context_stride = context_stride
        self.seed = seed
        self.trajectories: List[RawTrajectory] = []
        self.contexts: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(self.seed)
        # Resolve goal_reached config: dataset default + user override. Require supported dataset or user config.
        self._thresh_cfg = get_goal_reached_config(dataset_name)
        if self._thresh_cfg is None and not self.goal_reached_threshold_config:
            import warnings
            warnings.warn(
                f"OGBenchDataset: dataset '{dataset_name}' has no default goal_reached config. "
                "Pass goal_reached_threshold_config (pos_dims, distance_threshold) or goal_reached may raise."
            )
        if self._thresh_cfg is None:
            self._thresh_cfg = {}
        self._thresh_cfg = {**self._thresh_cfg, **self.goal_reached_threshold_config}
        if self._load_from_local():
            pass
        else:
            try:
                self.env, self.train_dataset, self.val_dataset = ogbench.make_env_and_datasets(
                    dataset_name, compact_dataset=compact_dataset
                )
            except TypeError:
                self.env, self.train_dataset, self.val_dataset = ogbench.make_env_and_datasets(dataset_name)
            self._load_from_ogbench(dataset_type, max_trajectories)
        self._create_contexts()
        if normalize and len(self.contexts) > 0:
            if dataset_type == "train":
                self._compute_normalization_stats()
            else:
                self._load_normalization_stats_from_train()
    
    def _load_from_local(self) -> bool:
        """Load state-only trajectories from local npz (observations, next_observations, trajectory_lengths). No rewards in file."""
        try:
            dataset_dir = os.path.join(self.data_dir, self.dataset_name)
            if not os.path.exists(dataset_dir):
                return False
            meta_file = os.path.join(dataset_dir, "metadata.json")
            if not os.path.exists(meta_file):
                return False
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)
            fname = "val.npz" if self.dataset_type == "val" else "train.npz"
            data_file = os.path.join(dataset_dir, fname)
            if not os.path.exists(data_file):
                return False
            data = np.load(data_file)
            observations = np.asarray(data["observations"])
            next_observations = np.asarray(data["next_observations"])
            lengths = np.asarray(data["trajectory_lengths"]) if "trajectory_lengths" in data else None
            if lengths is not None:
                start = 0
                for L in lengths:
                    end = start + int(L)
                    if end - start >= 10:
                        self.trajectories.append(RawTrajectory(
                            observations[start:end].copy(),
                            next_observations[start:end].copy(),
                            masks=None,
                        ))
                    start = end
            else:
                terminals = data.get("terminals", None)
                if terminals is not None:
                    terminals = np.asarray(terminals)
                    start_idx = 0
                    for i in range(len(terminals)):
                        if terminals[i] or i == len(terminals) - 1:
                            end_idx = i + 1
                            if end_idx - start_idx > 10:
                                self.trajectories.append(RawTrajectory(
                                    observations[start_idx:end_idx].copy(),
                                    next_observations[start_idx:end_idx].copy(),
                                    masks=None,
                                ))
                            start_idx = end_idx
                elif len(observations) >= 10:
                    self.trajectories.append(RawTrajectory(observations.copy(), next_observations.copy(), masks=None))
            if self.max_trajectories > 0 and len(self.trajectories) > self.max_trajectories:
                self.trajectories = self.trajectories[: self.max_trajectories]
            return len(self.trajectories) > 0
        except Exception:
            return False
    
    def _load_from_ogbench(self, dataset_type: str, max_trajectories: int):
        """Load raw OGBench flat arrays; reconstruct trajectories using terminals only (boundary). Do not create rewards from terminals."""
        raw_data = self.train_dataset if dataset_type == "train" else self.val_dataset
        self.trajectories = []
        if isinstance(raw_data, dict) and "observations" in raw_data and "next_observations" in raw_data and "terminals" in raw_data:
            observations = np.array(raw_data["observations"])
            next_observations = np.array(raw_data["next_observations"])
            terminals = np.array(raw_data["terminals"])
            masks = np.array(raw_data["masks"]) if "masks" in raw_data else None
            self.trajectories = _reconstruct_trajectories_from_terminals(
                observations, next_observations, terminals, masks
            )
            if max_trajectories > 0 and len(self.trajectories) > max_trajectories:
                self.trajectories = self.trajectories[:max_trajectories]
    
    def _create_contexts(self):
        """
        Current-timestep-centered sliding windows. For each current step t (with stride), build one sample
        whose last valid token is s_t. History = observations[max(0,t-L+1):t+1], left-padded to context_length.
        Goal is sampled from the FUTURE of t: u in [t, T) (or [t+1, T) for strict_future). Sparse rewards
        relabeled w.r.t. chosen goal; primary control target is the last valid transition (reward_t at last valid).
        """
        self.contexts = []
        L = self.context_length
        stride = max(1, int(self.context_stride))
        for traj in self.trajectories:
            traj_len = len(traj.observations)
            # t = current timestep (last valid token in context = state at t)
            for t in range(0, traj_len, stride):
                history_start = max(0, t - L + 1)
                history_end = t + 1
                window_len = history_end - history_start
                # Goal from future of t (not from past)
                goal = sample_goal_from_future(
                    traj.observations, t, traj_len, self.goal_sampling_strategy, self._rng
                )
                next_slice = traj.next_observations[history_start:history_end]
                rewards_slice = compute_sparse_rewards_for_window(
                    next_slice, goal, self.dataset_name, self._thresh_cfg
                )
                pad_len = L - window_len
                if pad_len > 0:
                    pad_obs = np.zeros((pad_len, traj.observations.shape[1]), dtype=traj.observations.dtype)
                    pad_next = np.zeros((pad_len, traj.next_observations.shape[1]), dtype=traj.next_observations.dtype)
                    obs = np.vstack([pad_obs, traj.observations[history_start:history_end]])
                    next_obs = np.vstack([pad_next, next_slice])
                    rewards = np.concatenate([np.zeros(pad_len, dtype=np.float32), rewards_slice])
                    mask = np.concatenate([np.zeros(pad_len), np.ones(window_len)])
                else:
                    obs = traj.observations[history_start:history_end].copy()
                    next_obs = next_slice.copy()
                    rewards = rewards_slice
                    mask = np.ones(L)
                dones = np.zeros(L)
                if t == traj_len - 1:
                    dones[L - 1] = 1.0
                last_valid_idx = L - 1
                self.contexts.append({
                    "observations": obs,
                    "rewards": rewards,
                    "next_observations": next_obs,
                    "dones": dones,
                    "mask": mask,
                    "goal": goal,
                    "last_valid_idx": last_valid_idx,
                })
    
    def _compute_normalization_stats(self):
        """Normalization from raw valid trajectory observations only (no padded context data)."""
        if not self.trajectories:
            return
        all_obs = np.vstack([t.observations for t in self.trajectories])
        self.state_mean = np.mean(all_obs, axis=0).astype(np.float32)
        self.state_std = np.std(all_obs, axis=0).astype(np.float32)
        self.state_std = np.where(self.state_std < 1e-8, 1.0, self.state_std)
        if self.dataset_type == "train":
            stats_path = os.path.join(self.data_dir, self.dataset_name, "norm_stats.npz")
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            np.savez_compressed(stats_path, state_mean=self.state_mean, state_std=self.state_std)
    
    def _load_normalization_stats_from_train(self):
        """Load train normalization stats from disk (norm_stats.npz) to avoid rebuilding train set."""
        stats_path = os.path.join(self.data_dir, self.dataset_name, "norm_stats.npz")
        if os.path.isfile(stats_path):
            with np.load(stats_path) as f:
                self.state_mean = np.asarray(f["state_mean"], dtype=np.float32)
                self.state_std = np.asarray(f["state_std"], dtype=np.float32)
            return
        obs_dim = self.contexts[0]["observations"].shape[1] if self.contexts else 1
        self.state_mean = np.zeros(obs_dim, dtype=np.float32)
        self.state_std = np.ones(obs_dim, dtype=np.float32)
    
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
        """Return one context. observations/next_observations/goal normalized. mask: 1=valid, 0=pad; last valid = current state."""
        ctx = self.contexts[idx]
        if self.normalize:
            obs = self.normalize_data(ctx["observations"])
            next_obs = self.normalize_data(ctx["next_observations"])
            goal = self.normalize_data(ctx["goal"].reshape(1, -1)).reshape(-1)
        else:
            obs = ctx["observations"]
            next_obs = ctx["next_observations"]
            goal = ctx["goal"].copy()
        return {
            "observations": torch.FloatTensor(obs),
            "rewards": torch.FloatTensor(ctx["rewards"]),
            "next_observations": torch.FloatTensor(next_obs),
            "dones": torch.FloatTensor(ctx["dones"]),
            "mask": torch.FloatTensor(ctx["mask"]),
            "goal": torch.FloatTensor(goal),
            "last_valid_idx": int(ctx["last_valid_idx"]),
        }


def create_dataloader(
    dataset_name: str,
    dataset_type: str = "train",
    max_trajectories: int = 100,
    batch_size: int = 32,
    normalize: bool = True,
    data_dir: str = "./datasets",
    context_length: int = 100,
    goal_sampling_strategy: str = "future",
    goal_reached_threshold_config: Optional[Dict[str, Any]] = None,
    context_stride: int = 1,
    seed: Optional[int] = None,
):
    """Create DataLoader for state-only sparse goal-conditioned OGBench dataset."""
    dataset = OGBenchDataset(
        dataset_name,
        dataset_type,
        max_trajectories,
        normalize,
        data_dir,
        context_length,
        goal_sampling_strategy=goal_sampling_strategy,
        goal_reached_threshold_config=goal_reached_threshold_config,
        context_stride=context_stride,
        seed=seed,
    )
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Summary: OGBench preprocessing semantics (current-timestep-centered)
# ---------------------------------------------------------------------------
# - Contexts: Current-timestep-centered sliding windows. For each t (stride), last valid token = s_t; history = obs[max(0,t-L+1):t+1], left-padded.
# - Future goal sampling: goal from [t, T) or [t+1, T); goal is never in the past relative to current state.
# - Sparse rewards: relabeled w.r.t. chosen goal; primary control target = last valid transition (reward at last valid = 1 iff next_obs[t] reaches goal).
# - Stride: configurable (default 1) for sampling density.
# - goal_reached: Dataset-specific defaults (antmaze-*, humanoidmaze-*: pos_dims=(0,1), threshold=0.5); unsupported + no config -> warning/error.
# - Seed: optional for reproducible goal sampling.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    downloaded = download_ogbench_datasets()
    dataset, dataloader = create_dataloader(
        "antmaze-medium-navigate-v0", "train", 50, 16, normalize=True, context_length=100, context_stride=1, seed=42
    )
    for batch in dataloader:
        print("Batch keys:", list(batch.keys()))
        print("observations:", batch["observations"].shape, "goal:", batch["goal"].shape, "rewards:", batch["rewards"].shape)
        break
    print("\n--- OGBench preprocessing summary ---")
    print("- Future-goal sampling: goal sampled from [t, T) or [t+1, T); no longer from chunk [start, end).")
    print("- Context construction: current-timestep-centered sliding windows; last valid token = s_t, history = obs[max(0,t-L+1):t+1] left-padded.")
    print("- Stride: configurable (default 1) for sampling density; many samples per trajectory.")
    print("- Dataset-specific goal matching: get_goal_reached_config() for antmaze-*, humanoidmaze-*; unsupported + no config -> warning/error.")
