"""
OPSO 진입점: python main.py offline ... / python main.py online ...
"""
import argparse
import torch

from utils.logging_utils import get_logger

LOG = get_logger("main")


def cmd_offline(args):
    from config import get_offline_config
    from trainers.offline_trainer import OfflineTrainer
    from utils.ogbench_utils import download_ogbench_datasets

    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("env", "command", "func", "use_focal_loss")}
    if getattr(args, "use_focal_loss", False):
        overrides["use_focal_loss"] = True
    if "dataset" in overrides:
        overrides["dataset_name"] = overrides.pop("dataset")
    if "device" not in overrides:
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    env = getattr(args, "env", None) or overrides.get("dataset_name")
    config = get_offline_config(overrides, env=env)
    config_env = env or "default"
    LOG.info(f"설정 파일: config/{config_env}.yaml")

    if config.get("data_source", "ogbench") == "ogbench":
        LOG.info("OGBench 데이터셋 확인 중...")
        download_ogbench_datasets()

    LOG.info("=" * 50)
    LOG.info(f"오프라인 훈련: {config['dataset_name']} (source: {config.get('data_source', 'ogbench')})")
    mt = config.get("max_trajectories", -1)
    LOG.info(f"max_trajectories: {mt} ({'전체' if mt == -1 else str(mt) + '개'})")
    LOG.info("=" * 50)

    skip_keys = ("save_every", "validate_every", "max_updates")
    trainer = OfflineTrainer(**{k: v for k, v in config.items() if k not in skip_keys})
    trainer.train(
        max_updates=config.get("max_updates", 1_000_000),
        save_every=config.get("save_every", 50_000),
        validate_every=config.get("validate_every", 10_000),
    )


def cmd_online(args):
    from config import get_online_config
    from trainers.online_trainer import OnlineTrainer

    overrides = {}
    if args.dataset is not None:
        overrides["dataset_name"] = args.dataset
    if args.max_episodes is not None:
        overrides["max_episodes"] = args.max_episodes
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.offline_checkpoint is not None:
        overrides["offline_checkpoint_path"] = args.offline_checkpoint
    if args.online_checkpoint is not None:
        overrides["online_checkpoint_path"] = args.online_checkpoint
    if args.student_checkpoint_path is not None:
        overrides["student_checkpoint_path"] = args.student_checkpoint_path
    if "device" not in overrides:
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    env = getattr(args, "env", None) or overrides.get("dataset_name")
    config = get_online_config(overrides, env=env)
    dataset_name = config["dataset_name"]
    if config.get("offline_checkpoint_path") is None:
        config["offline_checkpoint_path"] = f"./offline_checkpoints/{dataset_name}/best_offline_checkpoint_{dataset_name}.pth"
    if config.get("online_checkpoint_path") is None:
        config["online_checkpoint_path"] = f"./online_checkpoints/{dataset_name}/best_online_checkpoint_{dataset_name}.pth"

    LOG.info("=" * 50)
    LOG.info(f"설정 파일: config/{env or 'default'}.yaml")
    LOG.info(f"온라인 훈련: {dataset_name} (state/action_dim 자동 감지)")
    LOG.info(f"체크포인트 | 오프라인: {config['offline_checkpoint_path']}")
    LOG.info(f"체크포인트 | 온라인: {config['online_checkpoint_path']}")
    LOG.info("=" * 50)

    trainer = OnlineTrainer(**config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="OPSO: Offline Pretraining with State-Only Imitation")
    sub = parser.add_subparsers(dest="command", required=True)

    # offline
    p_off = sub.add_parser("offline", help="오프라인 잠재 사전학습")
    p_off.add_argument("--dataset", type=str, default=None, help="Dataset name (overrides config)")
    p_off.add_argument("--env", type=str, default=None, help="환경 설정 (config/<env>.yaml)")
    p_off.add_argument("--data", dest="data_source", type=str, default=None, help="ogbench | d4rl")
    p_off.add_argument("--max_trajectories", type=int, default=None)
    p_off.add_argument("--batch_size", type=int, default=None)
    p_off.add_argument("--max_updates", type=int, default=None, help="gradient update 수 (기본 1M).")
    p_off.add_argument("--device", type=str, default=None)
    p_off.add_argument("--data_dir", type=str, default=None)
    p_off.add_argument("--save_dir", type=str, default=None)
    p_off.add_argument("--context_stride", type=int, default=None)
    p_off.add_argument("--seed", type=int, default=None)
    p_off.add_argument("--beta_nce", type=float, default=None, help="InfoNCE weight (0=off). Default from config.")
    p_off.add_argument("--use_focal_loss", action="store_true", help="Use focal loss for reward/success head (class imbalance).")
    p_off.add_argument("--focal_alpha", type=float, default=None)
    p_off.add_argument("--focal_gamma", type=float, default=None)
    p_off.set_defaults(func=cmd_offline)

    # online
    p_on = sub.add_parser("online", help="온라인 제어 학습")
    p_on.add_argument("--dataset", type=str, default=None)
    p_on.add_argument("--env", type=str, default=None)
    p_on.add_argument("--max_episodes", type=int, default=None)
    p_on.add_argument("--batch_size", type=int, default=None)
    p_on.add_argument("--device", type=str, default=None)
    p_on.add_argument("--offline_checkpoint", type=str, default=None)
    p_on.add_argument("--online_checkpoint", type=str, default=None)
    p_on.add_argument("--student_checkpoint_path", type=str, default=None)
    p_on.set_defaults(func=cmd_online)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
