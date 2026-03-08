"""
OPSO 설정 모듈. YAML 기반.
- config/default.yaml 또는 config/<env>.yaml (env 지정 시)
"""
import os

try:
    import yaml
except ImportError:
    yaml = None

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YAML = os.path.join(_CONFIG_DIR, "default.yaml")

_cache = {}  # path -> raw dict


def _load_yaml(env=None):
    if env is None or env == "default":
        path = _DEFAULT_YAML
    else:
        path = os.path.join(_CONFIG_DIR, f"{env}.yaml")
    if path not in _cache:
        if yaml is None:
            raise ImportError("PyYAML required. pip install pyyaml")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _cache[path] = yaml.safe_load(f)
    return _cache[path]


def get_offline_config(overrides=None, env=None):
    """오프라인 훈련용 설정. default 먼저 로드 후 config/<env>.yaml로 override (env 있으면)."""
    base = _load_yaml(None)  # default
    if env and env != "default" and os.path.isfile(os.path.join(_CONFIG_DIR, f"{env}.yaml")):
        raw = _load_yaml(env)
        common = {**base.get("common", {}), **raw.get("common", {})}
        offline = {**base.get("offline", {}), **raw.get("offline", {})}
        offline_train = {**base.get("offline_train", {}), **raw.get("offline_train", {})}
        cfg = {**common, **offline, **offline_train}
    else:
        cfg = {**base.get("common", {}), **base.get("offline", {}), **base.get("offline_train", {})}
    if overrides:
        cfg = {**cfg, **overrides}
    return cfg


def get_online_config(overrides=None, env=None):
    """온라인 훈련용 설정. default 먼저 로드 후 config/<env>.yaml로 override (env 있으면)."""
    base = _load_yaml(None)
    if env and env != "default" and os.path.isfile(os.path.join(_CONFIG_DIR, f"{env}.yaml")):
        raw = _load_yaml(env)
        common = {**base.get("common", {}), **raw.get("common", {})}
        online = {**base.get("online", {}), **raw.get("online", {})}
        cfg = {**common, **online}
    else:
        cfg = {**base.get("common", {}), **base.get("online", {})}
    if overrides:
        cfg = {**cfg, **overrides}
    return cfg


def get_config(mode="offline", overrides=None, env=None):
    """mode: 'offline' | 'online'. env: config/<env>.yaml 사용."""
    if mode == "offline":
        return get_offline_config(overrides, env)
    if mode == "online":
        return get_online_config(overrides, env)
    raise ValueError(f"mode must be 'offline' or 'online', got {mode}")


__all__ = [
    "get_config",
    "get_offline_config",
    "get_online_config",
]
