"""Load config.yaml and resolve repo-relative paths."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a repo-relative path from config, e.g. resolve(cfg, 'paths.raw_dir')."""
    node = cfg
    for part in key.split("."):
        node = node[part]
    return Path(cfg["_repo_root"]) / node


def ensure_dirs(cfg: dict) -> None:
    for k in ("paths.raw_dir", "paths.interim_dir", "paths.processed_dir", "paths.output_dir"):
        resolve(cfg, k).mkdir(parents=True, exist_ok=True)
