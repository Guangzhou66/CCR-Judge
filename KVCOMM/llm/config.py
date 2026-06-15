from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict
import os


@dataclass(frozen=True)
class KVCommConfig:
    """
    Configuration for KV communication and scheduling.

    Args:
        threshold (float): The threshold for the KV communication.
        thread_pool_workers (int): The number of threads to use for the thread pool.
        worker_timeout (float): The timeout for the worker.
        group_judge_agent_anchors (bool): Whether judge agents should share
            anchor pools across different candidate agents (treating them
            as an equivalence class).
        group_judge_agent_anchors_by_position (bool): Whether judge agents
            should pool anchors by display position (slot) rather than by
            underlying agent id. This is mainly meaningful when candidate
            answers are shuffled.
    """
    threshold: float = 0.3
    max_anchor_num: int = 20
    window_size: int = 5
    thread_pool_workers: int = 8
    worker_timeout: float = 30.0
    group_judge_agent_anchors: bool = False
    group_judge_agent_anchors_by_position: bool = False
    reuse_strategy: str = "auto"

    @classmethod
    def from_env(cls) -> "KVCommConfig":
        """Create a config from environment variables with safe defaults."""
        return cls(
            threshold=float(os.environ.get("THRESHOLD", cls.threshold)),
            max_anchor_num=int(os.environ.get("MAX_ANCHOR_NUM", cls.max_anchor_num)),
            window_size=int(os.environ.get("WINDOW_SIZE", cls.window_size)),
            thread_pool_workers=int(os.environ.get("KVCOMM_THREAD_WORKERS", cls.thread_pool_workers)),
            worker_timeout=float(os.environ.get("KVCOMM_WORKER_TIMEOUT", cls.worker_timeout)),
            group_judge_agent_anchors=os.environ.get("KVCOMM_GROUP_JUDGE_ANCHORS", "0").lower()
            in {"1", "true", "yes", "y"},
            group_judge_agent_anchors_by_position=os.environ.get(
                "KVCOMM_GROUP_JUDGE_ANCHORS_BY_POSITION", "0"
            ).lower()
            in {"1", "true", "yes", "y"},
            reuse_strategy=os.environ.get("KVCOMM_REUSE_STRATEGY", cls.reuse_strategy),
        ).validate()

    def apply_overrides(self, **overrides: Any) -> "KVCommConfig":
        """Return a copy with provided non-None fields overridden."""
        current: Dict[str, Any] = asdict(self)
        for key, value in overrides.items():
            if value is None or key not in current:
                continue
            current[key] = value
        return replace(self, **current).validate()

    def validate(self) -> "KVCommConfig":
        """Validate value ranges and return self."""
        if self.thread_pool_workers <= 0:
            raise ValueError("thread_pool_workers must be positive")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        if self.reuse_strategy not in {"auto", "naive", "kvcomm"}:
            raise ValueError("reuse_strategy must be one of {'auto', 'naive', 'kvcomm'}")
        return self
