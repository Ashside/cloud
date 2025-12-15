import os
from typing import Callable, TypeVar

T = TypeVar("T")


def _clean_value(value: str) -> str:
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    return value


def load_env(env_path: str | None = None) -> None:
    """
    Lightweight .env loader to avoid adding third-party deps.
    Populates os.environ without overwriting existing keys.
    """
    path = env_path or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                key = key.strip()
                value = _clean_value(raw_value)
                os.environ.setdefault(key, value)
    except OSError:
        # 非关键路径，读取失败时不抛异常
        return


def env_or_default(key: str, default: T, cast: Callable[[str], T] | None = None) -> T:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return cast(val) if cast else val  # type: ignore[return-value]
    except Exception:
        return default


def env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
