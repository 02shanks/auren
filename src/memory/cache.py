"""Session working-memory cache.

In-process by default; Redis is optional and config-gated. An in-memory-only store
is never treated as durable state — it caches tool outputs *within* a session only
(the durable copy lives in ``MemoryStore``). If Redis is requested but unreachable,
we fall back to in-process rather than failing the session (DECISIONS D-08).
"""

import json
import time
from typing import Any, Protocol

from src.utils.config import env
from src.utils.logging_config import get_logger

log = get_logger("cache")


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def clear(self) -> None: ...


class InProcessCache:
    def __init__(self, ttl: int | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._expiry: dict[str, float] = {}
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        exp = self._expiry.get(key)
        if exp is not None and time.time() > exp:
            self.delete(key)
            return None
        return self._data.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._data[key] = value
        ttl = ttl if ttl is not None else self._ttl
        if ttl:
            self._expiry[key] = time.time() + ttl

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._expiry.pop(key, None)

    def clear(self) -> None:
        self._data.clear()
        self._expiry.clear()


class RedisSessionCache:
    def __init__(self, url: str, ttl: int | None = 3600) -> None:
        import redis  # lazy import; only on the --extra cache path

        self._r = redis.Redis.from_url(url)
        self._r.ping()
        self._ttl = ttl

    def get(self, key: str) -> Any | None:
        raw = self._r.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        self._r.set(key, json.dumps(value), ex=ttl if ttl is not None else self._ttl)

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def clear(self) -> None:
        self._r.flushdb()


def get_cache(config: dict) -> Cache:
    backend = config.get("memory", {}).get("session_backend", "in_process")
    if backend == "redis":
        url = env(config.get("memory", {}).get("redis_url_env", "REDIS_URL"))
        if url:
            try:
                cache = RedisSessionCache(url)
                log.info("using Redis session cache")
                return cache
            except Exception as exc:
                log.warning("Redis unavailable (%s); using in-process cache", exc)
        else:
            log.warning("session_backend=redis but no REDIS_URL set; using in-process cache")
    return InProcessCache()
