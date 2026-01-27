"""
Cache manager for response caching and memoization.
Supports multiple backends and eviction strategies.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import json
import asyncio
from pathlib import Path
from collections import OrderedDict
import pickle


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """A cached entry."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def update_access(self) -> None:
        """Update access statistics."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheBackend(ABC):
    """Abstract base for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def keys(self) -> List[str]:
        pass


class MemoryBackend(CacheBackend):
    """In-memory cache backend."""

    def __init__(self, max_size: int = 1000):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size

    async def get(self, key: str) -> Optional[CacheEntry]:
        entry = self.cache.get(key)
        if entry:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.update_access()
        return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = entry

        # Evict if over size
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    async def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    async def clear(self) -> None:
        self.cache.clear()

    async def keys(self) -> List[str]:
        return list(self.cache.keys())


class DiskBackend(CacheBackend):
    """Disk-based cache backend."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, str] = {}  # key -> filename
        self._load_index()

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"

    def _load_index(self) -> None:
        """Load index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self.index = json.load(f)
            except:
                self.index = {}

    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self.cache_dir / "index.json"
        with open(index_path, 'w') as f:
            json.dump(self.index, f)

    async def get(self, key: str) -> Optional[CacheEntry]:
        if key not in self.index:
            return None

        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
            entry.update_access()
            return entry
        except:
            return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        file_path = self._get_file_path(key)

        with open(file_path, 'wb') as f:
            pickle.dump(entry, f)

        self.index[key] = str(file_path)
        self._save_index()

    async def delete(self, key: str) -> bool:
        if key not in self.index:
            return False

        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()

        del self.index[key]
        self._save_index()
        return True

    async def clear(self) -> None:
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink()
        self.index.clear()
        self._save_index()

    async def keys(self) -> List[str]:
        return list(self.index.keys())


class CacheManager:
    """
    Main cache manager with multiple backends and eviction policies.
    """

    def __init__(self, backend: Optional[CacheBackend] = None,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 max_size: int = 1000,
                 default_ttl: Optional[int] = None):
        self.backend = backend or MemoryBackend(max_size)
        self.eviction_policy = eviction_policy
        self.max_size = max_size
        self.default_ttl = default_ttl  # seconds
        self.stats = CacheStats()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        entry = await self.backend.get(key)

        if entry is None:
            self.stats.misses += 1
            return None

        if entry.is_expired():
            await self.backend.delete(key)
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 metadata: Optional[Dict] = None) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

        # Estimate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 0

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            size_bytes=size_bytes,
            metadata=metadata or {}
        )

        await self.backend.set(key, entry)
        self.stats.total_size_bytes += size_bytes
        self.stats.entry_count += 1

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        entry = await self.backend.get(key)
        if entry:
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1

        return await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear all cache entries."""
        await self.backend.clear()
        self.stats = CacheStats()

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        entry = await self.backend.get(key)
        if entry and not entry.is_expired():
            return True
        return False

    async def get_or_set(self, key: str, factory: Callable[[], Any],
                        ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and store."""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    async def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally filtered by pattern."""
        all_keys = await self.backend.keys()

        if pattern:
            import fnmatch
            return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

        return all_keys

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        removed = 0
        keys = await self.backend.keys()

        for key in keys:
            entry = await self.backend.get(key)
            if entry and entry.is_expired():
                await self.backend.delete(key)
                removed += 1
                self.stats.evictions += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "evictions": self.stats.evictions,
            "total_size_bytes": self.stats.total_size_bytes,
            "entry_count": self.stats.entry_count,
            "eviction_policy": self.eviction_policy.value
        }

    def start_cleanup_task(self, interval: int = 60) -> None:
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval)
                await self.cleanup_expired()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        cache = CacheManager(default_ttl=ttl)

        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            # Check cache
            result = await cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await cache.set(key, result)
            return result

        return wrapper
    return decorator


class ResponseCache(CacheManager):
    """
    Specialized cache for LLM responses.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.semantic_dedup = True

    def _generate_key(self, prompt: str, model: str, **params) -> str:
        """Generate cache key from prompt and parameters."""
        key_data = {
            "prompt": prompt,
            "model": model,
            **params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get_response(self, prompt: str, model: str, **params) -> Optional[str]:
        """Get cached response for a prompt."""
        key = self._generate_key(prompt, model, **params)
        return await self.get(key)

    async def cache_response(self, prompt: str, model: str, response: str,
                            ttl: Optional[int] = None, **params) -> None:
        """Cache a response."""
        key = self._generate_key(prompt, model, **params)
        metadata = {
            "prompt_preview": prompt[:100],
            "model": model,
            "response_length": len(response)
        }
        await self.set(key, response, ttl, metadata)
