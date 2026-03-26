from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from schemas.models import PathProfile

if TYPE_CHECKING:
    from services.redis_service import RedisService

logger = logging.getLogger(__name__)


class CacheService:
    """Bulk cache CRUD operations for event chain documents in Redis."""

    def __init__(self, redis_svc: RedisService):
        self.redis_svc = redis_svc

    async def get_all_chain_keys(self) -> list[str]:
        """Return all event chain JSON document keys (excluding index and stream)."""
        r = self.redis_svc.r
        key_base = self.redis_svc.KEY_BASE
        result: list[str] = []
        async for k in r.scan_iter(match=f"{key_base}:*"):
            s = k.decode() if isinstance(k, bytes) else str(k)
            if s.endswith(":ecstream") or ":idx" in s:
                continue
            result.append(s)
        return result

    async def get_all_chains(self) -> list[dict]:
        """Fetch all event chain JSON documents from Redis."""
        keys = await self.get_all_chain_keys()
        if not keys:
            return []
        docs = await self.redis_svc.r.json().mget(keys, "$")
        return [doc[0] for doc in docs if doc]

    async def delete_all_chains(self) -> int:
        """Delete all event chain keys and the stream. Returns count deleted."""
        keys = await self.get_all_chain_keys()
        if not keys:
            return 0
        r = self.redis_svc.r
        p = r.pipeline()
        for key in keys:
            p.delete(key)
        p.delete(self.redis_svc.STREAM_NAME)
        results = await p.execute()
        return sum(1 for result in results if result)

    async def load_chains(self, chains: list[dict]) -> int:
        """Bulk-load chain documents into Redis. Returns count loaded."""
        if not chains:
            return 0
        r = self.redis_svc.r
        ttl = self.redis_svc.CHAIN_TTL_SECONDS
        p = r.pipeline()
        for chain in chains:
            key = self.redis_svc._create_key()
            p.json().set(key, "$", chain)
            p.expire(key, ttl)
        await p.execute()
        return len(chains)

    async def load_unterminated_chains(self, chains: list[dict]) -> dict:
        """Filter out terminated chains and load the rest into Redis.

        Returns a summary dict with total_chains, terminated, and loaded counts.
        """
        chains_to_load: list[dict] = []
        predictor = self.redis_svc._predictor
        profiles = self.redis_svc.path_profiles

        for chain in chains:
            events = set(chain["timestamps"].keys())
            context_keys = set(chain["context"].keys())

            terminated = False
            if predictor is not None:
                profile = predictor.predict(events, context_keys)
                if profile and profile.terminal_nodes and profile.terminal_nodes <= events:
                    terminated = True
            elif profiles:
                for profile in profiles:
                    if events <= profile.node_set:
                        if profile.terminal_nodes and profile.terminal_nodes <= events:
                            terminated = True
                        break

            if not terminated:
                chains_to_load.append(chain)

        loaded = await self.load_chains(chains_to_load)
        return {
            "total_chains": len(chains),
            "terminated": len(chains) - len(chains_to_load),
            "loaded": loaded,
        }
