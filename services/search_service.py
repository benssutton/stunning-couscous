from services.clickhouse_service import ClickHouseService
from services.redis_service import RedisService


class SearchService:
    """Search for event chains by reference ID with support for prefix matching."""

    def __init__(self, ch_svc: ClickHouseService, redis_svc: RedisService):
        self.ch_svc = ch_svc
        self.redis_svc = redis_svc

    def autocomplete_ref_ids(self, prefix: str, limit: int = 20) -> list[str]:
        """Return distinct ref IDs matching a prefix."""
        return self.ch_svc.search_ref_ids(prefix, limit)

    def search_chains_by_ref(self, ref_id: str, limit: int = 100) -> list[str]:
        """Return chain IDs containing the given ref ID."""
        return self.ch_svc.search_chains_by_ref_id(ref_id, limit)

    def search_chains_by_ref_prefix(self, prefix: str, limit: int = 100) -> list[str]:
        """Return chain IDs containing refs whose ID matches a prefix."""
        return self.ch_svc.search_chains_by_ref_prefix(prefix, limit)
