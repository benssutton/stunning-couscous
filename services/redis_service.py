from __future__ import annotations

import json
import logging
from io import StringIO
from typing import TYPE_CHECKING

import redis
import ulid
from redis.commands.search.field import TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from services.models import Event, PathProfile

if TYPE_CHECKING:
    from services.chain_classifier import ChainProfilePredictor

logger = logging.getLogger(__name__)


class RedisService:
    INDEX_NAME = "argus:ec:idx"
    KEY_BASE = "argus:ec"
    STREAM_NAME = "argus:ecstream"

    def __init__(self, pool: redis.ConnectionPool):
        self.r = redis.Redis(connection_pool=pool)
        self.expected_events: set[str] = set()
        self.path_profiles: list[PathProfile] = []
        self._predictor: ChainProfilePredictor | None = None

    def _create_key(self) -> str:
        return f"{self.KEY_BASE}:{ulid.ULID()}"

    def ensure_index(self) -> None:
        """Create the RediSearch index if it does not already exist."""
        try:
            self.r.ft(self.INDEX_NAME).info()
        except redis.ResponseError:
            self.r.ft(self.INDEX_NAME).create_index(
                [
                    TagField("$.concatenatedrefs[*]", as_name="concatenatedrefs"),
                    TagField("$.complete", as_name="complete"),
                    TagField("$.terminated", as_name="terminated"),
                ],
                definition=IndexDefinition(
                    index_type=IndexType.JSON,
                    prefix=[f"{self.KEY_BASE}:"],
                ),
            )

    def set_expected_events(self, events: set[str]) -> None:
        """Set the expected event names for completeness checking."""
        self.expected_events = events

    def set_path_profiles(self, profiles: list[PathProfile]) -> None:
        """Set path profiles for profile-aware completeness/termination."""
        self.path_profiles = profiles

    def set_predictor(self, predictor: ChainProfilePredictor) -> None:
        """Set the fitted classifier for runtime profile prediction."""
        self._predictor = predictor

    def add_or_merge_event(self, event: Event) -> str:
        """Process an incoming event into the event chain assembly.

        Searches for existing non-terminated chains matching any of the event's
        refs.  Creates a new chain or merges into existing chain(s) as
        appropriate.  Returns the chain_id that was created or updated.
        """
        concat_refs = [
            f"{ref.type}_{ref.id}_{ref.ver}" for ref in event.Refs
        ]

        # Search for existing non-terminated chains with matching refs
        query = (
            "@concatenatedrefs:{"
            + "|".join(concat_refs)
            + "} @terminated:{false}"
        )
        results = self.r.ft(self.INDEX_NAME).search(query)

        event_timestamp = {event.EventName: event.Timestamp}
        context = event.Context

        if len(results.docs) == 0:
            return self._create_new_chain(concat_refs, event_timestamp, context)

        # Merge with existing chain(s)
        # Warn on multi-chain merge (notebook TODO — deferred)
        if len(results.docs) > 1:
            logger.warning(
                "Event matched %d chains — multi-chain merge not yet implemented, "
                "processing each independently",
                len(results.docs),
            )

        chain_id = None
        p = self.r.pipeline()

        for result in results.docs:
            chain_json = json.load(StringIO(result.json))
            refs_on_chain = chain_json["concatenatedrefs"]
            events_on_chain = set(chain_json["timestamps"].keys())
            chain_id_this = result.id
            refs_not_found = set(concat_refs) - set(refs_on_chain)

            if len(refs_not_found) == 0:
                # All refs already present — just add event timestamp and context
                p.json().merge(chain_id_this, "$.timestamps", event_timestamp)
                p.json().merge(chain_id_this, "$.context", context)

            else:
                # Check for ref type conflicts
                types_on_chain = {r.split("_")[0] for r in refs_on_chain}
                types_not_found = {r.split("_")[0] for r in refs_not_found}
                types_in_conflict = types_on_chain & types_not_found

                if len(types_in_conflict) == 0:
                    # No conflicts — merge refs and event into existing chain
                    p.json().merge(chain_id_this, "$.timestamps", event_timestamp)
                    p.json().merge(chain_id_this, "$.context", context)
                    for ref in refs_not_found:
                        p.json().arrappend(chain_id_this, "$.concatenatedrefs", ref)

                else:
                    # Conflicts — split refs
                    refs_not_in_conflict = [
                        r for r in refs_not_found
                        if r.split("_")[0] not in types_in_conflict
                    ]
                    refs_in_conflict = [
                        r for r in refs_not_found
                        if r.split("_")[0] in types_in_conflict
                    ]
                    refs_minus_conflicts = [
                        r for r in refs_on_chain
                        if r.split("_")[0] not in types_in_conflict
                    ]

                    # Update existing chain with non-conflicting refs
                    p.json().merge(chain_id_this, "$.timestamps", event_timestamp)
                    p.json().merge(chain_id_this, "$.context", context)
                    for r in refs_not_in_conflict:
                        p.json().arrappend(chain_id_this, "$.concatenatedrefs", r)

                    # Create duplicate chain with conflicting refs
                    dup = dict(chain_json)
                    dup["concatenatedrefs"] = refs_minus_conflicts + refs_in_conflict
                    dup["timestamps"] = {**chain_json["timestamps"], **event_timestamp}
                    p.json().set(self._create_key(), "$", dup)

            # Completeness and termination checks
            all_events = events_on_chain | set(event_timestamp.keys())
            self._check_completeness(p, chain_id_this, all_events, chain_json)

            p.xadd(
                self.STREAM_NAME,
                {"ecid": chain_id_this},
                maxlen=1_000_000,
                approximate=True,
            )
            chain_id = chain_id_this

        p.execute()
        return chain_id

    def _check_completeness(
        self,
        pipe: redis.client.Pipeline,
        chain_id: str,
        all_events: set[str],
        chain_json: dict,
    ) -> None:
        """Check termination and completeness, updating flags via pipeline.

        Uses the fitted classifier (if available) to predict the single
        matching profile, then checks whether all terminal nodes are present.
        Falls back to profile iteration or flat expected_events check.
        """
        if self._predictor is not None:
            context_keys = set(chain_json.get("context", {}).keys())
            profile = self._predictor.predict(all_events, context_keys)
            if profile is not None:
                if profile.terminal_nodes and profile.terminal_nodes <= all_events:
                    pipe.delete(chain_id)
            return

        if self.path_profiles:
            # Fallback: iterate profiles when no predictor is loaded
            for profile in self.path_profiles:
                if not all_events <= profile.node_set:
                    continue
                if profile.terminal_nodes and profile.terminal_nodes <= all_events:
                    pipe.delete(chain_id)
                    break
                break
        elif self.expected_events:
            # Legacy flat check
            awaited = self.expected_events - all_events
            if len(awaited) == 0:
                pipe.delete(chain_id)

    def get_all_chain_keys(self) -> list[str]:
        """Return all event chain JSON document keys (excluding index and stream)."""
        raw_keys = self.r.keys(f"{self.KEY_BASE}:*")
        result: list[str] = []
        for k in raw_keys:
            s = k.decode() if isinstance(k, bytes) else str(k)
            if s.endswith(":ecstream") or ":idx" in s:
                continue
            result.append(s)
        return result

    def get_all_chains(self) -> list[dict]:
        """Fetch all event chain JSON documents from Redis."""
        keys = self.get_all_chain_keys()
        if not keys:
            return []
        docs = self.r.json().mget(keys, "$")
        return [doc[0] for doc in docs if doc]

    def delete_all_chains(self) -> int:
        """Delete all event chain keys and the stream. Returns count deleted."""
        keys = self.get_all_chain_keys()
        if not keys:
            return 0
        p = self.r.pipeline()
        for key in keys:
            p.delete(key)
        p.delete(self.STREAM_NAME)
        results = p.execute()
        return sum(1 for r in results if r)

    def load_chains(self, chains: list[dict]) -> int:
        """Bulk-load chain documents into Redis. Returns count loaded."""
        if not chains:
            return 0
        p = self.r.pipeline()
        for chain in chains:
            key = self._create_key()
            p.json().set(key, "$", chain)
        p.execute()
        return len(chains)

    def _create_new_chain(
        self,
        concat_refs: list[str],
        event_timestamp: dict,
        context: dict,
    ) -> str:
        chain = {
            "concatenatedrefs": concat_refs,
            "timestamps": event_timestamp,
            "context": context,
            "complete": False,
            "terminated": False,
        }
        chain_id = self._create_key()
        p = self.r.pipeline()
        p.json().set(chain_id, "$", chain)
        p.xadd(self.STREAM_NAME, {"ecid": chain_id})
        p.execute()
        return chain_id
