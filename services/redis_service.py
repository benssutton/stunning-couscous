from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
import ulid
from redis.commands.search.field import TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from schemas.models import Event, PathProfile

if TYPE_CHECKING:
    from services.chain_classifier_service import ChainClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lua script: atomic FT.SEARCH + chain create/merge in one round-trip.
#
# Returns a 4-element array:
#   [status, chain_id, all_events_json, chain_json]
#
# status values:
#   "CREATED"     – new chain was created (no prior match)
#   "MERGED"      – event merged into an existing chain (common path)
#   "CONFLICT"    – ref-type conflict detected; timestamps merged but
#                   Python must handle the duplicate-chain creation
#   "MULTI_MATCH" – more than one chain matched; Python must handle
# ---------------------------------------------------------------------------
_LUA_CHAIN_MERGE = """
local new_key  = KEYS[1]
local idx      = ARGV[1]
local query    = ARGV[2]
local ts_json  = ARGV[3]
local ctx_json = ARGV[4]
local stream   = ARGV[5]
local ttl      = tonumber(ARGV[6])
local maxlen   = ARGV[7]

local refs = {}
for i = 8, #ARGV do
    refs[#refs + 1] = ARGV[i]
end

-- 1. Search --
local raw = redis.call('FT.SEARCH', idx, query, 'LIMIT', '0', '10')
local total = tonumber(raw[1])

if total == 0 then
    local chain = cjson.encode({
        concatenatedrefs = refs,
        timestamps       = cjson.decode(ts_json),
        context          = cjson.decode(ctx_json),
        complete         = false,
        terminated       = false,
    })
    redis.call('JSON.SET', new_key, '$', chain)
    redis.call('EXPIRE', new_key, ttl)
    redis.call('XADD', stream, 'MAXLEN', '~', maxlen, '*', 'ecid', new_key)
    local evts = {}
    for k in pairs(cjson.decode(ts_json)) do evts[#evts + 1] = k end
    return {'CREATED', new_key, cjson.encode(evts), '{}'}
end

if total > 1 then
    return {'MULTI_MATCH', tostring(total), '', ''}
end

-- 2. Single match -> merge --
local chain_key = raw[2]
local fields = raw[3]
local doc_str = nil
for i = 1, #fields, 2 do
    if fields[i] == '$' then
        doc_str = fields[i + 1]
        break
    end
end
if not doc_str then
    doc_str = redis.call('JSON.GET', chain_key, '$')
end

local chain
if type(doc_str) == 'string' and string.sub(doc_str, 1, 1) == '[' then
    chain = cjson.decode(doc_str)[1]
else
    chain = cjson.decode(doc_str)
end

local refs_on = chain.concatenatedrefs or {}
local evt_ts  = cjson.decode(ts_json)

local on_set = {}
for _, r in ipairs(refs_on) do on_set[r] = true end
local not_found = {}
for _, r in ipairs(refs) do
    if not on_set[r] then not_found[#not_found + 1] = r end
end

if #not_found == 0 then
    redis.call('JSON.MERGE', chain_key, '$.timestamps', ts_json)
    redis.call('JSON.MERGE', chain_key, '$.context', ctx_json)
else
    local types_on = {}
    for _, r in ipairs(refs_on) do
        types_on[r:match('^([^_]+)')] = true
    end
    local conflict = false
    for _, r in ipairs(not_found) do
        if types_on[r:match('^([^_]+)')] then
            conflict = true
            break
        end
    end

    if conflict then
        redis.call('JSON.MERGE', chain_key, '$.timestamps', ts_json)
        redis.call('JSON.MERGE', chain_key, '$.context', ctx_json)
        redis.call('EXPIRE', chain_key, ttl)
        local evts = {}
        for k in pairs(chain.timestamps or {}) do evts[#evts + 1] = k end
        for k in pairs(evt_ts) do evts[#evts + 1] = k end
        return {'CONFLICT', chain_key, cjson.encode(evts), cjson.encode(chain)}
    end

    redis.call('JSON.MERGE', chain_key, '$.timestamps', ts_json)
    redis.call('JSON.MERGE', chain_key, '$.context', ctx_json)
    for _, ref in ipairs(not_found) do
        redis.call('JSON.ARRAPPEND', chain_key,
                    '$.concatenatedrefs', '"' .. ref .. '"')
    end
end

redis.call('EXPIRE', chain_key, ttl)
redis.call('XADD', stream, 'MAXLEN', '~', maxlen, '*', 'ecid', chain_key)

local all_evts = {}
for k in pairs(chain.timestamps or {}) do all_evts[#all_evts + 1] = k end
for k in pairs(evt_ts) do all_evts[#all_evts + 1] = k end
return {'MERGED', chain_key, cjson.encode(all_evts), cjson.encode(chain)}
"""


class RedisService:
    INDEX_NAME = "arestor:ec:idx"
    KEY_BASE = "arestor:ec"
    STREAM_NAME = "arestor:ecstream"
    CHAIN_TTL_SECONDS = 600  # 10-minute TTL for stale chain protection
    TERMINATED_TTL_SECONDS = 30  # Grace window for late-arriving events after termination

    def __init__(self, pool: aioredis.ConnectionPool):
        self.r = aioredis.Redis(connection_pool=pool)
        self.expected_events: set[str] = set()
        self.path_profiles: list[PathProfile] = []
        self._terminal_event_names: set[str] = set()
        self._predictor: ChainClassifier | None = None
        self._merge_script = self.r.register_script(_LUA_CHAIN_MERGE)

    def _create_key(self) -> str:
        return f"{self.KEY_BASE}:{ulid.new()}"

    async def ensure_index(self) -> None:
        """Create the RediSearch index if it does not already exist."""
        try:
            await self.r.ft(self.INDEX_NAME).info()
        except aioredis.ResponseError:
            await self.r.ft(self.INDEX_NAME).create_index(
                [
                    TagField(
                        "$.concatenatedrefs[*]",
                        as_name="concatenatedrefs",
                    ),
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
        self._terminal_event_names = {
            node
            for p in profiles
            for node in (p.terminal_nodes or set())
        }

    def set_predictor(self, predictor: ChainClassifier) -> None:
        """Set the fitted classifier for runtime profile prediction."""
        self._predictor = predictor

    async def add_or_merge_event(self, event: Event) -> str:
        """Atomically search-and-merge an event into the chain assembly.

        Uses a Lua script for the common cases (create / single-match merge)
        to guarantee atomicity in a single Redis round-trip.  Falls back to
        a Python pipeline for multi-match and ref-conflict edge cases.
        """
        concat_refs = [
            f"{ref.type}_{ref.id}_{ref.ver}" for ref in event.Refs
        ]
        query = (
            "@concatenatedrefs:{"
            + "|".join(concat_refs)
            + "} @terminated:{false}"
        )
        event_timestamp = {event.EventName: event.Timestamp}
        context = event.Context

        new_key = self._create_key()
        result = await self._merge_script(
            keys=[new_key],
            args=[
                self.INDEX_NAME,
                query,
                json.dumps(event_timestamp),
                json.dumps(context),
                self.STREAM_NAME,
                str(self.CHAIN_TTL_SECONDS),
                "1000000",
                *concat_refs,
            ],
        )

        status = result[0]
        if isinstance(status, bytes):
            status = status.decode()
        chain_id = result[1]
        if isinstance(chain_id, bytes):
            chain_id = chain_id.decode()
        events_json = result[2]
        if isinstance(events_json, bytes):
            events_json = events_json.decode()
        chain_json_str = result[3]
        if isinstance(chain_json_str, bytes):
            chain_json_str = chain_json_str.decode()

        if status == "MULTI_MATCH":
            return await self._fallback_pipeline(
                concat_refs, event_timestamp, context
            )

        if status == "CONFLICT":
            # Timestamps already merged by Lua; handle duplicate chain
            chain_json = json.loads(chain_json_str)
            await self._handle_conflict(
                chain_id, chain_json, concat_refs, event_timestamp
            )

        # Termination check — skip entirely if this event is not a terminal node
        if self._terminal_event_names and event.EventName not in self._terminal_event_names:
            return chain_id

        all_events = set(json.loads(events_json)) if events_json else set()
        if all_events:
            chain_ctx_keys: set[str] = set()
            if chain_json_str and chain_json_str != '{}':
                chain_data = json.loads(chain_json_str)
                chain_ctx_keys = set(chain_data.get("context", {}).keys())
            all_ctx_keys = chain_ctx_keys | set(event.Context.keys())
            if self._should_terminate(event.EventName, all_events, all_ctx_keys):
                await self.r.expire(chain_id, self.TERMINATED_TTL_SECONDS)

        return chain_id

    def _should_terminate(
        self,
        current_event_name: str,
        all_events: set[str],
        context_keys: set[str],
    ) -> bool:
        """Check whether a chain should be terminated after receiving an event.

        1. Return early if the current event is not a terminal node in any profile.
        2. Build non-terminal features (events minus terminals + context keys).
        3. Predict the chain's path profile using the fitted model.
        4. Terminate if the chain contains all terminal nodes of the predicted profile.
        """
        if self._predictor is None:
            if self.expected_events:
                return not (self.expected_events - all_events)
            return False

        profiles = list(self._predictor.profiles.values())

        # Skip prediction if current event is not a terminal in any profile
        if not any(
            current_event_name in p.terminal_nodes
            for p in profiles
            if p.terminal_nodes
        ):
            return False

        # Build feature inputs excluding all terminal nodes
        all_terminals = frozenset().union(
            *(p.terminal_nodes for p in profiles if p.terminal_nodes)
        )
        non_terminal_events = all_events - all_terminals

        # Predict profile from non-terminal features
        profile = self._predictor.predict(non_terminal_events, context_keys)

        if profile is None or not profile.terminal_nodes:
            return False

        return profile.terminal_nodes <= all_events

    async def _handle_conflict(
        self,
        chain_id: str,
        chain_json: dict,
        concat_refs: list[str],
        event_timestamp: dict,
    ) -> None:
        """Create the duplicate chain for ref-type conflicts."""
        refs_on_chain = chain_json["concatenatedrefs"]
        refs_not_found = set(concat_refs) - set(refs_on_chain)
        types_on_chain = {r.split("_")[0] for r in refs_on_chain}
        types_not_found = {r.split("_")[0] for r in refs_not_found}
        types_in_conflict = types_on_chain & types_not_found

        refs_in_conflict = [
            r for r in refs_not_found
            if r.split("_")[0] in types_in_conflict
        ]
        refs_minus_conflicts = [
            r for r in refs_on_chain
            if r.split("_")[0] not in types_in_conflict
        ]
        refs_not_in_conflict = [
            r for r in refs_not_found
            if r.split("_")[0] not in types_in_conflict
        ]

        p = self.r.pipeline()

        # Append non-conflicting new refs to existing chain
        for r in refs_not_in_conflict:
            p.json().arrappend(chain_id, "$.concatenatedrefs", r)

        # Create duplicate chain with conflicting refs
        dup = dict(chain_json)
        dup["concatenatedrefs"] = refs_minus_conflicts + refs_in_conflict
        dup["timestamps"] = {
            **chain_json["timestamps"],
            **event_timestamp,
        }
        dup_key = self._create_key()
        p.json().set(dup_key, "$", dup)
        p.expire(dup_key, self.CHAIN_TTL_SECONDS)

        await p.execute()

    async def _fallback_pipeline(
        self,
        concat_refs: list[str],
        event_timestamp: dict,
        context: dict,
    ) -> str:
        """Pipeline-based merge for multi-chain matches (rare)."""
        query = (
            "@concatenatedrefs:{"
            + "|".join(concat_refs)
            + "} @terminated:{false}"
        )
        results = await self.r.ft(self.INDEX_NAME).search(query)

        if len(results.docs) == 0:
            return await self._create_new_chain(
                concat_refs, event_timestamp, context
            )

        logger.warning(
            "Fallback pipeline: event matched %d chains",
            len(results.docs),
        )

        chain_id = None
        p = self.r.pipeline()

        for result in results.docs:
            chain_json = json.loads(result.json)
            refs_on_chain = chain_json["concatenatedrefs"]
            events_on_chain = set(chain_json["timestamps"].keys())
            chain_id_this = result.id
            refs_not_found = set(concat_refs) - set(refs_on_chain)

            if len(refs_not_found) == 0:
                p.json().merge(
                    chain_id_this, "$.timestamps", event_timestamp
                )
                p.json().merge(chain_id_this, "$.context", context)
            else:
                types_on_chain = {r.split("_")[0] for r in refs_on_chain}
                types_not_found = {r.split("_")[0] for r in refs_not_found}
                types_in_conflict = types_on_chain & types_not_found

                if len(types_in_conflict) == 0:
                    p.json().merge(
                        chain_id_this, "$.timestamps", event_timestamp
                    )
                    p.json().merge(chain_id_this, "$.context", context)
                    for ref in refs_not_found:
                        p.json().arrappend(
                            chain_id_this, "$.concatenatedrefs", ref
                        )
                else:
                    refs_not_in_conflict = [
                        r
                        for r in refs_not_found
                        if r.split("_")[0] not in types_in_conflict
                    ]
                    refs_in_conflict = [
                        r
                        for r in refs_not_found
                        if r.split("_")[0] in types_in_conflict
                    ]
                    refs_minus_conflicts = [
                        r
                        for r in refs_on_chain
                        if r.split("_")[0] not in types_in_conflict
                    ]
                    p.json().merge(
                        chain_id_this, "$.timestamps", event_timestamp
                    )
                    p.json().merge(chain_id_this, "$.context", context)
                    for r in refs_not_in_conflict:
                        p.json().arrappend(
                            chain_id_this, "$.concatenatedrefs", r
                        )
                    dup = dict(chain_json)
                    dup["concatenatedrefs"] = (
                        refs_minus_conflicts + refs_in_conflict
                    )
                    dup["timestamps"] = {
                        **chain_json["timestamps"],
                        **event_timestamp,
                    }
                    p.json().set(self._create_key(), "$", dup)

            p.expire(chain_id_this, self.CHAIN_TTL_SECONDS)

            current_event_name = next(iter(event_timestamp))
            all_events = events_on_chain | set(event_timestamp.keys())
            chain_ctx_keys = set(chain_json.get("context", {}).keys())
            all_ctx_keys = chain_ctx_keys | set(context.keys())
            if self._should_terminate(current_event_name, all_events, all_ctx_keys):
                p.expire(chain_id_this, self.TERMINATED_TTL_SECONDS)

            p.xadd(
                self.STREAM_NAME,
                {"ecid": chain_id_this},
                maxlen=1_000_000,
                approximate=True,
            )
            chain_id = chain_id_this

        await p.execute()
        return chain_id

    async def _create_new_chain(
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
        p.expire(chain_id, self.CHAIN_TTL_SECONDS)
        p.xadd(self.STREAM_NAME, {"ecid": chain_id})
        await p.execute()
        return chain_id
