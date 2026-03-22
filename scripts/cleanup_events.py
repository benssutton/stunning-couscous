"""Clean up all event data from ClickHouse and Redis.

Usage:
    python scripts/cleanup_events.py              # clean both ClickHouse + Redis
    python scripts/cleanup_events.py --clickhouse # clean ClickHouse only
    python scripts/cleanup_events.py --redis      # clean Redis only
"""

import argparse
import sys

import clickhouse_connect
import redis


def cleanup_clickhouse(host: str = "localhost", port: int = 8123, database: str = "argus"):
    """Truncate the events table and report row count before/after."""
    try:
        client = clickhouse_connect.get_client(host=host, port=port)
    except Exception as exc:
        print(f"ClickHouse: could not connect to {host}:{port} — {exc}")
        return False

    # Check if database/table exist
    databases = [row[0] for row in client.query("SHOW DATABASES").result_rows]
    if database not in databases:
        print(f"ClickHouse: database '{database}' does not exist — nothing to clean")
        client.close()
        return True

    tables = [row[0] for row in client.query(f"SHOW TABLES FROM {database}").result_rows]
    if "events" not in tables:
        print(f"ClickHouse: {database}.events does not exist — nothing to clean")
        client.close()
        return True

    row_count = client.query(f"SELECT count() FROM {database}.events").result_rows[0][0]
    print(f"ClickHouse: {database}.events has {row_count:,} rows")

    client.command(f"TRUNCATE TABLE {database}.events")
    print(f"ClickHouse: {database}.events truncated")
    client.close()
    return True


def cleanup_redis(host: str = "localhost", port: int = 6379):
    """Drop the search index and delete all event chain keys + the stream."""
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping()
    except redis.ConnectionError as exc:
        print(f"Redis: could not connect to {host}:{port} — {exc}")
        return False

    # Count event chain keys
    chain_keys = r.keys("argus:ec:*")
    # Exclude the stream from the key count
    chain_keys = [k for k in chain_keys if k != "argus:ecstream"]
    print(f"Redis: {len(chain_keys)} event chain keys found")

    # Drop the search index
    try:
        r.ft("argus:ec:idx").dropindex(delete_documents=True)
        print("Redis: dropped index argus:ec:idx (and associated documents)")
    except redis.ResponseError:
        print("Redis: index argus:ec:idx does not exist — skipping")

    # Delete the stream
    deleted = r.delete("argus:ecstream")
    if deleted:
        print("Redis: deleted stream argus:ecstream")
    else:
        print("Redis: stream argus:ecstream does not exist — skipping")

    r.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Clean up event data from ClickHouse and Redis")
    parser.add_argument("--clickhouse", action="store_true", help="Clean ClickHouse only")
    parser.add_argument("--redis", action="store_true", help="Clean Redis only")
    parser.add_argument("--ch-host", default="localhost", help="ClickHouse host")
    parser.add_argument("--ch-port", type=int, default=8123, help="ClickHouse port")
    parser.add_argument("--ch-database", default="argus", help="ClickHouse database")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    args = parser.parse_args()

    # If neither flag is set, clean both
    clean_ch = args.clickhouse or (not args.clickhouse and not args.redis)
    clean_redis = args.redis or (not args.clickhouse and not args.redis)

    ok = True
    if clean_ch:
        ok = cleanup_clickhouse(args.ch_host, args.ch_port, args.ch_database) and ok
    if clean_redis:
        ok = cleanup_redis(args.redis_host, args.redis_port) and ok

    if ok:
        print("\nCleanup complete.")
    else:
        print("\nCleanup finished with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
