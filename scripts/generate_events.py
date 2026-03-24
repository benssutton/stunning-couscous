import os
import sys
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.data_simulation import DataSimulator

BATCH_SIZE = 50


def post_events(events: list[dict], base_url: str) -> None:
    """POST events to /events in batches of BATCH_SIZE."""
    total = len(events)
    success = 0
    errors = 0
    t0 = time.perf_counter()

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for batch_start in range(0, total, BATCH_SIZE):
            batch = events[batch_start : batch_start + BATCH_SIZE]
            i = batch_start + len(batch) - 1  # index of last event in batch
            try:
                resp = client.post("/events", json=batch)
                resp.raise_for_status()
                success += len(batch)
            except httpx.HTTPError as exc:
                errors += len(batch)
                if errors <= 5 * BATCH_SIZE:
                    print(f"  ERROR on batch starting at {batch_start}: {exc}")
                elif errors == 6 * BATCH_SIZE:
                    print("  ... suppressing further error details")

            if (i + 1) % 200 == 0 or i == total - 1:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{total}] {rate:.0f} events/s  ({success} ok, {errors} err)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {success} succeeded, {errors} failed in {elapsed:.1f}s ({success/elapsed:.0f} events/s)")


def main(intervals: int = 10, seed: int = 42, url: str = "http://localhost:8000"):
    run_prefix = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S") + "_"
    print(f"Run prefix: {run_prefix}")

    print(f"Simulating {intervals} intervals...")
    simulator = DataSimulator(num_intervals=intervals, seed=seed)
    _, events = simulator.generate(prefix=run_prefix)
    print(f"  Total events generated: {len(events)}")

    print(f"\nPOSTing to {url}")
    post_events(events, url)

if __name__ == "__main__":
    main()
