import os
import sys
import time
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.data_simulation import DataSimulator

def post_events(events: list[dict], base_url: str) -> None:
    """POST each event to /events and print progress."""
    total = len(events)
    success = 0
    errors = 0
    t0 = time.perf_counter()

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        for i, event in enumerate(events):
            try:
                resp = client.post("/events", json=event)
                resp.raise_for_status()
                success += 1
            except httpx.HTTPError as exc:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR on event {i}: {exc}")
                elif errors == 6:
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
    events = simulator.generate(prefix=run_prefix)
    print(f"  Total events generated: {len(events)}")

    print(f"\nPOSTing to {url}")
    post_events(events, url)

if __name__ == "__main__":
    main()
