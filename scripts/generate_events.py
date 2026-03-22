"""Generate a large volume of simulated telemetry events and POST them to the
FastAPI /events endpoint.

The simulation logic is identical to notebooks/Event Chains Revisited.ipynb:
  Processing tree: A->B->D, A->C->E->F, A->C->E->G, A->H->J (J is random subset only)
  Latencies drawn from |Normal(mu, sigma)|, throttled by per-process service rates.
  Transaction refs and context match the notebook exactly.

Usage:
    python scripts/generate_events.py                 # defaults: 10 intervals, http://localhost:8000
    python scripts/generate_events.py --intervals 60  # more data
    python scripts/generate_events.py --base-url http://localhost:8000
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Simulation parameters (mirror notebook cell 2)
# ---------------------------------------------------------------------------
# Use current time so data is not expired by ClickHouse TTL (90-day retention)
START_TIME = np.datetime64(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000"))
T_MU = int(np.log(15))
T_SIGMA = int(np.log(50))
T_ALPHA = 75

PROC_PARAMS = {
    "A": {"mu": 0.5, "sigma": 0.5, "alpha": 60, "dep": "T", "refs": ["A"], "context": ["tea"]},
    "B": {"mu": 0.9, "sigma": 0.3, "alpha": 25, "dep": "A", "refs": ["A", "B"]},
    "C": {"mu": 0.9, "sigma": 0.3, "alpha": 25, "dep": "A", "refs": ["C", "A"], "context": ["tea", "coffee"]},
    "D": {"mu": 2.5, "sigma": 1.0, "alpha": 25, "dep": "B", "refs": ["D", "B"]},
    "E": {"mu": 2.5, "sigma": 1.0, "alpha": 12, "dep": "C", "refs": ["E", "C"]},
    "F": {"mu": 0.9, "sigma": 0.2, "alpha": 60, "dep": "E", "refs": ["F", "C", "E"], "context": ["milkshake"]},
    "G": {"mu": 0.5, "sigma": 0.1, "alpha": 7, "dep": "E", "refs": ["G", "E", ["A", "C"]]},
    "H": {"mu": 1.5, "sigma": 0.6, "alpha": 25, "dep": "A", "refs": ["A"]},
    "J": {"mu": 1.0, "sigma": 0.4, "alpha": 25, "dep": "H", "refs": ["J", "A"], "context": ["juice"]},
}


# ---------------------------------------------------------------------------
# Simulation functions (mirror notebook cells 3-8)
# ---------------------------------------------------------------------------

def apply_service_rate(timeseries: np.ndarray, rate_per_second: int) -> np.ndarray:
    """Throttle a timestamp array by a maximum service rate (notebook cell 3)."""
    df = pd.DataFrame(timeseries, columns=["timeseries"]).sort_values("timeseries")
    df["time_deltas"] = df["timeseries"].diff()
    min_delta = np.timedelta64(int(1 / rate_per_second * 1000), "ms")
    df["rate_limited_deltas"] = np.maximum(
        df["time_deltas"], np.full(df["time_deltas"].shape, min_delta)
    )
    df.loc[df["rate_limited_deltas"].isna(), "rate_limited_deltas"] = np.timedelta64(0, "ms")
    df["rate_limited_ts"] = df["rate_limited_deltas"].cumsum() + df.iloc[0]["timeseries"]
    return df.sort_index()["rate_limited_ts"].to_numpy()


def simulate_originating_events(num_intervals: int) -> np.ndarray:
    """Generate originating transaction timestamps T (notebook cell 4)."""
    T = np.ndarray([0], dtype=np.datetime64)
    for i in range(num_intervals):
        count = min(int(stats.lognorm(s=T_SIGMA, scale=np.exp(T_MU)).rvs(1)[0]), T_ALPHA)
        if count > 0:
            t_interval = (
                np.sort(np.random.random(count).round(3) * 1000).astype(np.timedelta64)
                + START_TIME
                + np.timedelta64(i, "s")
            )
            T = np.concatenate((T, t_interval), axis=0)
    return T


def simulate_timestamps(T: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Simulate per-process timestamps with latency + service rate (notebook cell 5).

    Returns (timestamps, j_mask) where j_mask is a boolean array indicating which
    transactions include event J.
    """
    num_obs = T.shape[0]
    timestamps = {"T": T}
    for proc, params in PROC_PARAMS.items():
        latency = (
            np.absolute(stats.norm(loc=params["mu"], scale=params["sigma"]).rvs(num_obs).round(3))
            * 1000
        ).astype(np.timedelta64)
        timestamps[proc] = apply_service_rate(timestamps[params["dep"]], params["alpha"]) + latency
    del timestamps["T"]

    # Only a random subset of transactions generate event J
    j_mask = np.random.random(num_obs) < 0.5
    timestamps["J"][~j_mask] = np.datetime64("NaT")

    return timestamps, j_mask


def simulate_refs(num_obs: int, prefix: str = "") -> dict[str, list[list[dict]]]:
    """Simulate transaction references per event (notebook cell 6)."""
    trans_refs = {}
    for proc, params in PROC_PARAMS.items():
        refs_proc = []
        for i in range(num_obs):
            refs_event = []
            for ref in params["refs"]:
                if isinstance(ref, str):
                    refs_event.append({"type": ref, "id": f"{prefix}{ref}{i}", "ver": 1})
                elif isinstance(ref, list):
                    chosen = np.random.choice(ref)
                    refs_event.append({"type": str(chosen), "id": f"{prefix}{chosen}{i}", "ver": 1})
            refs_proc.append(refs_event)
        trans_refs[proc] = refs_proc
    return trans_refs


def simulate_context(num_obs: int) -> dict[str, list[dict]]:
    """Simulate context payloads per event (notebook cell 7)."""
    context = {}
    for proc, params in PROC_PARAMS.items():
        ctx_proc = []
        for i in range(num_obs):
            ctx_event = {}
            for item in params.get("context", []):
                ctx_event[item] = i
            ctx_proc.append(ctx_event)
        context[proc] = ctx_proc
    return context


def build_event_list(
    timestamps: dict[str, np.ndarray],
    trans_refs: dict[str, list[list[dict]]],
    context: dict[str, list[dict]],
) -> list[dict]:
    """Assemble into a shuffled event list matching notebook cell 8."""
    num_obs = len(timestamps["A"])
    rows = []
    for proc, ts_array in timestamps.items():
        for i, ts in enumerate(ts_array):
            if np.isnat(ts):
                continue
            rows.append({
                "EventName": proc,
                "Timestamp": str(ts),
                "Refs": trans_refs[proc][i],
                "Context": context[proc][i],
            })

    df = pd.DataFrame(rows).sort_values("Timestamp").reset_index()
    df["index"] = df["index"] + np.random.randint(0, num_obs, df.shape[0])
    df = df.set_index("index").sort_index().reset_index(drop=True)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# POST events to the API
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate and POST simulated telemetry events")
    parser.add_argument("--intervals", type=int, default=10,
                        help="Number of 1-second time intervals to simulate (default: 10)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="FastAPI base URL (default: http://localhost:8000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate a run prefix from current timestamp to avoid ID collisions
    run_prefix = datetime.now(timezone.utc).strftime("%y%m%d%H%M%S") + "_"
    print(f"Run prefix: {run_prefix}")

    print(f"Simulating {args.intervals} intervals...")
    T = simulate_originating_events(args.intervals)
    num_obs = T.shape[0]
    print(f"  Originating transactions: {num_obs}")

    timestamps, j_mask = simulate_timestamps(T)
    j_count = int(j_mask.sum())
    print(f"  Transactions with event J: {j_count}/{num_obs}")
    trans_refs = simulate_refs(num_obs, prefix=run_prefix)
    context = simulate_context(num_obs)
    events = build_event_list(timestamps, trans_refs, context)
    print(f"  Total events generated: {len(events)}")

    print(f"\nPOSTing to {args.base_url}/events ...")
    post_events(events, args.base_url)


if __name__ == "__main__":
    main()
