from typing import Tuple

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

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


class DataSimulator:
    """Generates simulated telemetry events."""

    def __init__(self, num_intervals: int = 1, seed: int | None = None) -> None:
        self.num_intervals = num_intervals
        self._start_time = np.datetime64(
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")
        )
        if seed is not None:
            np.random.seed(seed)

    def generate(self, prefix: str = "") -> Tuple[int, list[dict]]:
        """Return simulated events as a list of JSON-serialisable dicts."""
        T = self._simulate_originating_events()
        num_obs = T.shape[0]
        timestamps, _ = self._simulate_timestamps(T)
        trans_refs = self._simulate_refs(num_obs, prefix=prefix)
        context = self._simulate_context(num_obs)
        return len(T), self._build_event_list(timestamps, trans_refs, context)

    # ------------------------------------------------------------------
    # Private simulation steps
    # ------------------------------------------------------------------

    def _simulate_originating_events(self) -> np.ndarray:
        T = np.ndarray([0], dtype=np.datetime64)
        for i in range(self.num_intervals):
            count = min(
                int(stats.lognorm(s=T_SIGMA, scale=np.exp(T_MU)).rvs(1)[0]),
                T_ALPHA,
            )
            if count > 0:
                t_interval = (
                    np.sort(np.random.random(count).round(3) * 1000).astype(np.timedelta64)
                    + self._start_time
                    + np.timedelta64(i, "s")
                )
                T = np.concatenate((T, t_interval), axis=0)
        return T

    def _simulate_timestamps(
        self, T: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        num_obs = T.shape[0]
        timestamps: dict[str, np.ndarray] = {"T": T}
        for proc, params in PROC_PARAMS.items():
            latency = (
                np.absolute(
                    stats.norm(loc=params["mu"], scale=params["sigma"]).rvs(num_obs).round(3)
                )
                * 1000
            ).astype(np.timedelta64)
            timestamps[proc] = (
                self._apply_service_rate(timestamps[params["dep"]], params["alpha"]) + latency
            )
        del timestamps["T"]
        j_mask = np.random.random(num_obs) < 0.5
        timestamps["J"][~j_mask] = np.datetime64("NaT")
        return timestamps, j_mask

    @staticmethod
    def _apply_service_rate(timeseries: np.ndarray, rate_per_second: int) -> np.ndarray:
        df = pd.DataFrame(timeseries, columns=["timeseries"]).sort_values("timeseries")
        df["time_deltas"] = df["timeseries"].diff()
        min_delta = np.timedelta64(int(1 / rate_per_second * 1000), "ms")
        df["rate_limited_deltas"] = np.maximum(
            df["time_deltas"], np.full(df["time_deltas"].shape, min_delta)
        )
        df.loc[df["rate_limited_deltas"].isna(), "rate_limited_deltas"] = np.timedelta64(0, "ms")
        df["rate_limited_ts"] = df["rate_limited_deltas"].cumsum() + df.iloc[0]["timeseries"]
        return df.sort_index()["rate_limited_ts"].to_numpy()

    @staticmethod
    def _simulate_refs(num_obs: int, prefix: str = "") -> dict[str, list[list[dict]]]:
        trans_refs: dict[str, list[list[dict]]] = {}
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

    @staticmethod
    def _simulate_context(num_obs: int) -> dict[str, list[dict]]:
        context: dict[str, list[dict]] = {}
        for proc, params in PROC_PARAMS.items():
            ctx_proc = []
            for i in range(num_obs):
                ctx_event = {item: i for item in params.get("context", [])}
                ctx_proc.append(ctx_event)
            context[proc] = ctx_proc
        return context

    @staticmethod
    def _build_event_list(
        timestamps: dict[str, np.ndarray],
        trans_refs: dict[str, list[list[dict]]],
        context: dict[str, list[dict]],
    ) -> list[dict]:
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
