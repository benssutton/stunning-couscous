import polars as pl

from schemas.models import BucketPoint, DateSeries, EventCountsResponse


class EventCountsService:
    def build_response(
        self,
        df: pl.DataFrame,
        metric: str,
    ) -> EventCountsResponse:
        if df.is_empty():
            return EventCountsResponse(series=[])

        valid_metrics = {"count", "rolling_avg", "cumulative_sum"}
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{metric}'. Valid values: {sorted(valid_metrics)}")

        if metric == "cumulative_sum":
            df = df.with_columns(
                pl.col("count").cast(pl.Float64).cum_sum().over("date").alias("value")
            )
        elif metric == "rolling_avg":
            df = df.with_columns(
                pl.col("count")
                .cast(pl.Float64)
                .rolling_mean(window_size=7)
                .over("date")
                .fill_null(0.0)
                .alias("value")
            )
        else:  # count
            df = df.with_columns(pl.col("count").cast(pl.Float64).alias("value"))

        series = []
        for date_str in df["date"].unique().sort().to_list():
            date_df = df.filter(pl.col("date") == date_str).sort("bucket_time")
            buckets = [
                BucketPoint(
                    time=row["bucket_time"].strftime("%H:%M:%S"),
                    value=row["value"],
                )
                for row in date_df.iter_rows(named=True)
            ]
            series.append(DateSeries(date=date_str, buckets=buckets))

        return EventCountsResponse(series=series)
