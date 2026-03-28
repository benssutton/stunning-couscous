from scipy import stats as scipy_stats
from schemas.models import TTestResult


class StatsService:
    def run_ttest(
        self,
        series_a: list[float],
        series_b: list[float],
        alpha: float = 0.05,
    ) -> TTestResult:
        result = scipy_stats.ttest_ind(series_a, series_b, equal_var=False)
        return TTestResult(
            t_statistic=float(result.statistic),
            p_value=float(result.pvalue),
            degrees_of_freedom=int(result.df),
            significant=bool(result.pvalue < alpha),
            alpha=alpha,
        )
