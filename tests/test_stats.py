import pytest
from services.stats_service import StatsService

pytestmark = pytest.mark.asyncio(loop_scope="session")


def test_ttest_significant():
    svc = StatsService()
    # Two clearly different distributions
    a = [10.0, 11.0, 10.5, 10.2, 11.1, 10.8, 10.3, 11.2, 10.6, 10.9]
    b = [20.0, 21.0, 19.5, 20.8, 21.3, 19.9, 20.5, 21.1, 20.2, 19.7]
    result = svc.run_ttest(a, b, alpha=0.05)
    assert result.significant is True
    assert result.p_value < 0.05
    assert result.degrees_of_freedom > 0
    assert result.alpha == 0.05


def test_ttest_not_significant():
    svc = StatsService()
    # Two samples from the same distribution
    a = [10.0, 10.1, 9.9, 10.2, 10.0, 9.8, 10.1, 10.3, 9.7, 10.0]
    b = [10.1, 10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 10.2, 10.1, 9.9]
    result = svc.run_ttest(a, b, alpha=0.05)
    assert result.significant is False
    assert result.p_value >= 0.05


def test_ttest_custom_alpha():
    svc = StatsService()
    a = [10.0, 11.0, 10.5, 10.2, 11.1]
    b = [10.3, 10.4, 10.1, 10.5, 10.2]
    result = svc.run_ttest(a, b, alpha=0.5)
    assert result.alpha == 0.5
    assert result.significant == (result.p_value < 0.5)
