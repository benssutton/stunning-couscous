import { test, expect, Page } from '@playwright/test';

const EVENT_NAMES_FIXTURE = { names: ['EventA', 'EventB', 'EventC'] };

const TIMESERIES_ONE_DATE_FIXTURE = {
  source_event: 'EventA',
  target_event: 'EventB',
  series: [
    {
      date: '2026-01-01',
      buckets: [
        { time: '00:00:00', mean_ms: 120, min_ms: 80, max_ms: 200, p5_ms: 85, p50_ms: 115, p95_ms: 190, event_count: 42 },
        { time: '00:00:30', mean_ms: 135, min_ms: 90, max_ms: 210, p5_ms: 95, p50_ms: 130, p95_ms: 200, event_count: 38 },
      ],
      raw_latencies: [80, 100, 120, 140, 160, 190, 200],
    },
  ],
};

const TIMESERIES_TWO_DATES_FIXTURE = {
  source_event: 'EventA',
  target_event: 'EventB',
  series: [
    {
      date: '2026-01-01',
      buckets: [
        { time: '00:00:00', mean_ms: 120, min_ms: 80, max_ms: 200, p5_ms: 85, p50_ms: 115, p95_ms: 190, event_count: 42 },
      ],
      raw_latencies: [80, 100, 120, 140, 160],
    },
    {
      date: '2026-01-02',
      buckets: [
        { time: '00:00:00', mean_ms: 250, min_ms: 180, max_ms: 380, p5_ms: 185, p50_ms: 245, p95_ms: 370, event_count: 55 },
      ],
      raw_latencies: [180, 220, 260, 310, 370],
    },
  ],
};

const TTEST_FIXTURE = {
  t_statistic: -3.45,
  p_value: 0.0061,
  degrees_of_freedom: 7,
  significant: true,
  alpha: 0.05,
};

async function mockApis(page: Page, timeseriesFixture = TIMESERIES_ONE_DATE_FIXTURE) {
  await page.route('**/events/names*', (route) =>
    route.fulfill({ json: EVENT_NAMES_FIXTURE }),
  );
  await page.route('**/latencies/timeseries*', (route) =>
    route.fulfill({ json: timeseriesFixture }),
  );
  await page.route('**/stats/ttest*', (route) =>
    route.fulfill({ json: TTEST_FIXTURE }),
  );
}

test.describe('Latency Time-Series screen', () => {
  test.beforeEach(async ({ page }) => {
    await mockApis(page);
    await page.goto('/latencies');
  });

  test('source and target event dropdowns are populated', async ({ page }) => {
    const selects = page.locator('select');
    const sourceSelect = selects.first();
    const targetSelect = selects.nth(1);

    await expect(sourceSelect).toBeVisible();
    await expect(targetSelect).toBeVisible();

    const sourceOptions = await sourceSelect.locator('option').allTextContents();
    expect(sourceOptions).toContain('EventA');
    expect(sourceOptions).toContain('EventB');

    const targetOptions = await targetSelect.locator('option').allTextContents();
    expect(targetOptions).toContain('EventA');
  });

  test('can select different source and target events', async ({ page }) => {
    const sourceSelect = page.locator('[data-testid="source-event-select"]');
    const targetSelect = page.locator('[data-testid="target-event-select"]');

    await sourceSelect.selectOption('EventA');
    await targetSelect.selectOption('EventC');

    await expect(sourceSelect).toHaveValue('EventA');
    await expect(targetSelect).toHaveValue('EventC');
  });

  test('can add and remove a date chip', async ({ page }) => {
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await expect(page.locator('.date-chip')).toHaveCount(1);

    await page.locator('.remove-btn').first().click();
    await expect(page.locator('.date-chip')).toHaveCount(0);
  });

  test('metric checkboxes are present and toggleable', async ({ page }) => {
    const meanCheckbox = page.locator('[data-testid="metric-mean_ms"] input[type="checkbox"]');
    await expect(meanCheckbox).toBeVisible();
    await expect(meanCheckbox).toBeChecked();

    await meanCheckbox.uncheck();
    await expect(meanCheckbox).not.toBeChecked();

    await meanCheckbox.check();
    await expect(meanCheckbox).toBeChecked();
  });

  test('all six metric checkboxes are present', async ({ page }) => {
    const metrics = ['mean_ms', 'min_ms', 'max_ms', 'p5_ms', 'p50_ms', 'p95_ms'];
    for (const m of metrics) {
      await expect(page.locator(`[data-testid="metric-${m}"]`)).toBeVisible();
    }
  });

  test('chart renders after Refresh with one date', async ({ page }) => {
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('[data-testid="refresh-btn"]').click();
    await expect(page.locator('[data-testid="latency-chart"]')).toBeVisible();
  });

  test('T-test panel is hidden with one date', async ({ page }) => {
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('[data-testid="refresh-btn"]').click();
    await expect(page.locator('[data-testid="ttest-panel"]')).not.toBeVisible();
  });

  test('T-test panel is shown with two dates', async ({ page }) => {
    await mockApis(page, TIMESERIES_TWO_DATES_FIXTURE);
    await page.goto('/latencies');

    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('input[type="date"]').fill('2026-01-02');
    await page.getByText('Add').click();

    await page.locator('[data-testid="refresh-btn"]').click();
    await expect(page.locator('[data-testid="ttest-panel"]')).toBeVisible({ timeout: 5000 });
  });

  test('T-test panel shows p-value and significance badge', async ({ page }) => {
    await mockApis(page, TIMESERIES_TWO_DATES_FIXTURE);
    await page.goto('/latencies');

    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('input[type="date"]').fill('2026-01-02');
    await page.getByText('Add').click();

    await page.locator('[data-testid="refresh-btn"]').click();
    await expect(page.locator('[data-testid="ttest-panel"]')).toBeVisible({ timeout: 5000 });

    const panelText = await page.locator('[data-testid="ttest-panel"]').textContent();
    expect(panelText).toContain('0.0061');
    expect(panelText).toContain('SIGNIFICANT');
  });

  test('error shown when no event selected', async ({ page }) => {
    // Remove event names so dropdowns are empty
    await page.route('**/events/names*', (route) =>
      route.fulfill({ json: { names: [] } }),
    );
    await page.goto('/latencies');

    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('[data-testid="refresh-btn"]').click();

    await expect(page.locator('[data-testid="error-msg"]')).toBeVisible();
  });

  test('request body contains correct source/target events', async ({ page }) => {
    let capturedBody: any = null;
    await page.route('**/latencies/timeseries*', async (route) => {
      capturedBody = route.request().postDataJSON();
      return route.fulfill({ json: TIMESERIES_ONE_DATE_FIXTURE });
    });
    await page.goto('/latencies');

    await page.locator('[data-testid="source-event-select"]').selectOption('EventA');
    await page.locator('[data-testid="target-event-select"]').selectOption('EventB');
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('[data-testid="refresh-btn"]').click();

    await page.waitForTimeout(500);
    expect(capturedBody?.source_event).toBe('EventA');
    expect(capturedBody?.target_event).toBe('EventB');
  });
});
