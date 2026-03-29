import { test, expect, Page } from '@playwright/test';

const EVENT_NAMES_FIXTURE = { names: ['EventA', 'EventB', 'EventC'] };

const COUNTS_FIXTURE = {
  series: [
    {
      date: '2026-01-01',
      buckets: [
        { time: '00:00:00', value: 5 },
        { time: '00:00:30', value: 8 },
        { time: '00:01:00', value: 3 },
      ],
    },
  ],
};

const COUNTS_TWO_DATES_FIXTURE = {
  series: [
    {
      date: '2026-01-01',
      buckets: [
        { time: '00:00:00', value: 5 },
        { time: '00:00:30', value: 8 },
      ],
    },
    {
      date: '2026-01-02',
      buckets: [
        { time: '00:00:00', value: 12 },
        { time: '00:00:30', value: 7 },
      ],
    },
  ],
};

const TTEST_FIXTURE = {
  t_statistic: -1.87,
  p_value: 0.0721,
  degrees_of_freedom: 3,
  significant: false,
  alpha: 0.05,
};

async function mockApis(page: Page, countFixture = COUNTS_FIXTURE) {
  await page.route('**/events/names*', (route) =>
    route.fulfill({ json: EVENT_NAMES_FIXTURE }),
  );
  await page.route('**/events/counts*', (route) =>
    route.fulfill({ json: countFixture }),
  );
  await page.route('**/stats/ttest*', (route) =>
    route.fulfill({ json: TTEST_FIXTURE }),
  );
}

test.describe('Event Counts screen', () => {
  test.beforeEach(async ({ page }) => {
    await mockApis(page);
    await page.goto('/counts');
  });

  test('event name dropdown is populated from /events/names', async ({ page }) => {
    const select = page.locator('select').first();
    await expect(select).toBeVisible();
    const options = await select.locator('option').allTextContents();
    expect(options).toContain('EventA');
    expect(options).toContain('EventB');
  });

  test('can add and remove a date chip', async ({ page }) => {
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await expect(page.locator('.date-chip')).toHaveCount(1);

    await page.locator('.remove-btn').first().click();
    await expect(page.locator('.date-chip')).toHaveCount(0);
  });

  test('bucket size select contains expected options', async ({ page }) => {
    const selects = page.locator('select');
    const bucketSelect = selects.nth(1);
    const options = await bucketSelect.locator('option').allTextContents();
    expect(options.some(o => o.includes('second') || o.includes('minute'))).toBeTruthy();
  });

  test('rolling window input is hidden for count metric', async ({ page }) => {
    const countRadio = page.locator('input[type="radio"][value="count"]');
    await countRadio.check();
    await expect(page.locator('input[type="number"]')).not.toBeVisible();
  });

  test('rolling window input is hidden for cumulative_sum metric', async ({ page }) => {
    const radio = page.locator('input[type="radio"][value="cumulative_sum"]');
    await radio.check();
    await expect(page.locator('input[type="number"]')).not.toBeVisible();
  });

  test('rolling window input is visible for rolling_avg metric', async ({ page }) => {
    const radio = page.locator('input[type="radio"][value="rolling_avg"]');
    await radio.check();
    await expect(page.locator('input[type="number"]')).toBeVisible();
  });

  test('rolling window input accepts valid values', async ({ page }) => {
    const radio = page.locator('input[type="radio"][value="rolling_avg"]');
    await radio.check();
    const windowInput = page.locator('input[type="number"]');
    await windowInput.fill('14');
    await expect(windowInput).toHaveValue('14');
  });

  test('chart renders after clicking Refresh with one date', async ({ page }) => {
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.getByRole('button', { name: 'Refresh' }).click();
    // Plotly creates a <canvas> or SVG inside the chart div
    await expect(page.locator('.plotly-chart')).toBeVisible();
  });

  test('T-test panel is hidden with one date', async ({ page }) => {
    await mockApis(page, COUNTS_FIXTURE);
    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.getByRole('button', { name: 'Refresh' }).click();
    await expect(page.locator('.ttest-panel')).not.toBeVisible();
  });

  test('T-test panel is shown with two dates', async ({ page }) => {
    await mockApis(page, COUNTS_TWO_DATES_FIXTURE);
    await page.goto('/counts');

    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('input[type="date"]').fill('2026-01-02');
    await page.getByText('Add').click();

    await page.getByRole('button', { name: 'Refresh' }).click();
    await expect(page.locator('.ttest-panel')).toBeVisible({ timeout: 5000 });
  });

  test('T-test panel shows p-value', async ({ page }) => {
    await mockApis(page, COUNTS_TWO_DATES_FIXTURE);
    await page.goto('/counts');

    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.locator('input[type="date"]').fill('2026-01-02');
    await page.getByText('Add').click();

    await page.getByRole('button', { name: 'Refresh' }).click();
    await expect(page.locator('.ttest-panel')).toBeVisible({ timeout: 5000 });
    const panelText = await page.locator('.ttest-panel').textContent();
    expect(panelText).toContain('0.0721');
  });

  test('rolling_window value is sent in request body', async ({ page }) => {
    let capturedBody: any = null;
    await page.route('**/events/counts*', async (route) => {
      capturedBody = route.request().postDataJSON();
      return route.fulfill({ json: COUNTS_FIXTURE });
    });
    await page.goto('/counts');

    const radio = page.locator('input[type="radio"][value="rolling_avg"]');
    await radio.check();
    const windowInput = page.locator('input[type="number"]');
    await windowInput.fill('14');

    await page.locator('input[type="date"]').fill('2026-01-01');
    await page.getByText('Add').click();
    await page.getByRole('button', { name: 'Refresh' }).click();

    await page.waitForTimeout(500);
    expect(capturedBody?.rolling_window).toBe(14);
  });
});
