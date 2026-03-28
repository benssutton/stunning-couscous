import { test, expect, Page } from '@playwright/test';
import refsFixture from './fixtures/refs-autocomplete.json';
import chainSearchFixture from './fixtures/chain-search.json';
import chainDetailFixture from './fixtures/chain-detail.json';
import latenciesFixture from './fixtures/latencies.json';
import avgLatenciesFixture from './fixtures/average-latencies.json';

async function mockAllApis(page: Page) {
  await page.route('**/search/refs*', (route) =>
    route.fulfill({ json: refsFixture }),
  );
  await page.route('**/search/chains*', (route) =>
    route.fulfill({ json: chainSearchFixture }),
  );
  await page.route('**/chains/*', (route) =>
    route.fulfill({ json: chainDetailFixture }),
  );
  await page.route('**/latencies/averages*', (route) =>
    route.fulfill({ json: avgLatenciesFixture }),
  );
  await page.route('**/latencies?*', (route) =>
    route.fulfill({ json: latenciesFixture }),
  );
  await page.route('**/latencies', (route) =>
    route.fulfill({ json: latenciesFixture }),
  );
}

/** Type into the ref input and wait for autocomplete to appear */
async function typeAndWaitForDropdown(page: Page, text: string) {
  const input = page.getByTestId('ref-input');
  await input.click();
  await input.pressSequentially(text, { delay: 50 });
  await expect(page.getByTestId('suggestions-dropdown')).toBeVisible({ timeout: 5000 });
}

/** Select first autocomplete suggestion via keyboard */
async function selectFirstSuggestion(page: Page) {
  const input = page.getByTestId('ref-input');
  await input.press('ArrowDown');
  await input.press('Enter');
}

test.describe('Event Chain Search', () => {
  test.beforeEach(async ({ page }) => {
    await mockAllApis(page);
    await page.goto('/');
  });

  test('page loads with search input', async ({ page }) => {
    await expect(page.getByTestId('ref-input')).toBeVisible();
    await expect(page.getByTestId('hours-input')).toBeVisible();
  });

  test('autocomplete does not trigger with fewer than 3 characters', async ({ page }) => {
    const input = page.getByTestId('ref-input');
    await input.click();
    await input.pressSequentially('sr', { delay: 50 });
    await page.waitForTimeout(500);
    await expect(page.getByTestId('suggestions-dropdown')).not.toBeVisible();
  });

  test('autocomplete triggers after 3 characters', async ({ page }) => {
    await typeAndWaitForDropdown(page, 'srch_');
    const items = page.getByTestId('suggestion-item');
    await expect(items).toHaveCount(5);
  });

  test('keyboard navigation and Enter selection', async ({ page }) => {
    await typeAndWaitForDropdown(page, 'srch_');

    const input = page.getByTestId('ref-input');
    await input.press('ArrowDown');
    await input.press('ArrowDown');
    await input.press('Enter');

    await expect(page.getByTestId('suggestions-dropdown')).not.toBeVisible();
    await expect(page.getByTestId('results')).toBeVisible({ timeout: 5000 });
  });

  test('chain table renders with edge data', async ({ page }) => {
    await typeAndWaitForDropdown(page, 'srch_');
    await selectFirstSuggestion(page);

    await expect(page.getByTestId('chain-table')).toBeVisible({ timeout: 5000 });

    const tableText = await page.getByTestId('chain-table').textContent();
    expect(tableText).toContain('A');
    expect(tableText).toContain('B');
    expect(tableText).toContain('1,200.0');
  });

  test('DAG renders with nodes and edges', async ({ page }) => {
    await typeAndWaitForDropdown(page, 'srch_');
    await selectFirstSuggestion(page);

    await expect(page.getByTestId('chain-dag')).toBeVisible({ timeout: 5000 });
    await expect(page.getByTestId('dag-svg')).toBeVisible();

    const nodeNames = page.getByTestId('dag-node-name');
    await expect(nodeNames.first()).toBeVisible();
    const names = await nodeNames.allTextContents();
    expect(names).toContain('A');
    expect(names).toContain('B');

    const edgeLatencies = page.getByTestId('dag-edge-latency');
    await expect(edgeLatencies.first()).toBeVisible();

    const avgLabels = page.getByTestId('dag-edge-avg');
    await expect(avgLabels.first()).toBeVisible();
    const avgText = await avgLabels.first().textContent();
    expect(avgText).toContain('avg:');
  });

  test('average latency labels have muted opacity', async ({ page }) => {
    await typeAndWaitForDropdown(page, 'srch_');
    await selectFirstSuggestion(page);

    await expect(page.getByTestId('dag-svg')).toBeVisible({ timeout: 5000 });

    const avgLabel = page.getByTestId('dag-edge-avg').first();
    const opacity = await avgLabel.getAttribute('opacity');
    expect(parseFloat(opacity!)).toBeLessThanOrEqual(0.7);
  });

  test('time period input changes average window', async ({ page }) => {
    let capturedStart = '';
    await page.route('**/latencies/averages*', (route) => {
      const url = new URL(route.request().url());
      capturedStart = url.searchParams.get('start') || '';
      return route.fulfill({ json: avgLatenciesFixture });
    });

    const hoursInput = page.getByTestId('hours-input');
    await hoursInput.fill('48');

    await typeAndWaitForDropdown(page, 'srch_');
    await selectFirstSuggestion(page);

    await expect(page.getByTestId('results')).toBeVisible({ timeout: 5000 });

    expect(capturedStart).toBeTruthy();
    const startDate = new Date(capturedStart);
    const hoursDiff = (Date.now() - startDate.getTime()) / 3600000;
    expect(hoursDiff).toBeGreaterThan(46);
    expect(hoursDiff).toBeLessThan(50);
  });

  test('shows error when no chains found', async ({ page }) => {
    await page.route('**/search/chains*', (route) =>
      route.fulfill({ json: { count: 0, chain_ids: [] } }),
    );

    await typeAndWaitForDropdown(page, 'srch_');
    await selectFirstSuggestion(page);

    await expect(page.getByTestId('error-message')).toBeVisible({ timeout: 5000 });
    const errorText = await page.getByTestId('error-message').textContent();
    expect(errorText).toContain('No chains found');
  });
});
