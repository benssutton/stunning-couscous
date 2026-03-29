import { Component, OnInit, OnDestroy, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { catchError, takeUntil } from 'rxjs/operators';
import Plotly from 'plotly.js-dist-min';

import { ApiService } from '../../services/api.service';
import { LatencyTimeseriesResponse, TTestResult } from '../../models/api.models';

type LatencyMetric = 'mean_ms' | 'min_ms' | 'max_ms' | 'p5_ms' | 'p50_ms' | 'p95_ms';

const PLOTLY_COLORS = [
  '#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657',
  '#79c0ff', '#56d364', '#ff7b72', '#bc8cff', '#ffb86c',
];

const METRIC_LINE_STYLES: Record<LatencyMetric, string> = {
  mean_ms:  'solid',
  p50_ms:   'dash',
  p5_ms:    'dot',
  p95_ms:   'dot',
  min_ms:   'dashdot',
  max_ms:   'dashdot',
};

const METRIC_LABELS: Record<LatencyMetric, string> = {
  mean_ms: 'Mean',
  min_ms:  'Min',
  max_ms:  'Max',
  p5_ms:   'P5',
  p50_ms:  'P50',
  p95_ms:  'P95',
};

const BUCKET_OPTIONS = [
  { label: '1 second',   value: 1 },
  { label: '5 seconds',  value: 5 },
  { label: '10 seconds', value: 10 },
  { label: '15 seconds', value: 15 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute',   value: 60 },
];

@Component({
  selector: 'app-latency-timeseries',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './latency-timeseries.component.html',
  styleUrl: './latency-timeseries.component.scss',
})
export class LatencyTimeseriesComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('chartEl') chartEl!: ElementRef<HTMLDivElement>;

  eventNames: string[] = [];
  sourceEvent = '';
  targetEvent = '';

  selectedDates: string[] = [];
  newDate = '';
  bucketSeconds = 30;

  selectedMetrics: Set<LatencyMetric> = new Set(['mean_ms', 'p50_ms', 'p95_ms']);
  metricOptions: { label: string; value: LatencyMetric }[] = [
    { label: 'Mean',  value: 'mean_ms' },
    { label: 'Min',   value: 'min_ms'  },
    { label: 'Max',   value: 'max_ms'  },
    { label: 'P5',    value: 'p5_ms'   },
    { label: 'P50',   value: 'p50_ms'  },
    { label: 'P95',   value: 'p95_ms'  },
  ];

  bucketOptions = BUCKET_OPTIONS;

  loading = false;
  error = '';
  ttestResult: TTestResult | null = null;
  ttestDates: string[] = [];

  private chartInitialised = false;
  private destroy$ = new Subject<void>();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getEventNames().pipe(takeUntil(this.destroy$)).subscribe({
      next: (res) => {
        this.eventNames = res.names;
        if (res.names.length >= 1) this.sourceEvent = res.names[0];
        if (res.names.length >= 2) this.targetEvent = res.names[1];
      },
      error: () => { this.error = 'Failed to load event names.'; },
    });
  }

  ngAfterViewInit(): void {
    Plotly.newPlot(this.chartEl.nativeElement, [], this.plotlyLayout(), { responsive: true });
    this.chartInitialised = true;
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.chartInitialised) Plotly.purge(this.chartEl.nativeElement);
  }

  addDate(): void {
    if (this.newDate && !this.selectedDates.includes(this.newDate)) {
      this.selectedDates = [...this.selectedDates, this.newDate];
    }
    this.newDate = '';
  }

  removeDate(date: string): void {
    this.selectedDates = this.selectedDates.filter(d => d !== date);
    if (this.selectedDates.length < 2) this.ttestResult = null;
  }

  dateColor(index: number): string {
    return PLOTLY_COLORS[index % PLOTLY_COLORS.length];
  }

  toggleMetric(metric: LatencyMetric): void {
    if (this.selectedMetrics.has(metric)) {
      this.selectedMetrics.delete(metric);
    } else {
      this.selectedMetrics.add(metric);
    }
    this.selectedMetrics = new Set(this.selectedMetrics);
  }

  isMetricSelected(metric: LatencyMetric): boolean {
    return this.selectedMetrics.has(metric);
  }

  refresh(): void {
    if (!this.sourceEvent || !this.targetEvent) {
      this.error = 'Select source and target events.';
      return;
    }
    if (this.selectedDates.length === 0) {
      this.error = 'Select at least one date.';
      return;
    }
    if (this.selectedMetrics.size === 0) {
      this.error = 'Select at least one metric.';
      return;
    }
    this.loading = true;
    this.error = '';
    this.ttestResult = null;

    this.api.getLatencyTimeseries({
      source_event: this.sourceEvent,
      target_event: this.targetEvent,
      dates: this.selectedDates,
      bucket_seconds: this.bucketSeconds,
    })
    .pipe(
      catchError(err => {
        this.error = 'Failed to load latency data.';
        this.loading = false;
        throw err;
      }),
      takeUntil(this.destroy$),
    )
    .subscribe(response => {
      this.renderChart(response);
      if (this.selectedDates.length >= 2) {
        this.runTTest(response);
      } else {
        this.loading = false;
      }
    });
  }

  private renderChart(response: LatencyTimeseriesResponse): void {
    const traces: any[] = [];

    response.series.forEach((series, dateIdx) => {
      const color = PLOTLY_COLORS[dateIdx % PLOTLY_COLORS.length];
      const times = series.buckets.map(b => b.time);

      // Latency metric traces (left y-axis)
      for (const metric of this.selectedMetrics) {
        traces.push({
          x: times,
          y: series.buckets.map(b => (b as any)[metric]),
          type: 'scatter',
          mode: 'lines',
          name: `${series.date} ${METRIC_LABELS[metric]}`,
          line: { color, width: 2, dash: METRIC_LINE_STYLES[metric] },
          yaxis: 'y',
        });
      }

      // Event count overlay (right y-axis), bar chart
      traces.push({
        x: times,
        y: series.buckets.map(b => b.event_count),
        type: 'bar',
        name: `${series.date} Count`,
        marker: { color, opacity: 0.25 },
        yaxis: 'y2',
      });
    });

    Plotly.react(this.chartEl.nativeElement, traces, this.plotlyLayout(), { responsive: true });
  }

  private runTTest(response: LatencyTimeseriesResponse): void {
    const [a, b] = this.selectedDates.slice(0, 2);
    const seriesA = response.series.find(s => s.date === a);
    const seriesB = response.series.find(s => s.date === b);

    if (!seriesA?.raw_latencies.length || !seriesB?.raw_latencies.length) {
      this.loading = false;
      return;
    }

    this.api.runTTest({
      series_a: seriesA.raw_latencies,
      series_b: seriesB.raw_latencies,
      alpha: 0.05,
    }).pipe(takeUntil(this.destroy$)).subscribe({
      next: (result) => {
        this.ttestResult = result;
        this.ttestDates = [a, b];
        this.loading = false;
      },
      error: () => {
        this.error = 'Failed to run T-test.';
        this.loading = false;
      },
    });
  }

  private plotlyLayout(): object {
    return {
      paper_bgcolor: '#161b22',
      plot_bgcolor: '#161b22',
      font: { color: '#e6edf3', size: 12 },
      xaxis: {
        title: 'Time of day',
        gridcolor: '#21262d',
        color: '#8b949e',
        type: 'category',
      },
      yaxis: {
        title: 'Latency (ms)',
        gridcolor: '#21262d',
        color: '#8b949e',
      },
      yaxis2: {
        title: 'Event count',
        overlaying: 'y',
        side: 'right',
        color: '#8b949e',
        showgrid: false,
      },
      barmode: 'overlay',
      legend: { bgcolor: '#161b22', bordercolor: '#30363d', borderwidth: 1 },
      margin: { t: 20, r: 60, b: 60, l: 60 },
    };
  }
}
