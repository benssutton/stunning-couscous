import { Component, OnInit, OnDestroy, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { catchError, takeUntil } from 'rxjs/operators';
import Plotly from 'plotly.js-dist-min';

import { ApiService } from '../../services/api.service';
import { EventCountsResponse, TTestResult } from '../../models/api.models';

type Metric = 'count' | 'rolling_avg' | 'cumulative_sum';

const PLOTLY_COLORS = [
  '#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657',
  '#79c0ff', '#56d364', '#ff7b72', '#bc8cff', '#ffb86c',
];

const BUCKET_OPTIONS = [
  { label: '1 second',  value: 1 },
  { label: '5 seconds', value: 5 },
  { label: '10 seconds', value: 10 },
  { label: '15 seconds', value: 15 },
  { label: '30 seconds', value: 30 },
  { label: '1 minute',  value: 60 },
];

@Component({
  selector: 'app-event-counts',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './event-counts.component.html',
  styleUrl: './event-counts.component.scss',
})
export class EventCountsComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('chartEl') chartEl!: ElementRef<HTMLDivElement>;

  eventNames: string[] = [];
  selectedEvent = '';
  selectedDates: string[] = [];
  newDate = '';
  bucketSeconds = 30;
  metric: Metric = 'count';

  bucketOptions = BUCKET_OPTIONS;
  metricOptions: { label: string; value: Metric }[] = [
    { label: 'Count', value: 'count' },
    { label: 'Rolling Average', value: 'rolling_avg' },
    { label: 'Cumulative Sum', value: 'cumulative_sum' },
  ];
  rollingWindow = 7;

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
        if (res.names.length) this.selectedEvent = res.names[0];
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
    if (this.chartInitialised) {
      Plotly.purge(this.chartEl.nativeElement);
    }
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

  refresh(): void {
    if (!this.selectedEvent || this.selectedDates.length === 0) {
      this.error = 'Select an event name and at least one date.';
      return;
    }
    this.loading = true;
    this.error = '';
    this.ttestResult = null;

    this.api.getEventCounts({
      event_name: this.selectedEvent,
      dates: this.selectedDates,
      bucket_seconds: this.bucketSeconds,
      metric: this.metric,
      rolling_window: this.rollingWindow,
    }).pipe(catchError(err => { this.error = 'Failed to load event counts.'; this.loading = false; throw err; }), takeUntil(this.destroy$))
    .subscribe(response => {
      this.renderChart(response);

      if (this.selectedDates.length >= 2) {
        this.runTTest(response);
      } else {
        this.loading = false;
      }
    });
  }

  private renderChart(response: EventCountsResponse): void {
    const traces: any[] = response.series.map((series, i) => ({
      x: series.buckets.map(b => b.time),
      y: series.buckets.map(b => b.value),
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: series.date,
      line: { color: PLOTLY_COLORS[i % PLOTLY_COLORS.length], width: 2 },
    }));

    Plotly.react(this.chartEl.nativeElement, traces, this.plotlyLayout(), { responsive: true });
  }

  private runTTest(response: EventCountsResponse): void {
    const [a, b] = this.selectedDates.slice(0, 2);
    const seriesA = response.series.find(s => s.date === a);
    const seriesB = response.series.find(s => s.date === b);

    if (!seriesA || !seriesB) { this.loading = false; return; }

    this.api.runTTest({
      series_a: seriesA.buckets.map(p => p.value),
      series_b: seriesB.buckets.map(p => p.value),
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
        title: this.metricOptions.find(m => m.value === this.metric)?.label ?? 'Value',
        gridcolor: '#21262d',
        color: '#8b949e',
      },
      legend: { bgcolor: '#161b22', bordercolor: '#30363d', borderwidth: 1 },
      margin: { t: 20, r: 20, b: 60, l: 60 },
    };
  }
}
