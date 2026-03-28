import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ChainDetail,
  ChainLatencyResponse,
  AverageLatencyResponse,
  EdgeLatencyStats,
} from '../../models/api.models';

interface TableRow {
  source: string;
  target: string;
  sourceTimestamp: string;
  targetTimestamp: string;
  deltaMs: number;
  avgMs: number | null;
  p50Ms: number | null;
  p95Ms: number | null;
  sampleCount: number | null;
}

@Component({
  selector: 'app-chain-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './chain-table.component.html',
  styleUrl: './chain-table.component.scss',
})
export class ChainTableComponent {
  @Input() chainDetail!: ChainDetail;
  @Input() latencies!: ChainLatencyResponse;
  @Input() averageLatencies: AverageLatencyResponse | null = null;

  get rows(): TableRow[] {
    if (!this.latencies) return [];

    const avgMap = new Map<string, EdgeLatencyStats>();
    if (this.averageLatencies) {
      for (const edge of this.averageLatencies.edges) {
        avgMap.set(`${edge.source}->${edge.target}`, edge);
      }
    }

    return this.latencies.latencies.map((lat) => {
      const key = `${lat.source}->${lat.target}`;
      const avg = avgMap.get(key);
      return {
        source: lat.source,
        target: lat.target,
        sourceTimestamp: this.chainDetail?.timestamps[lat.source] ?? '',
        targetTimestamp: this.chainDetail?.timestamps[lat.target] ?? '',
        deltaMs: lat.delta_ms,
        avgMs: avg?.avg_ms ?? null,
        p50Ms: avg?.p50_ms ?? null,
        p95Ms: avg?.p95_ms ?? null,
        sampleCount: avg?.sample_count ?? null,
      };
    });
  }
}
