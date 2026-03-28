import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject, Subscription, forkJoin, of } from 'rxjs';
import { catchError, debounceTime, distinctUntilChanged, filter, switchMap } from 'rxjs/operators';

import { ApiService } from '../../services/api.service';
import {
  ChainDetail,
  ChainLatencyResponse,
  AverageLatencyResponse,
} from '../../models/api.models';
import { ChainTableComponent } from '../chain-table/chain-table.component';
import { ChainDagComponent } from '../chain-dag/chain-dag.component';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, FormsModule, ChainTableComponent, ChainDagComponent],
  templateUrl: './search.component.html',
  styleUrl: './search.component.scss',
})
export class SearchComponent implements OnInit, OnDestroy {
  query = '';
  hoursBack = 24;
  suggestions: string[] = [];
  highlightedIndex = -1;
  showDropdown = false;
  loading = false;
  error = '';

  chainDetail: ChainDetail | null = null;
  latencies: ChainLatencyResponse | null = null;
  averageLatencies: AverageLatencyResponse | null = null;

  private input$ = new Subject<string>();
  private sub!: Subscription;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.sub = this.input$
      .pipe(
        debounceTime(300),
        distinctUntilChanged(),
        filter((q) => q.length >= 3),
        switchMap((q) => this.api.autocompleteRefs(q)),
      )
      .subscribe({
        next: (res) => {
          this.suggestions = res.results;
          this.highlightedIndex = -1;
          this.showDropdown = this.suggestions.length > 0;
        },
        error: () => {
          this.suggestions = [];
          this.showDropdown = false;
        },
      });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  onInput(value: string): void {
    this.query = value;
    if (value.length < 3) {
      this.suggestions = [];
      this.showDropdown = false;
      return;
    }
    this.input$.next(value);
  }

  onKeydown(event: KeyboardEvent): void {
    if (!this.showDropdown) return;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        this.highlightedIndex = Math.min(this.highlightedIndex + 1, this.suggestions.length - 1);
        break;
      case 'ArrowUp':
        event.preventDefault();
        this.highlightedIndex = Math.max(this.highlightedIndex - 1, 0);
        break;
      case 'Enter':
      case 'Tab':
        if (this.highlightedIndex >= 0) {
          event.preventDefault();
          this.selectRef(this.suggestions[this.highlightedIndex]);
        }
        break;
      case 'Escape':
        this.showDropdown = false;
        break;
    }
  }

  selectRef(refId: string): void {
    this.query = refId;
    this.showDropdown = false;
    this.suggestions = [];
    this.loadChain(refId);
  }

  private loadChain(refId: string): void {
    this.loading = true;
    this.error = '';
    this.chainDetail = null;
    this.latencies = null;
    this.averageLatencies = null;

    this.api.searchChains(refId).subscribe({
      next: (res) => {
        if (res.count === 0) {
          this.loading = false;
          this.error = 'No chains found for this reference ID.';
          return;
        }
        const chainId = res.chain_ids[0];
        const start = new Date(Date.now() - this.hoursBack * 3600000).toISOString();

        forkJoin({
          detail: this.api.getChainDetail(chainId),
          latencies: this.api.getLatencies(chainId).pipe(catchError(() => of([] as ChainLatencyResponse[]))),
          averages: this.api.getAverageLatencies(chainId, start).pipe(catchError(() => of(null as AverageLatencyResponse | null))),
        }).subscribe({
          next: (data) => {
            this.chainDetail = data.detail;
            this.latencies = data.latencies.length > 0 ? data.latencies[0] : null;
            this.averageLatencies = data.averages;
            this.loading = false;
            if (!this.latencies) {
              this.error = 'Latency data not available. Ensure adjacency matrix has been computed.';
            }
          },
          error: () => {
            this.loading = false;
            this.error = 'Failed to load chain details.';
          },
        });
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to search for chains.';
      },
    });
  }
}
