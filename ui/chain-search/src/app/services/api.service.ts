import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  RefAutocompleteResponse,
  ChainSearchResponse,
  ChainDetail,
  ChainLatencyResponse,
  AverageLatencyResponse,
  EventNamesResponse,
  EventCountsRequest,
  EventCountsResponse,
  TTestRequest,
  TTestResult,
  LatencyTimeseriesRequest,
  LatencyTimeseriesResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  autocompleteRefs(prefix: string, limit = 20): Observable<RefAutocompleteResponse> {
    const params = new HttpParams().set('q', prefix).set('limit', limit);
    return this.http.get<RefAutocompleteResponse>('/search/refs', { params });
  }

  searchChains(ref: string, limit = 100): Observable<ChainSearchResponse> {
    const params = new HttpParams().set('ref', ref).set('limit', limit);
    return this.http.get<ChainSearchResponse>('/search/chains', { params });
  }

  getChainDetail(chainId: string): Observable<ChainDetail> {
    return this.http.get<ChainDetail>(`/chains/${encodeURIComponent(chainId)}`);
  }

  getLatencies(chainId: string): Observable<ChainLatencyResponse[]> {
    const params = new HttpParams().set('chain_id', chainId);
    return this.http.get<ChainLatencyResponse[]>('/latencies', { params });
  }

  getAverageLatencies(chainId: string, start: string): Observable<AverageLatencyResponse> {
    const params = new HttpParams().set('chain_id', chainId).set('start', start);
    return this.http.get<AverageLatencyResponse>('/latencies/averages', { params });
  }

  getEventNames(): Observable<EventNamesResponse> {
    return this.http.get<EventNamesResponse>('/events/names');
  }

  getEventCounts(request: EventCountsRequest): Observable<EventCountsResponse> {
    return this.http.post<EventCountsResponse>('/events/counts', request);
  }

  runTTest(request: TTestRequest): Observable<TTestResult> {
    return this.http.post<TTestResult>('/stats/ttest', request);
  }

  getLatencyTimeseries(request: LatencyTimeseriesRequest): Observable<LatencyTimeseriesResponse> {
    return this.http.post<LatencyTimeseriesResponse>('/latencies/timeseries', request);
  }
}
