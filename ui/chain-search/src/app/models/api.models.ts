export interface RefAutocompleteResponse {
  prefix: string;
  results: string[];
}

export interface ChainSearchResponse {
  count: number;
  chain_ids: string[];
}

export interface ChainDetail {
  chain_id: string;
  concatenatedrefs: string[];
  timestamps: Record<string, string>;
  context: Record<string, string>;
  complete: boolean;
  terminated: boolean;
}

export interface ChainLatency {
  source: string;
  target: string;
  delta_ms: number;
}

export interface ChainLatencyResponse {
  chain_id: string;
  latencies: ChainLatency[];
}

export interface EdgeLatencyStats {
  source: string;
  target: string;
  avg_ms: number;
  stddev_ms: number;
  min_ms: number;
  max_ms: number;
  p5_ms: number;
  p50_ms: number;
  p95_ms: number;
  sample_count: number;
}

export interface AverageLatencyResponse {
  chain_id: string;
  profile_id: number;
  node_set: string[];
  matching_chains: number;
  start: string;
  end: string;
  edges: EdgeLatencyStats[];
}

// Event counts
export interface EventNamesResponse {
  names: string[];
}

export interface BucketPoint {
  time: string;   // HH:MM:SS
  value: number;
}

export interface DateSeries {
  date: string;   // YYYY-MM-DD
  buckets: BucketPoint[];
}

export interface EventCountsRequest {
  event_name: string;
  dates: string[];
  bucket_seconds: number;
  metric: 'count' | 'rolling_avg' | 'cumulative_sum';
}

export interface EventCountsResponse {
  series: DateSeries[];
}

// T-test
export interface TTestRequest {
  series_a: number[];
  series_b: number[];
  alpha?: number;
}

export interface TTestResult {
  t_statistic: number;
  p_value: number;
  degrees_of_freedom: number;
  significant: boolean;
  alpha: number;
}
