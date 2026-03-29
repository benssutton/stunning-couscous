import { Routes } from '@angular/router';
import { SearchComponent } from './components/search/search.component';
import { EventCountsComponent } from './components/event-counts/event-counts.component';
import { LatencyTimeseriesComponent } from './components/latency-timeseries/latency-timeseries.component';
import { PlaceholderComponent } from './components/placeholder/placeholder.component';

export const routes: Routes = [
  { path: 'search', component: SearchComponent },
  { path: 'counts', component: EventCountsComponent },
  { path: 'latencies', component: LatencyTimeseriesComponent },
  { path: 'realtime', component: PlaceholderComponent, data: { title: 'Real-time Dashboard' } },
  { path: 'models', component: PlaceholderComponent, data: { title: 'Models' } },
  { path: '', redirectTo: 'search', pathMatch: 'full' },
  { path: '**', redirectTo: 'search' },
];
