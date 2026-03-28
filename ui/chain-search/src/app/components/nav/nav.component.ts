import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-nav',
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './nav.component.html',
  styleUrl: './nav.component.scss',
})
export class NavComponent {
  readonly navItems = [
    { label: 'Search', route: '/search' },
    { label: 'Counts', route: '/counts' },
    { label: 'Latencies', route: '/latencies' },
    { label: 'Real-time Dashboard', route: '/realtime' },
    { label: 'Models', route: '/models' },
  ];
}
