import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-placeholder',
  standalone: true,
  imports: [],
  template: `
    <div class="placeholder-page">
      <h2>{{ title }}</h2>
      <p>Coming soon.</p>
    </div>
  `,
  styles: [`
    .placeholder-page {
      padding: 80px 32px 32px;
      color: #8b949e;
      h2 { color: #e6edf3; margin-bottom: 8px; }
    }
  `],
})
export class PlaceholderComponent implements OnInit {
  title = 'This screen';

  constructor(private route: ActivatedRoute) {}

  ngOnInit(): void {
    this.title = this.route.snapshot.data['title'] ?? 'This screen';
  }
}
