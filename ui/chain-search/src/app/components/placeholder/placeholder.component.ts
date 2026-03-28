import { Component, input } from '@angular/core';

@Component({
  selector: 'app-placeholder',
  standalone: true,
  template: `
    <div class="placeholder-page">
      <h2>{{ title() }}</h2>
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
export class PlaceholderComponent {
  title = input<string>('This screen');
}
