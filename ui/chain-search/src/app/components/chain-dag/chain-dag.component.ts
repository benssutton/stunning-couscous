import {
  Component,
  Input,
  ElementRef,
  ViewChild,
  AfterViewInit,
  OnChanges,
  SimpleChanges,
} from '@angular/core';
import * as d3 from 'd3';
import * as dagre from 'dagre';
import {
  ChainDetail,
  ChainLatencyResponse,
  AverageLatencyResponse,
  EdgeLatencyStats,
} from '../../models/api.models';

const NODE_WIDTH = 200;
const NODE_HEIGHT = 56;
const PADDING = 40;

@Component({
  selector: 'app-chain-dag',
  standalone: true,
  template: `
    <section class="chain-dag" data-testid="chain-dag">
      <h2>Event Chain DAG</h2>
      <div #dagContainer class="dag-container"></div>
    </section>
  `,
  styleUrl: './chain-dag.component.scss',
})
export class ChainDagComponent implements AfterViewInit, OnChanges {
  @Input() chainDetail!: ChainDetail;
  @Input() latencies!: ChainLatencyResponse;
  @Input() averageLatencies: AverageLatencyResponse | null = null;

  @ViewChild('dagContainer', { static: true }) container!: ElementRef<HTMLDivElement>;

  private ready = false;

  ngAfterViewInit(): void {
    this.ready = true;
    this.render();
  }

  ngOnChanges(_changes: SimpleChanges): void {
    if (this.ready) this.render();
  }

  private render(): void {
    const el = this.container.nativeElement;
    el.innerHTML = '';

    if (!this.chainDetail || !this.latencies || this.latencies.latencies.length === 0) return;

    const avgMap = new Map<string, EdgeLatencyStats>();
    if (this.averageLatencies) {
      for (const edge of this.averageLatencies.edges) {
        avgMap.set(`${edge.source}->${edge.target}`, edge);
      }
    }

    // Build dagre graph
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'TB', nodesep: 80, ranksep: 100, marginx: PADDING, marginy: PADDING });
    g.setDefaultEdgeLabel(() => ({}));

    // Collect nodes from latency edges
    const nodes = new Set<string>();
    for (const lat of this.latencies.latencies) {
      nodes.add(lat.source);
      nodes.add(lat.target);
    }

    for (const node of nodes) {
      g.setNode(node, { label: node, width: NODE_WIDTH, height: NODE_HEIGHT });
    }

    for (const lat of this.latencies.latencies) {
      g.setEdge(lat.source, lat.target);
    }

    dagre.layout(g);

    const graphInfo = g.graph();
    const svgWidth = (graphInfo.width ?? 600) + PADDING * 2;
    const svgHeight = (graphInfo.height ?? 400) + PADDING * 2;

    const svg = d3
      .select(el)
      .append('svg')
      .attr('width', svgWidth)
      .attr('height', svgHeight)
      .attr('data-testid', 'dag-svg');

    // Arrowhead marker
    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 10)
      .attr('refY', 5)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 Z')
      .attr('fill', '#666');

    // Render edges
    for (const edgeObj of g.edges()) {
      const edgeData = g.edge(edgeObj);
      const points = edgeData.points as { x: number; y: number }[];

      const lineGen = d3
        .line<{ x: number; y: number }>()
        .x((d) => d.x)
        .y((d) => d.y)
        .curve(d3.curveBasis);

      svg
        .append('path')
        .attr('d', lineGen(points)!)
        .attr('fill', 'none')
        .attr('stroke', '#888')
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)')
        .attr('data-testid', 'dag-edge');

      // Edge labels at midpoint
      const mid = points[Math.floor(points.length / 2)];
      const lat = this.latencies.latencies.find(
        (l) => l.source === edgeObj.v && l.target === edgeObj.w,
      );
      const avg = avgMap.get(`${edgeObj.v}->${edgeObj.w}`);

      if (lat) {
        svg
          .append('text')
          .attr('x', mid.x + 8)
          .attr('y', mid.y - 4)
          .attr('font-size', '11px')
          .attr('font-family', 'Consolas, Monaco, monospace')
          .attr('fill', '#333')
          .attr('data-testid', 'dag-edge-latency')
          .text(`${lat.delta_ms.toFixed(1)}ms`);
      }

      if (avg) {
        svg
          .append('text')
          .attr('x', mid.x + 8)
          .attr('y', mid.y + 12)
          .attr('font-size', '10px')
          .attr('font-family', 'Consolas, Monaco, monospace')
          .attr('fill', '#999')
          .attr('opacity', 0.6)
          .attr('data-testid', 'dag-edge-avg')
          .text(`avg: ${avg.avg_ms.toFixed(1)}ms`);
      }
    }

    // Render nodes
    for (const nodeId of g.nodes()) {
      const nodeData = g.node(nodeId);
      const group = svg.append('g').attr('data-testid', 'dag-node');

      group
        .append('rect')
        .attr('x', nodeData.x - NODE_WIDTH / 2)
        .attr('y', nodeData.y - NODE_HEIGHT / 2)
        .attr('width', NODE_WIDTH)
        .attr('height', NODE_HEIGHT)
        .attr('rx', 6)
        .attr('ry', 6)
        .attr('fill', '#fff')
        .attr('stroke', '#4a6fa5')
        .attr('stroke-width', 1.5);

      // Event name
      group
        .append('text')
        .attr('x', nodeData.x)
        .attr('y', nodeData.y - 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', '13px')
        .attr('font-weight', '600')
        .attr('fill', '#1a1a2e')
        .attr('data-testid', 'dag-node-name')
        .text(nodeId);

      // Timestamp
      const ts = this.chainDetail.timestamps[nodeId];
      if (ts) {
        const display = ts.replace('T', ' ').substring(0, 23);
        group
          .append('text')
          .attr('x', nodeData.x)
          .attr('y', nodeData.y + 16)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#888')
          .attr('font-family', 'Consolas, Monaco, monospace')
          .text(display);
      }
    }
  }
}
