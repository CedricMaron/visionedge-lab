// Canvas drawing helpers for detection overlays.
// Kept out of the component module so react-refresh can treat that file as
// components-only. Coordinates from the backend are in SOURCE pixel space; they
// are scaled here to the rendered canvas size.

import type { Detection } from '@/types';

// Stable, readable palette keyed by classId.
const COLORS = [
  '#38bdf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa',
  '#f472b6', '#22d3ee', '#a3e635', '#fb923c', '#e879f9',
];

export function colorForClass(classId: number): string {
  return COLORS[Math.abs(classId) % COLORS.length];
}

export function drawDetections(
  canvas: HTMLCanvasElement,
  detections: Detection[],
  sourceW: number,
  sourceH: number,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!sourceW || !sourceH) return;

  const sx = canvas.width / sourceW;
  const sy = canvas.height / sourceH;
  ctx.font = '600 13px ui-monospace, monospace';
  ctx.textBaseline = 'top';

  for (const d of detections) {
    const color = colorForClass(d.classId);
    const x = d.x1 * sx;
    const y = d.y1 * sy;
    const w = (d.x2 - d.x1) * sx;
    const h = (d.y2 - d.y1) * sy;

    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const label = `${d.className} ${(d.confidence * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    const th = 18;
    const ly = y - th >= 0 ? y - th : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw + 10, th);
    ctx.fillStyle = '#0b0f17';
    ctx.fillText(label, x + 5, ly + 2);
  }
}
