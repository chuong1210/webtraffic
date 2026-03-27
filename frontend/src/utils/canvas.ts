/**
 * Canvas utilities – draw bounding boxes, ROI, counting line, HUD.
 */

import type { Detection } from '../types/detection';

/* Neutral blue/gray palette for vehicle classes (professional, not flashy) */
const CLASS_COLORS: Record<string, string> = {
  car:        '#2563eb',
  truck:      '#475569',
  bus:        '#0ea5e9',
  motorcycle: '#64748b',
  bicycle:    '#0891b2',
  person:     '#6366f1',
};

function color(cls: string): string {
  return CLASS_COLORS[cls] ?? '#aaaaaa';
}

// ── Bounding Boxes ────────────────────────────────────────────────────────────

export function drawDetections(
  ctx: CanvasRenderingContext2D,
  detections: Detection[],
  scaleX: number,
  scaleY: number
): void {
  detections.forEach((det) => {
    const { x1, y1, x2, y2 } = det.bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sw  = (x2 - x1) * scaleX;
    const sh  = (y2 - y1) * scaleY;
    const c   = color(det.class_name);

    // Box
    ctx.strokeStyle = c;
    ctx.lineWidth = 2;
    ctx.shadowColor = c;
    ctx.shadowBlur = 6;
    ctx.strokeRect(sx1, sy1, sw, sh);
    ctx.shadowBlur = 0;

    // Label background (include ID if available)
    const label =
      det.track_id != null
        ? `${det.class_name} ID${det.track_id} ${(det.confidence * 100).toFixed(0)}%`
        : `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    ctx.font = 'bold 12px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    const th = 18;
    ctx.fillStyle = c;
    ctx.fillRect(sx1, sy1 - th - 2, tw + 8, th + 2);

    // Label text
    ctx.fillStyle = '#000';
    ctx.textBaseline = 'bottom';
    ctx.fillText(label, sx1 + 4, sy1 - 2);
  });
}

// ── ROI Polygon ───────────────────────────────────────────────────────────────

export function drawRoi(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  active: boolean
): void {
  if (points.length === 0) return;

  const c = active ? '#2563eb' : '#64748b';
  ctx.save();
  ctx.strokeStyle = c;
  ctx.lineWidth = 2;
  ctx.setLineDash(active ? [] : [6, 3]);
  ctx.shadowColor = c;
  ctx.shadowBlur = 6;

  if (points.length >= 3) {
    ctx.fillStyle = active ? 'rgba(37,99,235,0.08)' : 'rgba(100,116,139,0.06)';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fill();
  }

  ctx.beginPath();
  points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
  if (points.length >= 3) ctx.closePath();
  ctx.stroke();

  // Vertex dots
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? '#2563eb' : '#64748b';
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.8)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.stroke();
  });

  ctx.restore();
}

// ── Counting Line ─────────────────────────────────────────────────────────────

export function drawCountingLine(
  ctx: CanvasRenderingContext2D,
  lineY: number,
  width: number
): void {
  ctx.save();
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 4]);
  ctx.shadowColor = '#2563eb';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.moveTo(0, lineY);
  ctx.lineTo(width, lineY);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.setLineDash([]);
  ctx.restore();
}
