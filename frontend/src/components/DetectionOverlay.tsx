// Canvas overlay that draws detection bounding boxes + label + confidence.
// The drawing itself lives in @/utils/drawDetections so this module exports
// components only (required for react-refresh).

import { useEffect, useRef } from 'react';
import { drawDetections } from '@/utils/drawDetections';
import type { Detection } from '@/types';

export function DetectionOverlay({
  detections,
  sourceW,
  sourceH,
  className,
}: {
  detections: Detection[];
  sourceW: number;
  sourceH: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    // Match the canvas backing store to the source aspect for crisp boxes.
    if (sourceW && sourceH && (canvas.width !== sourceW || canvas.height !== sourceH)) {
      canvas.width = sourceW;
      canvas.height = sourceH;
    }
    drawDetections(canvas, detections, sourceW, sourceH);
  }, [detections, sourceW, sourceH]);

  return <canvas ref={ref} className={className} />;
}
