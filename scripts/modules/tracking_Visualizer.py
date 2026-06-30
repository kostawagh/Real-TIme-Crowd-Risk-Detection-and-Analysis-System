"""
visualizer.py
Draws bounding boxes, IDs, centroids, smoothed centroid trails,
and HUD overlay on frames.
"""

import cv2
import numpy as np
from collections import defaultdict


# Color palette — one color per ID (cycles)
PALETTE = [
    (0, 255, 0),    (0, 200, 255),  (255, 100, 0),
    (255, 0, 200),  (0, 255, 200),  (200, 255, 0),
    (100, 0, 255),  (255, 200, 0),  (0, 100, 255),
    (200, 0, 255),
]


class Visualizer:
    def __init__(self, config: dict):
        self.max_w    = config["display_w"]
        self.max_h    = config["display_h"]
        self.trails   = defaultdict(list)   # obj_id → list of smoothed centroids
        self.trail_len = 30                 # max trail length in frames

    def _color(self, obj_id: int) -> tuple:
        return PALETTE[obj_id % len(PALETTE)]

    def draw(self, frame, tracks: list[dict], frame_no: int, inference_ms: float) -> np.ndarray:
        annotated = frame.copy()

        for t in tracks:
            obj_id = t["obj_id"]
            x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
            cx, cy = int(t["smooth_cx"]), int(t["smooth_cy"])
            color = (255, 255, 255)  # white

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Centroid dot
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            # Trail (smoothed centroid history)
            self.trails[obj_id].append((cx, cy))
            if len(self.trails[obj_id]) > self.trail_len:
                self.trails[obj_id].pop(0)

            pts = self.trails[obj_id]
            for k in range(1, len(pts)):
                alpha = k / len(pts)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(annotated, pts[k - 1], pts[k], trail_color, 1, cv2.LINE_AA)

        # HUD overlay
        self._draw_hud(annotated, len(tracks), frame_no, inference_ms)

        # Resize for display
        h, w = annotated.shape[:2]
        scale   = min(self.max_w / w, self.max_h / h)
        display = cv2.resize(annotated,
                             (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return display

    @staticmethod
    def _draw_hud(frame, count: int, frame_no: int, inference_ms: float):
        """Top-left HUD with crowd count, FPS, frame number."""
        fps_val = 1000.0 / inference_ms if inference_ms > 0 else 0

        lines = [
            f"Tracked : {count}",
            f"FPS     : {fps_val:.1f}",
            f"Frame   : {frame_no}",
        ]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (200, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
