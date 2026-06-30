"""
Mod_SortTracker.py
Custom SORT implementation with Kalman filtering.
Fixes: memory leak, class counter reset, tracker pruning order, min_hits logic.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────
# Kalman Box Tracker
# ─────────────────────────────────────────
class KalmanBoxTracker:
    _count = 0  # name-mangled to reduce accidental access

    @classmethod
    def reset_count(cls):
        """Call this between video runs to reset ID counter."""
        cls._count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition: [cx, cy, s, r, vx, vy, vs]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P         *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.time_since_update = 0
        self.hits              = 0
        self.hit_streak        = 0
        self.age               = 0

        self._last_bbox = None   # replaces unbounded history list

    def update(self, bbox):
        self.time_since_update = 0
        self.hits             += 1
        self.hit_streak       += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self._last_bbox = self._convert_x_to_bbox(self.kf.x)
        return self._last_bbox   # (4,) — no unbounded list

    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)

    @staticmethod
    def _convert_bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        w  = max(0.0, x2 - x1)
        h  = max(0.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s  = max(0.0, w * h)
        r  = w / (h + 1e-6)
        return np.array([cx, cy, s, r], dtype=np.float32).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x):
        cx, cy, s, r = x[:4].reshape(-1)
        s  = max(0.0, s)
        r  = max(1e-6, r)
        w  = np.sqrt(s * r)
        h  = s / (w + 1e-6)
        return np.array([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0
        ], dtype=np.float32)


# ─────────────────────────────────────────
# SORT Tracker Manager
# ─────────────────────────────────────────
class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def reset(self):
        """Full reset between video runs."""
        self.trackers    = []
        self.frame_count = 0
        KalmanBoxTracker.reset_count()

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Args:
            detections: (N, 5) array [x1, y1, x2, y2, conf]
        Returns:
            (M, 5) array [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict new positions for all existing trackers
        predicted = []
        for t in self.trackers:
            predicted.append(t.predict())   # (4,)
        predicted = np.array(predicted, dtype=np.float32)  # (T, 4)

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate(detections, predicted)

        # Update matched trackers
        for trk_idx, det_idx in matched:
            self.trackers[trk_idx].update(dets_to_bbox(detections[det_idx]))

        # Spawn new trackers for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets_to_bbox(detections[det_idx])))

        # ✅ Collect results BEFORE pruning dead trackers
        results = []
        for t in self.trackers:
            if t.time_since_update <= 1 and t.hit_streak >= self.min_hits:
                bbox = t.get_state()
                results.append(np.append(bbox, t.id))

        # Prune dead trackers AFTER collecting results
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        return np.array(results, dtype=np.float32) if results else np.empty((0, 5), dtype=np.float32)

    def _associate(self, detections, trackers):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(trackers)))

        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)

        for t, trk in enumerate(trackers):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = compute_iou(trk, det[:4])

        # Guard against numerical issues
        iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched         = []
        unmatched_dets  = list(range(len(detections)))
        unmatched_trks  = list(range(len(trackers)))

        for t, d in zip(row_ind, col_ind):
            if iou_matrix[t, d] < self.iou_threshold:
                continue
            matched.append((t, d))
            unmatched_dets.remove(d)
            unmatched_trks.remove(t)

        return matched, unmatched_dets, unmatched_trks


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def compute_iou(bb1: np.ndarray, bb2: np.ndarray) -> float:
    xx1 = max(bb1[0], bb2[0])
    yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2])
    yy2 = min(bb1[3], bb2[3])

    w     = max(0.0, xx2 - xx1)
    h     = max(0.0, yy2 - yy1)
    inter = w * h

    area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])
    union = area1 + area2 - inter + 1e-6

    return inter / union


def dets_to_bbox(det: np.ndarray) -> np.ndarray:
    return det[:4].astype(np.float32)
