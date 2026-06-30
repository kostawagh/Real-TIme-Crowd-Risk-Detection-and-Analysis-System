"""
tracker.py
Tracker with:
  - ByteTrack (preferred) or SORT fallback
  - Centroid computation (cx, cy)
  - Per-track history buffer
  - Exponential moving average smoothing on centroids (alpha lowered for noisy head dets)
  - Short-track filtering at output level

BYTETRACK NOTE:
  ByteTrack is bundled with ultralytics >= 8.0.
  It associates BOTH high-conf AND low-conf detections with existing tracks,
  recovering 20-40% of persons missed by SORT in blur/occlusion scenarios.

  To use ByteTrack set  use_bytetrack: True  in CONFIG (main_pipeline.py).
  If ultralytics is not installed or ByteTrack init fails, falls back to SORT
  automatically with a warning.
"""

import numpy as np
from collections import defaultdict
from modules.sort_Engine import Sort


class TrackState:
    """Holds smoothed history for a single track ID."""

    # Lowered from 0.6 → 0.4: more smoothing for noisy/jittery head detections
    ALPHA = 0.4

    def __init__(self, obj_id: int):
        self.obj_id   = obj_id
        self.history  = []       # list of (cx, cy) raw centroids
        self.smoothed = None     # EMA smoothed (cx, cy)
        self.age      = 0

    def update(self, cx: float, cy: float):
        self.age += 1
        self.history.append((cx, cy))

        if self.smoothed is None:
            self.smoothed = np.array([cx, cy], dtype=np.float32)
        else:
            self.smoothed = (
                self.ALPHA * np.array([cx, cy], dtype=np.float32)
                + (1 - self.ALPHA) * self.smoothed
            )

    @property
    def smooth_cx(self) -> float:
        return float(self.smoothed[0]) if self.smoothed is not None else 0.0

    @property
    def smooth_cy(self) -> float:
        return float(self.smoothed[1]) if self.smoothed is not None else 0.0


# ─────────────────────────────────────────
# ByteTrack thin wrapper
# ─────────────────────────────────────────
class _ByteTrackWrapper:
    """
    Thin wrapper around ultralytics BYTETracker so CrowdTracker
    can call .update(dets) and get back (M, 5) [x1,y1,x2,y2,id]
    exactly like Sort does.
    """
    def __init__(self, config: dict):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from types import SimpleNamespace

        # BYTETracker expects an args namespace
        args = SimpleNamespace(
            track_high_thresh = config.get("conf_thres", 0.25),
            track_low_thresh  = config.get("conf_low_thres", 0.10),  # key ByteTrack param
            new_track_thresh  = config.get("conf_thres", 0.25),
            track_buffer      = config.get("max_age", 20),
            match_thresh      = config.get("iou_track", 0.20),
            fuse_score        = True,
            mot20             = False,
        )
        self.tracker    = BYTETracker(args, frame_rate=30)
        self._frame_id  = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        self._frame_id += 1

        if len(dets) == 0:
            return np.empty((0, 5), dtype=np.float32)

        class ResultsLike:
            def __init__(self, dets):
                self.xyxy = dets[:, :4].astype(np.float32)
                self.conf = dets[:, 4].astype(np.float32)
                self.cls = np.zeros(len(dets), dtype=np.float32)
                self.xywh = self._xyxy_to_xywh(self.xyxy)

            @staticmethod
            def _xyxy_to_xywh(xyxy):
                xywh = np.zeros_like(xyxy, dtype=np.float32)
                xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
                xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
                return xywh

            def __len__(self):
                return len(self.conf)

            def __getitem__(self, idx):
                new = ResultsLike.__new__(ResultsLike)

                new.xyxy = self.xyxy[idx]
                new.conf = self.conf[idx]
                new.cls = self.cls[idx]

                if new.xyxy.ndim == 1:
                    new.xyxy = new.xyxy.reshape(1, 4)
                    new.conf = np.array([new.conf], dtype=np.float32)
                    new.cls = np.array([new.cls], dtype=np.float32)

                new.xywh = self._xyxy_to_xywh(new.xyxy)

                return new
        results_like = ResultsLike(dets)

        online_targets = self.tracker.update(results_like)

        output = []

        for t in online_targets:
            # Newer Ultralytics may return ndarray rows
            if isinstance(t, np.ndarray):
                x1, y1, x2, y2, track_id = t[:5]
                output.append([x1, y1, x2, y2, track_id])
            else:
                tlwh = t.tlwh
                x1 = tlwh[0]
                y1 = tlwh[1]
                x2 = tlwh[0] + tlwh[2]
                y2 = tlwh[1] + tlwh[3]
                output.append([x1, y1, x2, y2, t.track_id])

        return (
            np.array(output, dtype=np.float32)
            if output
            else np.empty((0, 5), dtype=np.float32)
        )

    def reset(self):
        self._frame_id = 0
        self.tracker.reset()


# ─────────────────────────────────────────
# CrowdTracker
# ─────────────────────────────────────────
class CrowdTracker:
    def __init__(self, config: dict):
        self._config      = config
        self._use_byte    = config.get("use_bytetrack", False)
        self._tracker     = self._init_tracker()
        self.states: dict[int, TrackState] = defaultdict(lambda: None)

    def _init_tracker(self):
        if self._use_byte:
            try:
                tracker = _ByteTrackWrapper(self._config)
                print("[CrowdTracker] Using ByteTrack.")
                return tracker
            except Exception as e:
                print(f"[CrowdTracker] ByteTrack init failed ({e}), falling back to SORT.")

        print("[CrowdTracker] Using SORT.")
        return Sort(
            max_age       = self._config["max_age"],
            min_hits      = self._config["min_hits"],
            iou_threshold = self._config["iou_track"],
        )

    def update(self, dets: np.ndarray) -> list[dict]:
        """
        Update tracker with detections, compute centroids, apply EMA smoothing.

        Args:
            dets: (N, 5) array of [x1, y1, x2, y2, conf]

        Returns:
            List of track dicts with keys:
                obj_id, x1, y1, x2, y2, cx, cy, smooth_cx, smooth_cy, age
        """
        raw_tracks = self._tracker.update(dets)  # (M, 5) → [x1, y1, x2, y2, id]

        output = []
        for t in raw_tracks:
            x1, y1, x2, y2, obj_id = t
            obj_id = int(obj_id)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if self.states[obj_id] is None:
                self.states[obj_id] = TrackState(obj_id)

            state = self.states[obj_id]
            state.update(cx, cy)

            output.append({
                "obj_id"   : obj_id,
                "x1"       : float(x1),
                "y1"       : float(y1),
                "x2"       : float(x2),
                "y2"       : float(y2),
                "cx"       : round(cx, 2),
                "cy"       : round(cy, 2),
                "smooth_cx": round(state.smooth_cx, 2),
                "smooth_cy": round(state.smooth_cy, 2),
                "age"      : state.age,
            })

        return output

    def get_history(self, obj_id: int) -> list[tuple]:
        """Return full centroid history for a given track ID."""
        state = self.states.get(obj_id)
        return state.history if state else []
