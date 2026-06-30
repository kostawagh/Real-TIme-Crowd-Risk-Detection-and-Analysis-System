"""
detector.py
YOLOv8 person/head detector with:
  - Optional preprocessing (unsharp mask + CLAHE) for motion blur
  - Tiled inference for small/distant targets in dense crowds
  - Soft-NMS overlap filtering
Returns detections in [x1, y1, x2, y2, conf] format.

"""

import time
import cv2
import numpy as np
from ultralytics import YOLO


class CrowdDetector:
    def __init__(self, config: dict):
        self.model      = YOLO(config["model_path"])
        self.conf       = config["conf_thres"]
        self.iou        = config["iou_thres"]
        self.img_size   = config["img_size"]

        # Head-detection mode tweaks NMS and overlap thresholds
        self.detect_heads   = config.get("detect_heads", False)
        self.use_preprocess = config.get("use_preprocess", True)
        self.use_tiling     = config.get("use_tiling", False)

        # Overlap filter threshold — looser for heads (they sit close together)
        self._overlap_thresh = 0.95 if not self.detect_heads else 0.80

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Run inference on a single frame.

        Returns:
            dets (np.ndarray): shape (N, 5) — [x1, y1, x2, y2, conf]
            inference_ms (float): total processing time in milliseconds
        """
        t0 = time.perf_counter()

        if self.use_preprocess:
            frame = self._preprocess(frame)

        if self.use_tiling:
            dets = self._detect_tiled(frame)
        else:
            dets = self._run_yolo(frame)

        dets = self._filter_overlaps(dets, self._overlap_thresh)

        inference_ms = (time.perf_counter() - t0) * 1000.0
        return dets, inference_ms

    # ─────────────────────────────────────────
    # Preprocessing — motion blur + contrast
    # ─────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Unsharp masking to recover edges lost to motion blur,
        followed by CLAHE on the luminance channel to boost
        contrast in dense/low-light crowd scenes.
        """
        # Unsharp mask
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharp    = cv2.addWeighted(frame, 1.8, gaussian, -0.8, 0)

        # CLAHE on L channel (LAB space)
        lab              = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        clahe            = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0]     = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ─────────────────────────────────────────
    # Single-pass YOLO inference
    # ─────────────────────────────────────────
    def _run_yolo(self, frame: np.ndarray, offset_xy: tuple = (0, 0)) -> np.ndarray:
        """
        Run YOLO on one frame/tile and return (N,5) detections.
        offset_xy shifts bbox coords back into full-frame space when tiling.
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=[0],
            verbose=False,
            augment=True   
        )

        dx, dy = offset_xy
        dets   = []
        for r in results:
            if r.boxes is None:
                continue

            for box, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box

                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh
                aspect = bh / (bw + 1e-6)

                # reject invalid boxes
                if bw <= 0 or bh <= 0:
                    continue

                # reject tiny detections (shoes, heads, noise)
                if area < 1800:
                    continue

                # remove very thin/wide non-human shapes
                if aspect < 1.1 or aspect > 4.8:
                    continue

                # remove very low-confidence false humans
                if conf < self.conf:
                    continue        

                dets.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy, float(conf)])

        return (
            np.array(dets, dtype=np.float32)
            if dets
            else np.empty((0, 5), dtype=np.float32)
        )

    # ─────────────────────────────────────────
    # Tiled inference — forces small targets
    # ─────────────────────────────────────────
    def _detect_tiled(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on the full frame + 4 quadrant tiles, then merge.
        Helps detect small/distant persons that get squashed at full scale.
        ~5× compute cost — recommended with frame_skip >= 3.
        """
        h, w    = frame.shape[:2]
        tiles   = [
            (frame,                    (0,    0   )),   # full frame
            (frame[:h//2, :w//2],      (0,    0   )),   # top-left
            (frame[:h//2, w//2:],      (w//2, 0   )),   # top-right
            (frame[h//2:, :w//2],      (0,    h//2)),   # bottom-left
            (frame[h//2:, w//2:],      (w//2, h//2)),   # bottom-right
        ]

        all_dets = []
        for tile, (dx, dy) in tiles:
            d = self._run_yolo(tile, offset_xy=(dx, dy))
            if len(d):
                all_dets.append(d)

        return (
            np.vstack(all_dets)
            if all_dets
            else np.empty((0, 5), dtype=np.float32)
        )

    # ─────────────────────────────────────────
    # Overlap filter
    # ─────────────────────────────────────────
    @staticmethod
    def _filter_overlaps(dets: np.ndarray, overlap_thresh: float = 0.95) -> np.ndarray:
        if len(dets) == 0:
            return dets

        # sort by confidence (VERY IMPORTANT FIX)
        dets = dets[np.argsort(-dets[:, 4])]

        keep = []
        for i in range(len(dets)):
            x1, y1, x2, y2, _ = dets[i]
            area_i = (x2 - x1) * (y2 - y1)

            keep_flag = True

            for j in keep:
                xx1 = max(x1, dets[j, 0])
                yy1 = max(y1, dets[j, 1])
                xx2 = min(x2, dets[j, 2])
                yy2 = min(y2, dets[j, 3])

                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                area_j = (dets[j, 2] - dets[j, 0]) * (dets[j, 3] - dets[j, 1])

                # only suppress if almost fully inside AND lower confidence
                if inter / (area_i + 1e-6) > overlap_thresh:
                    keep_flag = False
                    break

            if keep_flag:
                keep.append(i)

        return dets[keep]