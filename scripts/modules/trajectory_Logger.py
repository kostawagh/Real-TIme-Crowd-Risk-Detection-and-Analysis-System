"""
logger.py
Per-frame + per-object CSV logger.
Clean schema optimized for ML feature extraction in Phase 3.
Supports post-run short-track filtering.
"""

import csv
import os
import pandas as pd
import numpy as np


# CSV columns — clean, ML-ready schema
COLUMNS = [
    "frame",        # frame number
    "obj_id",       # track ID
    "x1", "y1",     # bbox top-left
    "x2", "y2",     # bbox bottom-right
    "cx", "cy",     # raw centroid
    "smooth_cx",    # EMA smoothed centroid x
    "smooth_cy",    # EMA smoothed centroid y
    "track_age",    # how many frames this ID has been alive
    "inf_ms",       # inference time (ms) — only on first row per frame, else ""
]


class TrackLogger:
    def __init__(self, config: dict, video_name: str):
        self.csv_path = os.path.join(config["output_dir"], f"tracking_{video_name}.csv")
        self._file    = None
        self._writer  = None

    def __enter__(self):
        self._file   = open(self.csv_path, mode="w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def log_frame(self, frame_no: int, tracks: list[dict], inference_ms: float):
        """Log all tracks for a single frame. FPS only on first row."""
        for i, t in enumerate(tracks):
            row = [
                frame_no,
                t["obj_id"],
                round(t["x1"], 1),
                round(t["y1"], 1),
                round(t["x2"], 1),
                round(t["y2"], 1),
                t["cx"],
                t["cy"],
                t["smooth_cx"],
                t["smooth_cy"],
                t["age"],
                round(inference_ms, 2) if i == 0 else "",
            ]
            self._writer.writerow(row)

        self._file.flush()

    def filter_short_tracks(self, min_len: int = 5):
        """
        Post-process: remove track IDs that appear in fewer than
        min_len frames (noise/jitter filter). Overwrites the CSV.
        """
        df = pd.read_csv(self.csv_path)

        track_lengths = df.groupby("obj_id")["frame"].count()
        valid_ids     = track_lengths[track_lengths >= min_len].index

        before = len(df)
        df_filtered = df[df["obj_id"].isin(valid_ids)]
        after  = len(df_filtered)

        df_filtered.to_csv(self.csv_path, index=False)

        removed_tracks = len(track_lengths) - len(valid_ids)
        print(f"\nTrack filter: removed {removed_tracks} short tracks "
              f"({before - after} rows dropped). "
              f"{len(valid_ids)} valid tracks remain.")

    def fill_short_gaps(self, max_gap: int = 6):
        """
        Fill short missing gaps inside each track using linear interpolation.
        Helps recover temporary occlusions / missed detections.
        max_gap is in frame numbers, so with frame_skip=2, max_gap=6 fills up to ~3 processed steps.
        """
        df = pd.read_csv(self.csv_path)

        if df.empty:
            print("[GapFill] CSV empty, skipping.")
            return

        df = df.sort_values(["obj_id", "frame"]).reset_index(drop=True)
        filled_rows = []

        for obj_id, group in df.groupby("obj_id"):
            group = group.sort_values("frame").reset_index(drop=True)

            for i in range(len(group) - 1):
                curr = group.iloc[i]
                nxt = group.iloc[i + 1]

                filled_rows.append(curr.to_dict())

                f1 = int(curr["frame"])
                f2 = int(nxt["frame"])
                gap = f2 - f1

                if 1 < gap <= max_gap:
                    step = 2
                    missing_frames = list(range(f1 + step, f2, step))

                    for mf in missing_frames:
                        ratio = (mf - f1) / gap
                        new_row = curr.copy()

                        new_row["frame"] = mf

                        for col in ["x1", "y1", "x2", "y2", "cx", "cy", "smooth_cx", "smooth_cy"]:
                            new_row[col] = curr[col] + ratio * (nxt[col] - curr[col])

                        new_row["track_age"] = curr["track_age"]
                        new_row["inf_ms"] = np.nan

                        filled_rows.append(new_row.to_dict())

            filled_rows.append(group.iloc[-1].to_dict())

        df_filled = pd.DataFrame(filled_rows)
        df_filled = df_filled.sort_values(["frame", "obj_id"]).reset_index(drop=True)
        df_filled.to_csv(self.csv_path, index=False)

        print(f"[GapFill] Added {len(df_filled) - len(df)} interpolated rows.")