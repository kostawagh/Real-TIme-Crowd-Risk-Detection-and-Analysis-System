"""
analytics_pipeline.py
Unified crowd analytics pipeline — Phase 2 post-processing.

Stages:
  1. FeatureExtractor  — tracking CSV  → motion features CSV
  2. DensityAnalyzer   — features CSV  → grid analytics CSV + plot
  3. RiskScorer        — analytics CSV → risk scores CSV

Usage (from another script):
    from analytics_pipeline import AnalyticsPipeline

    pipeline = AnalyticsPipeline(
        video_name  = "umn3",
        video_path  = "data/umn/umn3.mp4",
        output_root = "output",
    )
    pipeline.run()                  # run all three stages
    # or run individually:
    pipeline.extract_features()
    pipeline.analyze_density()
    pipeline.score_risk()

    # Access output paths after running:
    pipeline.paths["features"]
    pipeline.paths["analytics"]
    pipeline.paths["risk"]
"""

import os
import pandas as pd
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# Stage 1 — Feature Extraction
# ══════════════════════════════════════════════════════════════════
class FeatureExtractor:
    """
    Reads a tracking CSV (output of main.py) and computes
    per-frame, per-object motion features: velocity, dx, dy.

    Input  : output/tracking/tracking_<name>.csv
    Output : output/features/features_<name>.csv
    """

    def __init__(self, tracking_csv: str, output_dir: str):
        self.tracking_csv = tracking_csv
        self.output_dir   = output_dir

    def run(self) -> str:
        """Returns path to the saved features CSV."""
        df = pd.read_csv(self.tracking_csv)

        video_name = (
            os.path.splitext(os.path.basename(self.tracking_csv))[0]
            .replace("tracking_", "")
        )

        # Build per-object sorted trajectories
        tracks: dict[int, list] = {}
        for _, row in df.iterrows():
            obj_id = int(row["obj_id"])
            tracks.setdefault(obj_id, []).append(
                (int(row["frame"]), row["smooth_cx"], row["smooth_cy"])
            )

        for obj_id in tracks:
            tracks[obj_id].sort(key=lambda x: x[0])

        motion_data = []

        # per-frame motion and track-risk features
        for obj_id, traj in tracks.items():
            prev_angle = None

            for i in range(1, len(traj)):
                f_prev, x_prev, y_prev = traj[i - 1]
                f_curr, x_curr, y_curr = traj[i]

                dx = x_curr - x_prev
                dy = y_curr - y_prev
                velocity = (dx ** 2 + dy ** 2) ** 0.5
                angle = np.arctan2(dy, dx)

                # ── direction change (NEW, per frame) ──
                if prev_angle is None:
                    dir_change = 0.0
                else:
                    dtheta = angle - prev_angle
                    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                    dir_change = abs(dtheta)

                prev_angle = angle

                # acceleration
                acceleration = 0.0
                if i > 1:
                    prev_dx = x_prev - traj[i - 2][1]
                    prev_dy = y_prev - traj[i - 2][2]
                    prev_velocity = (prev_dx ** 2 + prev_dy ** 2) ** 0.5
                    acceleration = velocity - prev_velocity


                # Track-based Scores
                speed_score = min(velocity / 10.0, 1.0)
                acc_score = min(max(acceleration, 0.0) / 5.0, 1.0)
                dir_score = min(dir_change / np.pi, 1.0)

                track_risk = (
                    0.45 * speed_score +
                    0.40 * acc_score +
                    0.15 * dir_score
                )

                motion_data.append({
                    "frame": f_curr,
                    "obj_id": obj_id,
                    "cx": x_curr,
                    "cy": y_curr,
                    "velocity": velocity,
                    "acceleration": acceleration,
                    "dx": dx,
                    "dy": dy,
                    "angle": angle,
                    "dir_change": dir_change,   # ⭐ IMPORTANT
                    "track_risk": track_risk
                })

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"features_{video_name}.csv")
        pd.DataFrame(motion_data).to_csv(output_path, index=False)

        print(f"[FeatureExtractor] {len(motion_data)} rows → {output_path}")
        return output_path


# ══════════════════════════════════════════════════════════════════
# Stage 2 — Density / Grid Analytics
# ══════════════════════════════════════════════════════════════════
class DensityAnalyzer:
    """
    Divides the frame into a GRID_X x GRID_Y grid and computes
    per-cell crowd metrics: density, avg velocity, congestion,
    turbulence, and flow conflict.

    Input  : output/features/features_<name>.csv
    Output : output/analytics/analytics_<name>.csv
              output/analytics/analytics_<name>.png
    """

    EPS = 1e-5

    def __init__(
        self,
        features_csv : str,
        video_path   : str,
        output_dir   : str,
        grid_x       : int = 5,
        grid_y       : int = 5,
    ):
        self.features_csv = features_csv
        self.video_path   = video_path
        self.output_dir   = output_dir
        self.grid_x       = grid_x
        self.grid_y       = grid_y

    def _get_frame_size(self) -> tuple[int, int]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h

    def run(self) -> tuple[str, str]:
        """Returns (analytics_csv_path, plot_png_path)."""
        df    = pd.read_csv(self.features_csv)
        fw, fh = self._get_frame_size()
        print(f"[DensityAnalyzer] Frame size: {fw} × {fh}")

        cell_w = fw / self.grid_x
        cell_h = fh / self.grid_y
        GX, GY = self.grid_x, self.grid_y
        EPS    = self.EPS

        final_rows = []

        for frame_id in sorted(df["frame"].unique()):
            frame_df = df[df["frame"] == frame_id]

            density_grid   = np.zeros((GY, GX), dtype=int)
            velocity_sum   = np.zeros((GY, GX))
            count_grid     = np.zeros((GY, GX), dtype=int)
            acc_sum        = np.zeros((GY, GX))
            track_risk_sum = np.zeros((GY, GX))
            cell_dx        = [[[] for _ in range(GX)] for _ in range(GY)]
            cell_dy        = [[[] for _ in range(GX)] for _ in range(GY)]
            cell_dirs      = [[[] for _ in range(GX)] for _ in range(GY)]

            for _, row in frame_df.iterrows():
                gx = min(int(row["cx"] // cell_w), GX - 1)
                gy = min(int(row["cy"] // cell_h), GY - 1)

                v   = row["velocity"]
                dx  = row["dx"]
                dy  = row["dy"]
                acc = row.get("acceleration", 0.0)
                track_risk = row.get("track_risk", 0.0)
                

                density_grid[gy, gx] += 1
                velocity_sum[gy, gx] += v
                count_grid[gy, gx]   += 1
                acc_sum[gy, gx]      += acc
                track_risk_sum[gy,gx]+= track_risk

                cell_dx[gy][gx].append(dx)
                cell_dy[gy][gx].append(dy)

                mag = np.sqrt(dx ** 2 + dy ** 2 + EPS)
                cell_dirs[gy][gx].append((dx / mag, dy / mag))

            avg_velocity = velocity_sum / (count_grid + EPS)
            avg_acc      = acc_sum / (count_grid + EPS)
            avg_track_risk = track_risk_sum / (count_grid + EPS)

            # FIX 1: stable congestion (avoid explosion when velocity ~ 0)
            congestion_grid = density_grid / (avg_velocity + 1e-5 + EPS)

            turbulence_grid = np.zeros((GY, GX))
            flow_conflict_grid = np.zeros((GY, GX))

            for y in range(GY):
                for x in range(GX):

                    # FIX 2: turbulence slightly more meaningful (velocity variance proxy)
                    if len(cell_dx[y][x]) > 1:
                        vel_local = np.sqrt(
                            np.array(cell_dx[y][x])**2 +
                            np.array(cell_dy[y][x])**2
                        )
                        turbulence_grid[y, x] = np.var(vel_local)

                    dirs = cell_dirs[y][x]

                    # FIX 3: remove density bias from flow conflict
                    if len(dirs) > 1:
                        d_arr = np.array(dirs)
                        mean_dir = np.mean(d_arr, axis=0)
                        coherence = np.linalg.norm(mean_dir) / (np.linalg.norm(d_arr.sum(axis=0)) + EPS)
                        flow_conflict_grid[y, x] = 1 - coherence

            for y in range(GY):
                for x in range(GX):
                    final_rows.append({
                        "frame"        : frame_id,
                        "cell_x"       : x,
                        "cell_y"       : y,
                        "density"      : density_grid[y, x],
                        "avg_velocity" : avg_velocity[y, x],
                        "avg_acc"      : avg_acc[y, x],
                        "congestion"   : congestion_grid[y, x],
                        "turbulence"   : turbulence_grid[y, x],
                        "flow_conflict": flow_conflict_grid[y, x],
                        "avg_track_risk": avg_track_risk[y,x],
                    })

        final_df = pd.DataFrame(final_rows)

        video_name = (
            os.path.splitext(os.path.basename(self.features_csv))[0]
            .replace("features_", "")
        )

        os.makedirs(self.output_dir, exist_ok=True)
        csv_out  = os.path.join(self.output_dir, f"analytics_{video_name}.csv")
        plot_out = os.path.join(self.output_dir, f"analytics_{video_name}.png")

        final_df.to_csv(csv_out, index=False)
        print(f"[DensityAnalyzer] Saved analytics: {csv_out}")

        self._plot(final_df, plot_out)
        return csv_out, plot_out

    @staticmethod
    def _plot(df: pd.DataFrame, path: str):
        frame_ids, avg_cong, avg_turb, avg_flow = [], [], [], []

        for fid in sorted(df["frame"].unique()):
            fd = df[df["frame"] == fid]
            frame_ids.append(fid)
            avg_cong.append(fd["congestion"].mean())
            avg_turb.append(fd["turbulence"].mean())
            avg_flow.append(fd["flow_conflict"].mean())

        fig, axes = plt.subplots(3, 1, figsize=(12, 9))

        axes[0].plot(frame_ids, avg_cong, color="red", linewidth=2)
        axes[0].set_title("Crowd Congestion Over Time")
        axes[0].set_ylabel("Congestion")
        axes[0].grid(True, alpha=0.4)

        axes[1].plot(frame_ids, avg_turb, color="purple", linewidth=2)
        axes[1].set_title("Crowd Turbulence Over Time")
        axes[1].set_ylabel("Turbulence")
        axes[1].grid(True, alpha=0.4)

        axes[2].plot(frame_ids, avg_flow, color="blue", linewidth=2)
        axes[2].set_title("Flow Conflict Over Time")
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("Conflict")
        axes[2].grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[DensityAnalyzer] Saved plot    : {path}")


# ══════════════════════════════════════════════════════════════════
# Stage 3 — Risk Scoring
# ══════════════════════════════════════════════════════════════════
class RiskScorer:
    """
    Normalises the four grid metrics and computes a weighted risk score
    with a LOW / WARNING / DANGER label per cell per frame.

    Input  : output/analytics/analytics_<name>.csv
    Output : output/risk/risk_<name>.csv
    """

    EPS = 1e-5

    DEFAULT_WEIGHTS = {
        "density"      : 0.02,
        "dispersion"   : 0.00,
        "congestion"   : 0.06,
        "turbulence"   : 0.25,
        "flow_conflict": 0.03,
        "velocity"     : 0.34,
        "acceleration" : 0.22,
        "track_risk"   : 0.08,
    }

    THRESHOLDS = {"SAFE": 0.35, "WARNING": 0.8}

    def __init__(
        self,
        analytics_csv : str,
        output_dir    : str,
        weights       : dict | None = None,
    ):
        self.analytics_csv = analytics_csv
        self.output_dir    = output_dir
        self.weights       = weights or self.DEFAULT_WEIGHTS

    def run(self) -> str:
        df  = pd.read_csv(self.analytics_csv)
        EPS = self.EPS

        GLOBAL_LIMITS = {
            "avg_velocity": (0, 25),
            "avg_acc": (-10, 15),
            "turbulence": (0, 40),
            "congestion": (0, 15),
            "flow_conflict": (0, 1),
            "density": (0, 20),
            "avg_track_risk": (0, 1),
        }

        def normalize_global(series, col_name):
            lo, hi = GLOBAL_LIMITS[col_name]
            return np.clip((series - lo) / (hi - lo + EPS), 0, 1)

        # ── Normalization ─────────────────────────────────────────
        df["density_n"]       = normalize_global(df["density"],"density")
        df["dispersion_n"]    = 1.0 - df["density_n"]
        df["congestion_n"]    = normalize_global(df["congestion"],"congestion")
        
        df["turbulence_n"]    = normalize_global(df["turbulence"],"turbulence")
        df = df.sort_values(by=["cell_y", "cell_x", "frame"])
        df["turbulence_n"] = (
            df.groupby(["cell_y", "cell_x"])["turbulence_n"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
        )

        df["flow_conflict_n"] = normalize_global(df["flow_conflict"],"flow_conflict")
        df["velocity_n"]      = normalize_global(df["avg_velocity"],"avg_velocity")
        df["acc_n"]           = normalize_global(df["avg_acc"].fillna(0.0),"avg_acc")
        df["track_risk_n"]    = normalize_global(df["avg_track_risk"].fillna(0.0),"avg_track_risk")

        # ── Raw Risk (unchanged) ─────────────────────────────────
        w = self.weights
        df["risk"] = (
            w["density"]       * df["density_n"] +
            w["dispersion"]    * df["dispersion_n"] + 
            w["congestion"]    * df["congestion_n"] +
            w["turbulence"]    * df["turbulence_n"] +
            w["flow_conflict"] * df["flow_conflict_n"] +
            w["velocity"]      * df["velocity_n"]+
            w["acceleration"]  * df["acc_n"] +
            w["track_risk"]    * df["track_risk_n"]
        )

        # ─────────────────────────────────────────────────────────
        # Temporal Smoothing (EMA per cell)
        # ─────────────────────────────────────────────────────────
        df = df.sort_values(by=["cell_y", "cell_x", "frame"])

        alpha = 0.3  # smoothing factor (tune 0.2–0.4)

        df["risk_smooth"] = 0.0

        for (cx, cy), group in df.groupby(["cell_x", "cell_y"]):
            prev = None
            smoothed_vals = []

            for r in group["risk"]:
                if prev is None:
                    val = r
                else:
                    val = alpha * r + (1 - alpha) * prev

                smoothed_vals.append(val)
                prev = val

            df.loc[group.index, "risk_smooth"] = smoothed_vals

        # ── Risk Level (NOW based on smoothed risk) ───────────────
        lo, hi = self.THRESHOLDS["SAFE"], self.THRESHOLDS["WARNING"]

        df["risk_level"] = df["risk_smooth"].apply(
            lambda r: "SAFE" if r < lo else ("WARNING" if r < hi else "DANGER")
        )

        # ── Save ─────────────────────────────────────────────────
        video_name = (
            os.path.splitext(os.path.basename(self.analytics_csv))[0]
            .replace("analytics_", "")
        )

        os.makedirs(self.output_dir, exist_ok=True)
        out = os.path.join(self.output_dir, f"risk_{video_name}.csv")
        df.to_csv(out, index=False)

        print(f"[RiskScorer] Saved risk scores : {out}")
        print(f"[RiskScorer] Max risk per frame (first 10):")
        print(df.groupby("frame")["risk_smooth"].max().head(10).to_string())

        return out


# ══════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════
class AnalyticsPipeline:
    """
    Runs FeatureExtractor → DensityAnalyzer → RiskScorer in sequence
    and wires their file paths together automatically.

    Parameters
    ----------
    video_name  : str  — e.g. "umn3"  (must match tracking_<name>.csv)
    video_path  : str  — path to the original video file (needed for frame size)
    output_root : str  — root output directory (default: "output")
    grid_x/y    : int  — density grid dimensions
    weights     : dict — risk score weights (optional)

    After run(), paths to all outputs are in  self.paths  dict.
    """

    def __init__(
        self,
        video_name  : str,
        video_path  : str,
        output_root : str = "output",
        grid_x      : int = 5,
        grid_y      : int = 5,
        weights     : dict | None = None,
    ):
        self.video_name  = video_name
        self.video_path  = video_path
        self.root        = output_root
        self.grid_x      = grid_x
        self.grid_y      = grid_y
        self.weights     = weights

        # Derive canonical paths
        self._tracking_csv = os.path.join(
            output_root, "tracking", f"tracking_{video_name}.csv"
        )
        self.paths: dict[str, str] = {}

    # ── individual stages ──────────────────────────────────────────
    def extract_features(self) -> str:
        stage = FeatureExtractor(
            tracking_csv = self._tracking_csv,
            output_dir   = os.path.join(self.root, "features"),
        )
        path = stage.run()
        self.paths["features"] = path
        return path

    def analyze_density(self) -> tuple[str, str]:
        features_csv = self.paths.get(
            "features",
            os.path.join(self.root, "features", f"features_{self.video_name}.csv"),
        )
        stage = DensityAnalyzer(
            features_csv = features_csv,
            video_path   = self.video_path,
            output_dir   = os.path.join(self.root, "analytics"),
            grid_x       = self.grid_x,
            grid_y       = self.grid_y,
        )
        csv_path, plot_path = stage.run()
        self.paths["analytics"] = csv_path
        self.paths["plot"]      = plot_path
        return csv_path, plot_path

    def score_risk(self) -> str:
        analytics_csv = self.paths.get(
            "analytics",
            os.path.join(self.root, "analytics", f"analytics_{self.video_name}.csv"),
        )
        stage = RiskScorer(
            analytics_csv = analytics_csv,
            output_dir    = os.path.join(self.root, "risk"),
            weights       = self.weights,
        )
        path = stage.run()
        self.paths["risk"] = path
        return path

    # ── run all stages ─────────────────────────────────────────────
    def run(self) -> dict[str, str]:
        """Run all three stages. Returns self.paths dict."""
        print(f"\n{'='*55}")
        print(f"  AnalyticsPipeline  —  {self.video_name}")
        print(f"{'='*55}")
        self.extract_features()
        self.analyze_density()
        self.score_risk()
        print(f"\nAll outputs:")
        for k, v in self.paths.items():
            print(f"  {k:<12} → {v}")
        return self.paths
    
    