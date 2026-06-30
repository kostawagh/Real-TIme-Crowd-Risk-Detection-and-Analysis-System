import os
import csv
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd


class RealtimeMLEngine:
    EPS = 1e-5

    GLOBAL_LIMITS = {
        "avg_velocity": (0, 25),
        "avg_acc": (-10, 15),
        "turbulence": (0, 40),
        "congestion": (0, 15),
        "flow_conflict": (0, 1),
        "density": (0, 20),
        "avg_track_risk": (0, 1),
    }

    RISK_WEIGHTS = {
        "density": 0.02,
        "dispersion": 0.00,
        "congestion": 0.06,
        "turbulence": 0.25,
        "flow_conflict": 0.03,
        "velocity": 0.34,
        "acceleration": 0.22,
        "track_risk": 0.08,
    }

    def __init__(
        self,
        model_path,
        output_csv,
        frame_width,
        frame_height,
        grid_x=5,
        grid_y=5,
        safe_threshold=0.35,
        warning_threshold=0.70,
        risk_alpha=0.3,
        turbulence_window=5,
    ):
        package = joblib.load(model_path)

        self.model = package["model"]
        self.features = package["features"]
        self.binary_threshold = package.get("threshold", 0.5)

        self.output_csv = output_csv

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.grid_x = grid_x
        self.grid_y = grid_y

        self.cell_w = frame_width / grid_x
        self.cell_h = frame_height / grid_y

        self.safe_threshold = safe_threshold
        self.warning_threshold = warning_threshold

        self.risk_alpha = risk_alpha

        self.track_history = defaultdict(list)

        self.turbulence_history = {
            (x, y): deque(maxlen=turbulence_window)
            for y in range(grid_y)
            for x in range(grid_x)
        }

        self.prev_risk = {}

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        self.csv_file = open(output_csv, "w", newline="")
        self.writer = None

    def close(self):
        if self.csv_file:
            self.csv_file.close()

    def normalize(self, value, col_name):
        lo, hi = self.GLOBAL_LIMITS[col_name]
        return float(np.clip((value - lo) / (hi - lo + self.EPS), 0, 1))

    def get_ml_risk_level(self, probability):
        if probability < self.safe_threshold:
            return "Safe"
        elif probability < self.warning_threshold:
            return "Warning"
        return "Danger"

    def get_manual_risk_level(self, risk_smooth):
        if risk_smooth < 0.35:
            return "SAFE"
        elif risk_smooth < 0.8:
            return "WARNING"
        return "DANGER"

    def _extract_motion_rows(self, frame_no, tracks):
        motion_rows = []

        for track in tracks:
            obj_id = int(track["obj_id"])

            # IMPORTANT:
            # Training pipeline uses smooth_cx / smooth_cy,
            # so realtime inference must also use smoothed centroids.
            cx = float(track["smooth_cx"])
            cy = float(track["smooth_cy"])

            hist = self.track_history[obj_id]
            hist.append((frame_no, cx, cy))

            if len(hist) < 2:
                continue

            _, x_prev, y_prev = hist[-2]
            _, x_curr, y_curr = hist[-1]

            dx = x_curr - x_prev
            dy = y_curr - y_prev

            velocity = float((dx ** 2 + dy ** 2) ** 0.5)
            angle = float(np.arctan2(dy, dx))

            acceleration = 0.0
            dir_change = 0.0

            if len(hist) > 2:
                _, x_prev2, y_prev2 = hist[-3]

                prev_dx = x_prev - x_prev2
                prev_dy = y_prev - y_prev2

                prev_velocity = float((prev_dx ** 2 + prev_dy ** 2) ** 0.5)
                acceleration = velocity - prev_velocity

                prev_angle = float(np.arctan2(prev_dy, prev_dx))
                dtheta = angle - prev_angle
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                dir_change = abs(dtheta)

            speed_score = min(velocity / 10.0, 1.0)
            acc_score = min(max(acceleration, 0.0) / 5.0, 1.0)
            dir_score = min(dir_change / np.pi, 1.0)

            track_risk = (
                0.45 * speed_score +
                0.40 * acc_score +
                0.15 * dir_score
            )

            motion_rows.append({
                "frame": frame_no,
                "obj_id": obj_id,
                "cx": cx,
                "cy": cy,
                "velocity": velocity,
                "acceleration": acceleration,
                "dx": dx,
                "dy": dy,
                "angle": angle,
                "dir_change": dir_change,
                "track_risk": track_risk,
            })

        return motion_rows

    def process_frame(self, frame_no, tracks):
        motion_rows = self._extract_motion_rows(frame_no, tracks)

        GX, GY = self.grid_x, self.grid_y

        density_grid = np.zeros((GY, GX), dtype=int)
        velocity_sum = np.zeros((GY, GX))
        count_grid = np.zeros((GY, GX), dtype=int)
        acc_sum = np.zeros((GY, GX))
        track_risk_sum = np.zeros((GY, GX))

        cell_dx = [[[] for _ in range(GX)] for _ in range(GY)]
        cell_dy = [[[] for _ in range(GX)] for _ in range(GY)]
        cell_dirs = [[[] for _ in range(GX)] for _ in range(GY)]

        for row in motion_rows:
            gx = min(int(row["cx"] // self.cell_w), GX - 1)
            gy = min(int(row["cy"] // self.cell_h), GY - 1)

            gx = max(gx, 0)
            gy = max(gy, 0)

            density_grid[gy, gx] += 1
            velocity_sum[gy, gx] += row["velocity"]
            count_grid[gy, gx] += 1
            acc_sum[gy, gx] += row["acceleration"]
            track_risk_sum[gy, gx] += row["track_risk"]

            dx = row["dx"]
            dy = row["dy"]

            cell_dx[gy][gx].append(dx)
            cell_dy[gy][gx].append(dy)

            mag = np.sqrt(dx ** 2 + dy ** 2 + self.EPS)
            cell_dirs[gy][gx].append((dx / mag, dy / mag))

        avg_velocity = velocity_sum / (count_grid + self.EPS)
        avg_acc = acc_sum / (count_grid + self.EPS)
        avg_track_risk = track_risk_sum / (count_grid + self.EPS)

        congestion_grid = density_grid / (avg_velocity + 1e-5 + self.EPS)

        turbulence_grid = np.zeros((GY, GX))
        flow_conflict_grid = np.zeros((GY, GX))

        for y in range(GY):
            for x in range(GX):
                if len(cell_dx[y][x]) > 1:
                    vel_local = np.sqrt(
                        np.array(cell_dx[y][x]) ** 2 +
                        np.array(cell_dy[y][x]) ** 2
                    )
                    turbulence_grid[y, x] = np.var(vel_local)

                dirs = cell_dirs[y][x]

                if len(dirs) > 1:
                    d_arr = np.array(dirs)
                    mean_dir = np.mean(d_arr, axis=0)
                    coherence = np.linalg.norm(mean_dir) / (
                        np.linalg.norm(d_arr.sum(axis=0)) + self.EPS
                    )
                    flow_conflict_grid[y, x] = 1 - coherence

        rows = []

        for y in range(GY):
            for x in range(GX):
                density = density_grid[y, x]
                avg_v = avg_velocity[y, x]
                avg_a = avg_acc[y, x]
                congestion = congestion_grid[y, x]
                turbulence = turbulence_grid[y, x]
                flow_conflict = flow_conflict_grid[y, x]
                avg_tr = avg_track_risk[y, x]

                density_n = self.normalize(density, "density")
                dispersion_n = 1.0 - density_n
                congestion_n = self.normalize(congestion, "congestion")

                turbulence_n_raw = self.normalize(turbulence, "turbulence")
                self.turbulence_history[(x, y)].append(turbulence_n_raw)
                turbulence_n = float(np.mean(self.turbulence_history[(x, y)]))

                flow_conflict_n = self.normalize(flow_conflict, "flow_conflict")
                velocity_n = self.normalize(avg_v, "avg_velocity")
                acc_n = self.normalize(avg_a, "avg_acc")
                track_risk_n = self.normalize(avg_tr, "avg_track_risk")

                w = self.RISK_WEIGHTS

                risk = (
                    w["density"] * density_n +
                    w["dispersion"] * dispersion_n +
                    w["congestion"] * congestion_n +
                    w["turbulence"] * turbulence_n +
                    w["flow_conflict"] * flow_conflict_n +
                    w["velocity"] * velocity_n +
                    w["acceleration"] * acc_n +
                    w["track_risk"] * track_risk_n
                )

                prev = self.prev_risk.get((x, y))

                if prev is None:
                    risk_smooth = risk
                else:
                    risk_smooth = self.risk_alpha * risk + (1 - self.risk_alpha) * prev

                self.prev_risk[(x, y)] = risk_smooth

                rows.append({
                    "frame": frame_no,
                    "cell_x": x,
                    "cell_y": y,
                    "density": density,
                    "avg_velocity": avg_v,
                    "avg_acc": avg_a,
                    "congestion": congestion,
                    "turbulence": turbulence,
                    "flow_conflict": flow_conflict,
                    "avg_track_risk": avg_tr,
                    "density_n": density_n,
                    "dispersion_n": dispersion_n,
                    "congestion_n": congestion_n,
                    "turbulence_n": turbulence_n,
                    "flow_conflict_n": flow_conflict_n,
                    "velocity_n": velocity_n,
                    "acc_n": acc_n,
                    "track_risk_n": track_risk_n,
                    "risk": risk,
                    "risk_smooth": risk_smooth,
                    "manual_risk_level": self.get_manual_risk_level(risk_smooth),
                })

        df = pd.DataFrame(rows)

        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required ML features: {missing}")

        # ==========================================================
        # Run ML only on active crowd cells
        # (same logic used during training)
        # ==========================================================

        df["anomaly_probability"] = 0.0
        df["ml_anomaly"] = 0
        df["ml_risk_level"] = "Safe"

        active_mask = df["density"] > 0

        if active_mask.any():
            X_active = df.loc[active_mask, self.features]

            probs = self.model.predict_proba(X_active)[:, 1]

            df.loc[active_mask, "anomaly_probability"] = probs

            df.loc[active_mask, "ml_anomaly"] = (
                probs >= self.binary_threshold
            ).astype(int)

            df.loc[active_mask, "ml_risk_level"] = [
                self.get_ml_risk_level(p)
                for p in probs
            ]

        self._log_df(df)

        # ==========================================================
        # Frame-level status
        # ==========================================================

        if active_mask.any():
            frame_prob = float(
                df.loc[active_mask, "anomaly_probability"].max()
            )
        else:
            frame_prob = 0.0

        frame_level = self.get_ml_risk_level(frame_prob)

        danger_cells = int((df["ml_risk_level"] == "Danger").sum())
        warning_cells = int((df["ml_risk_level"] == "Warning").sum())

        return df, frame_prob, frame_level, warning_cells, danger_cells

    def _log_df(self, df):
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=df.columns.tolist()
            )
            self.writer.writeheader()

        for row in df.to_dict("records"):
            self.writer.writerow(row)

        self.csv_file.flush()