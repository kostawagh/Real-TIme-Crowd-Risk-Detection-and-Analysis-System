"""
crowd_Visualization.py
=================
Unified module for crowd anomaly detection and visualization.

Usage from another script:
    from crowd_analysis import AnomalyDetector, CrowdPlotter, RiskVisualizer

    detector = AnomalyDetector("output/risk_umn1.csv")
    anomalies = detector.detect()
    detector.plot(save=True)

    plotter = CrowdPlotter()
    plotter.plot_risk("output/risk_umn1.csv")
    plotter.plot_velocity("output/features/features_umn1.csv")

    viz = RiskVisualizer("data/umn/umn1.mp4", "output/risk_umn1.csv")
    viz.run()
"""

import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import cv2


# ──────────────────────────────────────────────
# 1. ANOMALY DETECTOR  (from anomalyTimeline.py)
# ──────────────────────────────────────────────

class AnomalyDetector:
    """
    Detects and visualizes anomaly events from a per-frame risk CSV.

    Parameters
    ----------
    risk_csv : str
        Path to the CSV file with at least 'frame' and 'risk' columns.
    risk_threshold : float
        Normalized risk value above which a frame is considered anomalous.
    min_event_length : int
        Minimum number of consecutive anomalous frames to count as an event.
    """

    def __init__(
        self,
        risk_csv: str,
        risk_threshold: float = 0.65,
        min_event_length: int = 3,
    ):
        self.risk_csv = risk_csv
        self.risk_threshold = risk_threshold
        self.min_event_length = min_event_length

        self._frames: np.ndarray | None = None
        self._risk_values: np.ndarray | None = None
        self._anomalies: list[tuple[int, int]] | None = None

    # ── internal helpers ──────────────────────

    def _load(self):
        """Load and normalize frame-level risk data (lazy)."""
        if self._frames is not None:
            return

        df = pd.read_csv(self.risk_csv)
        risk_col = "risk_smooth" if "risk_smooth" in df.columns else "risk"

        frame_risk = df.groupby("frame")[risk_col].max().reset_index()

        self._frames = frame_risk["frame"].values
        self._risk_values = frame_risk[risk_col].values
        

    # ── public API ────────────────────────────

    def detect(self) -> list[tuple[int, int]]:
        """
        Run anomaly detection and return a list of (start_frame, end_frame) tuples.
        Results are cached; call detect() again after changing threshold/min_length.
        """
        self._load()

        anomalies = []
        start = None

        for frame, risk in zip(self._frames, self._risk_values):
            if risk >= self.risk_threshold:
                if start is None:
                    start = frame
                end = frame
            else:
                if start is not None:
                    if (end - start) >= self.min_event_length:
                        anomalies.append((start, end))
                    start = None

        if start is not None:
            anomalies.append((start, end))

        self._anomalies = anomalies
        return anomalies

    def plot(
        self,
        output_dir: str = "output/plots",
        filename: str | None = None,
        save: bool = True,
        show: bool = True,
    ) -> str | None:
        """
        Plot the risk timeline with anomaly regions highlighted.

        Parameters
        ----------
        output_dir : str
            Directory to save the plot.
        filename : str | None
            Output filename. Defaults to 'anomaly_timeline_<video>.png'.
        save : bool
            Whether to save the figure to disk.
        show : bool
            Whether to call plt.show().

        Returns
        -------
        str | None
            Path to the saved plot, or None if save=False.
        """
        self._load()
        if self._anomalies is None:
            self.detect()

        plt.figure(figsize=(14, 6))
        plt.plot(self._frames, self._risk_values, label="Risk Score", color="blue", linewidth=2)
        plt.axhline(self.risk_threshold, color="black", linestyle="--", label="Threshold")

        for start, end in self._anomalies:
            plt.axvspan(start, end, color="red", alpha=0.3)

        plt.title("Crowd Anomaly Timeline (Risk Over Time)")
        plt.xlabel("Frame")
        plt.ylabel("Normalized Risk")
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = None
        if save:
            video_name = os.path.splitext(os.path.basename(self.risk_csv))[0].replace("risk_", "")
            fname = filename or f"anomaly_timeline_{video_name}.png"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, fname)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {output_path}")

        if show:
            print("show=True ignored (Agg backend active)")

        plt.close()
        return output_path

    def summary(self) -> None:
        """Print a human-readable summary of detected anomaly events."""
        if self._anomalies is None:
            self.detect()

        print("\n🔥 Anomaly Events:")
        if not self._anomalies:
            print("  No anomalies detected.")
            return
        for i, (s, e) in enumerate(self._anomalies, 1):
            print(f"  {i}. Start: {s}, End: {e}, Duration: {e - s} frames")


# ──────────────────────────────────────────────
# 2. CROWD PLOTTER  (from plots.py)
# ──────────────────────────────────────────────

class CrowdPlotter:
    """
    Generates time-series plots for crowd risk and velocity metrics.

    Parameters
    ----------
    output_dir : str
        Default directory for saving plots.
    """

    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = output_dir

    # ── internal helper ───────────────────────

    @staticmethod
    def _video_name(csv_path: str, prefix: str) -> str:
        return os.path.splitext(os.path.basename(csv_path))[0].replace(prefix, "")

    def plot_risk(
        self,
        risk_csv: str,
        output_dir: str | None = None,
        save: bool = True,
        show: bool = True,
    ) -> str | None:
        """
        Plot average crowd risk score over time.

        Parameters
        ----------
        risk_csv : str
            Path to risk CSV.
        output_dir : str | None
            Override instance output_dir for this call.
        save : bool
            Save plot to disk.
        show : bool
            Display plot interactively.

        Returns
        -------
        str | None
            Saved file path, or None.
        """

        out_dir = output_dir or self.output_dir

        df = pd.read_csv(risk_csv)
        video_name = self._video_name(risk_csv, "risk_")

        # ── Aggregate per frame ─────────────────────────────
        frame_risk = df.groupby("frame")[["risk", "risk_smooth"]].mean().reset_index()

        # ── Plot ────────────────────────────────────────────
        plt.figure(figsize=(12, 6))

        # Raw risk
        plt.plot(
            frame_risk["frame"],
            frame_risk["risk"],
            linewidth=1.5,
            alpha=0.4,
            label="Raw Risk"
        )

        # Smoothed risk
        plt.plot(
            frame_risk["frame"],
            frame_risk["risk_smooth"],
            linewidth=2.5,
            label="Smoothed Risk"
        )

        plt.title(f"Crowd Risk Over Time - {video_name}", fontsize=14)
        plt.xlabel("Frame Number")
        plt.ylabel("Average Risk Score")

        frames = frame_risk["frame"].values
        step = max(1, len(frames) // 10)
        plt.xticks(frames[::step])

        plt.grid(True, alpha=0.4)
        plt.legend()

        # ── Save ────────────────────────────────────────────
        output_path = None

        if save:
            os.makedirs(out_dir, exist_ok=True)

            output_path = os.path.join(
                out_dir,
                f"risk_{video_name}.png"
            )

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {output_path}")

        if show:
            print("show=True ignored (Agg backend active)")

        plt.close()

        return output_path

    def plot_velocity(
        self,
        features_csv: str,
        output_dir: str | None = None,
        save: bool = True,
        show: bool = True,
    ) -> str | None:
        """
        Plot average crowd velocity over time.

        Parameters
        ----------
        features_csv : str
            Path to features CSV (columns: frame, velocity).
        output_dir : str | None
            Override instance output_dir for this call.
        save : bool
            Save plot to disk.
        show : bool
            Display plot interactively.

        Returns
        -------
        str | None
            Saved file path, or None.
        """
        out_dir = output_dir or self.output_dir
        df = pd.read_csv(features_csv)
        video_name = self._video_name(features_csv, "features_")
        frame_velocity = df.groupby("frame")["velocity"].mean().reset_index()

        plt.figure(figsize=(12, 6))
        plt.plot(frame_velocity["frame"], frame_velocity["velocity"], linewidth=2)
        plt.title(f"Crowd Velocity Over Time - {video_name}", fontsize=14)
        plt.xlabel("Frame Number")
        plt.ylabel("Average Velocity")

        frames = frame_velocity["frame"].values
        step = max(1, len(frames) // 10)
        plt.xticks(frames[::step])
        plt.grid(True, alpha=0.4)

        output_path = None
        if save:
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"velocity_{video_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {output_path}")

        if show:
            print("show=True ignored (Agg backend active)")

        plt.close()
        return output_path


# ──────────────────────────────────────────────
# 3. RISK VISUALIZER  (from visualise.py)
# ──────────────────────────────────────────────

class RiskVisualizer:
    """
    Overlays a per-cell risk heatmap on a video and shows a side-by-side
    comparison of the original frame vs. the heatmap overlay.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    risk_csv : str
        Path to the risk CSV (columns: frame, cell_x, cell_y, risk).
    grid_x : int
        Number of horizontal grid cells.
    grid_y : int
        Number of vertical grid cells.
    alpha : float
        Blend weight for the heatmap overlay (0 = invisible, 1 = opaque).
    """

    def __init__(
        self,
        video_path: str,
        risk_csv: str,
        grid_x: int = 5,
        grid_y: int = 5,
        alpha: float = 0.5,
    ):
        self.video_path = video_path
        self.risk_csv = risk_csv
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.alpha = alpha

    def run(self, window_size: tuple[int, int] = (1200, 600)) -> None:
        """
        Open a live OpenCV window showing original vs. risk heatmap.
        Press ESC to quit.

        Parameters
        ----------
        window_size : tuple[int, int]
            Display window width and height in pixels.
        """
        df = pd.read_csv(self.risk_csv)
        risk_col = "risk_smooth" if "risk_smooth" in df.columns else "risk"
        frame_groups = {f: g for f, g in df.groupby("frame")}
        frame_max_risk = df.groupby("frame")[risk_col].max().to_dict()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cell_w = frame_w / self.grid_x
        cell_h = frame_h / self.grid_y

        cv2.namedWindow("Original vs Risk Heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original vs Risk Heatmap", *window_size)

        frame_no = 0
        last_valid = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_no += 1
                frame_df = frame_groups.get(frame_no)

                if frame_df is None:
                    frame_df = last_valid
                else:
                    last_valid = frame_df

                if frame_df is None:
                    continue

                # Build heatmap
                heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
                for _, row in frame_df.iterrows():
                    gx, gy, risk = int(row["cell_x"]), int(row["cell_y"]), row[risk_col]
                    x1, y1 = int(gx * cell_w), int(gy * cell_h)
                    x2, y2 = int((gx + 1) * cell_w), int((gy + 1) * cell_h)
                    heatmap[y1:y2, x1:x2] = risk

                if heatmap.max() > 0:
                    heatmap /= heatmap.max()

                heatmap_color = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                overlay = cv2.addWeighted(frame, 1 - self.alpha, heatmap_color, self.alpha, 0)

                # Risk text
                max_risk = frame_max_risk.get(frame_no, 0)
                cv2.putText(overlay, f"Max Risk: {max_risk:.2f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Grid lines
                for x in range(1, self.grid_x):
                    cv2.line(overlay, (int(x * cell_w), 0), (int(x * cell_w), frame_h), (255, 255, 255), 1)
                for y in range(1, self.grid_y):
                    cv2.line(overlay, (0, int(y * cell_h)), (frame_w, int(y * cell_h)), (255, 255, 255), 1)

                # Labels
                orig_display = frame.copy()
                heat_display = overlay.copy()
                cv2.putText(orig_display, "ORIGINAL",
                            (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(heat_display, "RISK HEATMAP",
                            (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                combined = np.hstack((
                    cv2.resize(orig_display, (frame_w, frame_h)),
                    cv2.resize(heat_display, (frame_w, frame_h)),
                ))
                cv2.imshow("Original vs Risk Heatmap", combined)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
