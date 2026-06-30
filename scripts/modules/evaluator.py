"""
evaluator.py
Frame-level evaluation helper for crowd anomaly detection.
"""

import os
import pandas as pd


class CrowdAnomalyEvaluator:
    def __init__(
        self,
        ground_truth_csv: str,
        risk_dir: str = "output/risk",
        risk_threshold: float = 0.7,
        min_event_length: int = 3,
    ):
        self.ground_truth_csv = ground_truth_csv
        self.risk_dir = risk_dir
        self.risk_threshold = risk_threshold
        self.min_event_length = min_event_length
        self.gt_df = pd.read_csv(ground_truth_csv)

    def _load_risk_csv(self, video_name: str) -> pd.DataFrame:
        risk_path = os.path.join(self.risk_dir, f"risk_{video_name}.csv")
        if not os.path.exists(risk_path):
            raise FileNotFoundError(f"Risk CSV not found: {risk_path}")
        return pd.read_csv(risk_path)

    def _get_ground_truth_range(self, video_name: str) -> tuple[int, int]:
        row = self.gt_df[self.gt_df["video_name"] == video_name]
        if row.empty:
            raise ValueError(f"No ground truth found for video: {video_name}")
        return int(row.iloc[0]["anomaly_start"]), int(row.iloc[0]["anomaly_end"])

    def extract_predicted_anomalies(self, video_name: str) -> list[tuple[int, int]]:
        df = self._load_risk_csv(video_name)
        risk_col = "risk_smooth" if "risk_smooth" in df.columns else "risk"

        frame_risk = (
            df.groupby("frame")[risk_col]
            .max()
            .reset_index()
            .sort_values("frame")
        )

        anomalies = []
        start = None
        end = None
        active_count = 0
        required_consecutive = 10

        for _, row in frame_risk.iterrows():
            frame = int(row["frame"])
            risk = float(row[risk_col])

            if risk >= self.risk_threshold:
                active_count += 1

                if active_count >= required_consecutive:
                    if start is None:
                        start = frame - (required_consecutive - 1)
                    end = frame

            else:
                active_count = 0

                if start is not None:
                    if (end - start) >= self.min_event_length:
                        anomalies.append((start, end))

                    start = None
                    end = None

        if start is not None and (end - start) >= self.min_event_length:
            anomalies.append((start, end))

        return anomalies

    def evaluate_video(self, video_name: str) -> dict:
        gt_start, gt_end = self._get_ground_truth_range(video_name)

        gt_frames = set()
        if not (gt_start == -1 and gt_end == -1):
            gt_frames = set(range(gt_start, gt_end + 1))

        pred_segments = self.extract_predicted_anomalies(video_name)
        predicted_ranges = (
            "; ".join([f"{s}-{e}" for s, e in pred_segments])
            if pred_segments else "None"
        )

        pred_frames = set()
        for s, e in pred_segments:
            pred_frames.update(range(s, e + 1))

        tp = len(gt_frames & pred_frames)
        fp = len(pred_frames - gt_frames)
        fn = len(gt_frames - pred_frames)

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)

        detection_delay = None
        if gt_frames and pred_segments:
            first_pred = pred_segments[0][0]
            detection_delay = first_pred - gt_start

        return {
            "video_name": video_name,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "predicted_ranges": predicted_ranges,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "detection_delay": detection_delay,
        }

    def evaluate_all(self) -> pd.DataFrame:
        results = []

        for video_name in self.gt_df["video_name"].tolist():
            try:
                results.append(self.evaluate_video(video_name))
            except FileNotFoundError:
                print(f"[WARN] Missing risk CSV for {video_name}")

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(results_df.to_string(index=False))

        return results_df

    def inspect_video(self, video_name: str) -> None:
        gt_start, gt_end = self._get_ground_truth_range(video_name)
        pred = self.extract_predicted_anomalies(video_name)

        print("\n" + "-" * 50)
        print(f"Video: {video_name}")
        print("-" * 50)

        if gt_start == -1 and gt_end == -1:
            print("Ground Truth: NORMAL video")
        else:
            print(f"Ground Truth: {gt_start} → {gt_end}")

        if pred:
            print("Predicted Anomalies:")
            for s, e in pred:
                print(f"  {s} → {e}")
        else:
            print("Predicted Anomalies: None")

    def inspect_all(self) -> None:
        for video_name in self.gt_df["video_name"].tolist():
            self.inspect_video(video_name)


    def plot_debug_dashboard(self, video_name: str, output_dir: str = "output/evaluation_plots") -> str:
        """
        Creates a 4-subplot debug dashboard:
        1. Risk timeline with GT + predicted anomaly regions
        2. Velocity + acceleration
        3. Congestion + turbulence
        4. Track risk + flow conflict + density
        """

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)

        # Load risk CSV
        df = self._load_risk_csv(video_name)

        risk_col = "risk_smooth" if "risk_smooth" in df.columns else "risk"

        # Ground truth + prediction
        gt_start, gt_end = self._get_ground_truth_range(video_name)
        pred_segments = self.extract_predicted_anomalies(video_name)

        # Frame-level aggregation
        frame_df = (
            df.groupby("frame")
            .agg({
                "risk": "max",
                "risk_smooth": "max",
                "avg_velocity": "mean",
                "avg_acc": "mean",
                "congestion": "mean",
                "turbulence": "mean",
                "avg_track_risk": "mean",
                "flow_conflict": "mean",
                "density": "mean",
            })
            .reset_index()
            .sort_values("frame")
        )

        frames = frame_df["frame"]

        fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

        # ─────────────────────────────
        # 1. Risk timeline
        # ─────────────────────────────
        axes[0].plot(frames, frame_df["risk"], label="Raw Risk", alpha=0.45)
        axes[0].plot(frames, frame_df[risk_col], label="Smoothed Risk", linewidth=2)

        axes[0].axhline(self.risk_threshold, linestyle="--", label="Threshold")

        # Ground Truth → Green
        if gt_start != -1 and gt_end != -1:
            axes[0].axvspan(
                gt_start,
                gt_end,
                color="limegreen",
                alpha=0.30,
                label="Ground Truth"
            )

        # Predicted → Red
        for i, (s, e) in enumerate(pred_segments):
            axes[0].axvspan(
                s,
                e,
                color="red",
                alpha=0.22,
                label="Predicted" if i == 0 else None
            )

        axes[0].set_title(f"Risk Timeline — {video_name}")
        axes[0].set_ylabel("Risk")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ─────────────────────────────
        # 2. Velocity + acceleration
        # ─────────────────────────────
        if "avg_velocity" in frame_df.columns:
            axes[1].plot(frames, frame_df["avg_velocity"], label="Avg Velocity")

        if "avg_acc" in frame_df.columns:
            axes[1].plot(frames, frame_df["avg_acc"], label="Avg Acceleration")

        axes[1].set_title("Motion Metrics")
        axes[1].set_ylabel("Value")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ─────────────────────────────
        # 3. Congestion + turbulence
        # ─────────────────────────────
        if "congestion" in frame_df.columns:
            axes[2].plot(frames, frame_df["congestion"], label="Congestion")

        if "turbulence" in frame_df.columns:
            axes[2].plot(frames, frame_df["turbulence"], label="Turbulence")

        axes[2].set_title("Crowd Instability Metrics")
        axes[2].set_ylabel("Value")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # ─────────────────────────────
        # 4. Track risk + flow conflict + density
        # ─────────────────────────────
        if "avg_track_risk" in frame_df.columns:
            axes[3].plot(frames, frame_df["avg_track_risk"], label="Avg Track Risk")

        if "flow_conflict" in frame_df.columns:
            axes[3].plot(frames, frame_df["flow_conflict"], label="Flow Conflict")

        if "density" in frame_df.columns:
            axes[3].plot(frames, frame_df["density"], label="Density")

        axes[3].set_title("Behavioral / Spatial Metrics")
        axes[3].set_xlabel("Frame")
        axes[3].set_ylabel("Value")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{video_name}_debug.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[DebugPlot] Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    #==============================
    VIDEO_NAME = "umn4"
    #==============================
    evaluator = CrowdAnomalyEvaluator(
        ground_truth_csv="data/ground_truth.csv",
        risk_dir="output/risk",
        risk_threshold=0.268,
        min_event_length=3,
    )
    print("\n" + "=" * 60)
    print("Risk Smooth Ranges")
    print("=" * 60)

    for video_name in evaluator.gt_df["video_name"].tolist():
        try:
            df = evaluator._load_risk_csv(video_name)

            print(
                video_name,
                "min:", round(df["risk_smooth"].min(), 3),
                "mean:", round(df["risk_smooth"].mean(), 3),
                "max:", round(df["risk_smooth"].max(), 3),
            )

        except FileNotFoundError:
            print(f"[WARN] Missing risk CSV for {video_name}")

'''
    evaluator.inspect_video(VIDEO_NAME)
    metrics = evaluator.evaluate_video(VIDEO_NAME)

    print("\n" + "=" * 50)
    print("Evaluation Metrics")
    print("=" * 50)
    print(f"Video            : {metrics['video_name']}")
    print(f"Ground Truth     : {metrics['gt_start']} → {metrics['gt_end']}")
    print(f"Predicted Ranges : {metrics['predicted_ranges']}")
    print(f"True Positives   : {metrics['tp']}")
    print(f"False Positives  : {metrics['fp']}")
    print(f"False Negatives  : {metrics['fn']}")

    print(f"\nPrecision        : {metrics['precision']:.3f}")
    print(f"Recall           : {metrics['recall']:.3f}")
    print(f"F1-score         : {metrics['f1_score']:.3f}")

    delay = metrics["detection_delay"]

    if delay is None:
        print("Detection Delay  : N/A")
    elif delay < 0:
        print(f"Detection Delay  : {abs(delay)} frames EARLY")
    elif delay > 0:
        print(f"Detection Delay  : {delay} frames LATE")
    else:
        print("Detection Delay  : Perfect")

    evaluator.plot_debug_dashboard(VIDEO_NAME)
'''

results = evaluator.evaluate_all()
results.to_csv("output/evaluation_report.csv", index=False)
print("\nSaved evaluation report: output/evaluation_report.csv")


for video_name in evaluator.gt_df["video_name"].tolist():
    try:
        evaluator.plot_debug_dashboard(video_name)
    except FileNotFoundError:
        print(f"[WARN] Missing risk CSV for {video_name}")
