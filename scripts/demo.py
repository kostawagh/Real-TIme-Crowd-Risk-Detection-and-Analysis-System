import os
import cv2
import torch
import numpy as np
from collections import deque

from modules.crowd_Detector_Yolo import CrowdDetector
from modules.crowd_Tracker import CrowdTracker
from modules.tracking_Visualizer import Visualizer
from modules.realtime_ml_engine import RealtimeMLEngine


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

VIDEO_PATH = "data/umn2.mp4"

OUTPUT_ROOT = "output"
YOLO_MODEL_PATH = "models/yolov8n_crowdhuman.pt"
ML_MODEL_PATH = "output/random_forest_anomaly_model.pkl"

GRID_X = 5
GRID_Y = 5

SAFE_THRESHOLD = 0.35
WARNING_THRESHOLD = 0.70

DISPLAY_W = 900
DISPLAY_H = 600

SAVE_OUTPUT_VIDEO = True

DETECT_CONFIG = {
    "conf_thres": 0.28,
    "iou_thres": 0.5,
    "img_size": 640,
    "detect_heads": False,
    "use_preprocess": True,
    "use_tiling": True,
    "use_bytetrack": False,
    "conf_low_thres": 0.10,
    "min_hits": 2,
    "max_age": 30,
    "iou_track": 0.12,
    "frame_skip": 1,
    "min_track_len": 10,
    "display_w": DISPLAY_W,
    "display_h": DISPLAY_H,
}


# ═══════════════════════════════════════════════════════════
# Live Graph Buffers
# ═══════════════════════════════════════════════════════════

prob_history = deque(maxlen=120)
count_history = deque(maxlen=120)
danger_history = deque(maxlen=120)


# ═══════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════

def get_status_color(level):
    if level == "Safe":
        return (0, 200, 0)
    elif level == "Warning":
        return (0, 220, 255)
    return (0, 0, 255)


def draw_grid_overlay(frame, frame_df, grid_x, grid_y):
    h, w = frame.shape[:2]
    cell_w = w / grid_x
    cell_h = h / grid_y

    overlay = frame.copy()

    for _, row in frame_df.iterrows():
        x = int(row["cell_x"])
        y = int(row["cell_y"])

        level = row["ml_risk_level"]
        prob = float(row["anomaly_probability"])
        color = get_status_color(level)

        x1 = int(x * cell_w)
        y1 = int(y * cell_h)
        x2 = int((x + 1) * cell_w)
        y2 = int((y + 1) * cell_h)

        # Fill only warning/danger cells
        if level in ["Warning", "Danger"]:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Grid outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 1)

        # Show probability only if meaningful
        if prob >= SAFE_THRESHOLD:
            cv2.putText(
                frame,
                f"{prob:.2f}",
                (x1 + 5, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    return frame


def draw_ml_status(
    frame,
    frame_no,
    frame_prob,
    frame_level,
    warning_cells,
    danger_cells,
):
    color = get_status_color(frame_level)
    h, w = frame.shape[:2]

    # Compact top-right HUD
    box_w = 320
    box_h = 95

    x1 = w - box_w - 15
    y1 = 15
    x2 = w - 15
    y2 = y1 + box_h

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(
        frame,
        f"Frame {frame_no}",
        (x1 + 15, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Status: {frame_level}",
        (x1 + 15, y1 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Prob: {frame_prob:.2f}",
        (x1 + 15, y1 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"W:{warning_cells}  D:{danger_cells}",
        (x1 + 165, y1 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_live_graph_panel(
    display,
    prob_history,
    count_history,
    danger_history,
    frame_level,
    warning_cells,
    danger_cells,
    panel_width=360,
):
    h, _ = display.shape[:2]

    panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

    cv2.putText(
        panel,
        "Realtime Analytics",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    def draw_line_graph(values, y_top, title, max_value, color):
        cv2.putText(
            panel,
            title,
            (20, y_top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        x0, y0 = 20, y_top
        graph_w, graph_h = panel_width - 40, 90

        cv2.rectangle(
            panel,
            (x0, y0),
            (x0 + graph_w, y0 + graph_h),
            (80, 80, 80),
            1,
        )

        if len(values) < 2:
            return

        vals = list(values)
        max_value = max(max_value, 1e-6)

        points = []
        for i, val in enumerate(vals):
            x = int(x0 + (i / (len(vals) - 1)) * graph_w)
            y = int(
                y0 + graph_h -
                (min(float(val), max_value) / max_value) * graph_h
            )
            points.append((x, y))

        for i in range(1, len(points)):
            cv2.line(panel, points[i - 1], points[i], color, 2)

        cv2.putText(
            panel,
            f"{vals[-1]:.2f}",
            (x0 + graph_w - 65, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    draw_line_graph(
        prob_history,
        80,
        "Anomaly Probability",
        1.0,
        (0, 0, 255),
    )

    draw_line_graph(
        count_history,
        210,
        "Tracked Count",
        max(1, max(count_history, default=1)),
        (0, 255, 0),
    )

    draw_line_graph(
        danger_history,
        340,
        "Danger Cells",
        max(1, max(danger_history, default=1)),
        (0, 220, 255),
    )

    # Live summary numbers below the third graph
    summary_y = 465

    cv2.putText(
        panel,
        "Live Summary",
        (20, summary_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    latest_prob = prob_history[-1] if len(prob_history) > 0 else 0.0
    latest_count = count_history[-1] if len(count_history) > 0 else 0

    status_color = get_status_color(frame_level)

    summary_lines = [
        ("Status", frame_level, status_color),
        ("Probability", f"{latest_prob:.2f}", (220, 220, 220)),
        ("Tracked Count", str(latest_count), (220, 220, 220)),
        ("Warning Cells", str(warning_cells), (0, 220, 255)),
        ("Danger Cells", str(danger_cells), (0, 0, 255)),
    ]

    y = summary_y + 35

    for label, value, color in summary_lines:
        cv2.putText(
            panel,
            f"{label:<14}: {value}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 24

    combined = np.hstack([display, panel])
    return combined


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    if isinstance(VIDEO_PATH, int):
        video_name = f"camera_{VIDEO_PATH}"
    else:
        video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        
    realtime_dir = os.path.join(OUTPUT_ROOT, "realtime_ml")
    os.makedirs(realtime_dir, exist_ok=True)

    output_csv = os.path.join(
        realtime_dir,
        f"realtime_ml_{video_name}.csv"
    )

    output_video_path = os.path.join(
        realtime_dir,
        f"realtime_ml_{video_name}.mp4"
    )

    config = {
        "video_path": VIDEO_PATH,
        "output_dir": os.path.join(OUTPUT_ROOT, "realtime_tracking"),
        "model_path": YOLO_MODEL_PATH,
        **DETECT_CONFIG,
    }

    print("GPU available:", torch.cuda.is_available())
    print("Video:", VIDEO_PATH)
    print("YOLO model:", YOLO_MODEL_PATH)
    print("ML model:", ML_MODEL_PATH)
    print("Output CSV:", output_csv)

    detector = CrowdDetector(config)
    tracker = CrowdTracker(config)
    visualizer = Visualizer(config)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    ml_engine = RealtimeMLEngine(
        model_path=ML_MODEL_PATH,
        output_csv=output_csv,
        frame_width=frame_width,
        frame_height=frame_height,
        grid_x=GRID_X,
        grid_y=GRID_Y,
        safe_threshold=SAFE_THRESHOLD,
        warning_threshold=WARNING_THRESHOLD,
    )

    writer = None

    cv2.namedWindow("Realtime Crowd Risk ML Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Realtime Crowd Risk ML Demo", DISPLAY_W + 360, DISPLAY_H)

    frame_no = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            if frame_no % config["frame_skip"] != 0:
                continue

            try:
                dets, inference_ms = detector.detect(frame)
            except Exception as e:
                print(f"[WARN] Frame {frame_no}: detection failed — {e}")
                continue

            tracks = tracker.update(dets)

            frame_df, frame_prob, frame_level, warning_cells, danger_cells = (
                ml_engine.process_frame(
                    frame_no=frame_no,
                    tracks=tracks,
                )
            )

            # Draw ML grid on original frame first
            annotated = frame.copy()
            annotated = draw_grid_overlay(
                annotated,
                frame_df,
                GRID_X,
                GRID_Y,
            )

            # Then draw regular tracking visualization
            display = visualizer.draw(
                annotated,
                tracks,
                frame_no,
                inference_ms,
            )

            # Draw compact ML HUD after visualizer resize
            display = draw_ml_status(
                display,
                frame_no,
                frame_prob,
                frame_level,
                warning_cells,
                danger_cells,
            )

            # Update graph histories
            prob_history.append(frame_prob)
            count_history.append(len(tracks))
            danger_history.append(danger_cells)

            # Add right-side live analytics graph panel
            display = draw_live_graph_panel(
                display,
                prob_history,
                count_history,
                danger_history,
                frame_level,
                warning_cells,
                danger_cells,
            )

            if SAVE_OUTPUT_VIDEO:
                if writer is None:
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        output_video_path,
                        fourcc,
                        fps,
                        (w, h),
                    )

                writer.write(display)

            cv2.imshow("Realtime Crowd Risk ML Demo", display)

            print(
                f"Frame {frame_no:04d} | "
                f"Det: {len(dets):3d} | "
                f"Tracked: {len(tracks):3d} | "
                f"ML Prob: {frame_prob:.2f} | "
                f"Level: {frame_level} | "
                f"W: {warning_cells} | D: {danger_cells}"
            )

            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC pressed — stopping.")
                break

    finally:
        ml_engine.close()
        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    print("\nRealtime demo complete.")
    print(f"CSV saved   : {output_csv}")

    if SAVE_OUTPUT_VIDEO:
        print(f"Video saved : {output_video_path}")


if __name__ == "__main__":
    main()
