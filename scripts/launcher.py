# .\venv\Scripts\activate
# python scripts\launcher.py

"""
launcher.py

GUI launcher for the realtime crowd risk ML demo.

Options:
1. Browse and select a video file
2. Select a connected camera feed
3. Use IP camera stream
"""

import os
import cv2
import tkinter as tk

from tkinter import filedialog, messagebox
from tkinter import ttk

import demo


# ═══════════════════════════════════════════════════════════
# Demo Base Config
# ═══════════════════════════════════════════════════════════

demo.OUTPUT_ROOT = "output"

demo.YOLO_MODEL_PATH = "models/yolov8n_crowdhuman.pt"
demo.ML_MODEL_PATH = "output/random_forest_anomaly_model.pkl"

demo.GRID_X = 5
demo.GRID_Y = 5

demo.SAFE_THRESHOLD = 0.35
demo.WARNING_THRESHOLD = 0.70

demo.DISPLAY_W = 900
demo.DISPLAY_H = 600

demo.SAVE_OUTPUT_VIDEO = True


demo.DETECT_CONFIG.update({
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
    "display_w": demo.DISPLAY_W,
    "display_h": demo.DISPLAY_H,
})


# ═══════════════════════════════════════════════════════════
# Camera Detection
# ═══════════════════════════════════════════════════════════

def get_available_cameras(max_cameras=10):
    available = []

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap is not None and cap.isOpened():
            ret, _ = cap.read()

            if ret:
                available.append(index)

        cap.release()

    return available


# ═══════════════════════════════════════════════════════════
# GUI Launcher
# ═══════════════════════════════════════════════════════════

class DemoLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Crowd Risk Detection Demo Launcher")
        self.root.geometry("520x300")
        self.root.resizable(False, False)

        self.selected_video_path = None
        self.available_cameras = get_available_cameras()

        self.mode_var = tk.StringVar(value="Browse Video")
        self.camera_var = tk.StringVar()

        # Default IP Webcam stream URL
        self.ip_url_var = tk.StringVar(value="http://192.168.1.4:8080/video")

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Realtime Crowd Risk Demo",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=15)

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=8)

        tk.Label(
            mode_frame,
            text="Select Input Source:",
            font=("Arial", 11)
        ).grid(row=0, column=0, padx=8)

        mode_dropdown = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=["Browse Video", "Select Camera", "IP Camera Stream"],
            state="readonly",
            width=22
        )
        mode_dropdown.grid(row=0, column=1, padx=8)
        mode_dropdown.bind("<<ComboboxSelected>>", self.update_mode_ui)

        # Video browse frame
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(pady=10)

        self.browse_button = tk.Button(
            self.video_frame,
            text="Browse Video",
            width=18,
            command=self.browse_video
        )
        self.browse_button.grid(row=0, column=0, padx=8)

        self.video_label = tk.Label(
            self.video_frame,
            text="No video selected",
            width=35,
            anchor="w"
        )
        self.video_label.grid(row=0, column=1, padx=8)

        # Camera frame
        self.camera_frame = tk.Frame(self.root)

        camera_options = [
            f"Camera {idx}" for idx in self.available_cameras
        ]

        if camera_options:
            self.camera_var.set(camera_options[0])
        else:
            self.camera_var.set("No camera found")

        tk.Label(
            self.camera_frame,
            text="Connected Cameras:",
            font=("Arial", 11)
        ).grid(row=0, column=0, padx=8)

        self.camera_dropdown = ttk.Combobox(
            self.camera_frame,
            textvariable=self.camera_var,
            values=camera_options if camera_options else ["No camera found"],
            state="readonly",
            width=25
        )
        self.camera_dropdown.grid(row=0, column=1, padx=8)

        # IP stream frame
        self.ip_frame = tk.Frame(self.root)

        tk.Label(
            self.ip_frame,
            text="IP Stream URL:",
            font=("Arial", 11)
        ).grid(row=0, column=0, padx=8)

        self.ip_entry = tk.Entry(
            self.ip_frame,
            textvariable=self.ip_url_var,
            width=38
        )
        self.ip_entry.grid(row=0, column=1, padx=8)

        self.test_ip_button = tk.Button(
            self.ip_frame,
            text="Test",
            width=8,
            command=self.test_ip_stream
        )
        self.test_ip_button.grid(row=1, column=1, sticky="e", pady=8)

        # Run button
        self.run_button = tk.Button(
            self.root,
            text="Run Demo",
            width=20,
            height=2,
            bg="#1f7a1f",
            fg="white",
            font=("Arial", 11, "bold"),
            command=self.run_demo
        )
        self.run_button.pack(pady=20)

        self.update_mode_ui()

    def update_mode_ui(self, event=None):
        mode = self.mode_var.get()

        self.video_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.ip_frame.pack_forget()

        if mode == "Browse Video":
            self.video_frame.pack(pady=10)

        elif mode == "Select Camera":
            self.camera_frame.pack(pady=10)

        elif mode == "IP Camera Stream":
            self.ip_frame.pack(pady=10)

    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self.selected_video_path = file_path
            self.video_label.config(
                text=os.path.basename(file_path)
            )

    def test_ip_stream(self):
        url = self.ip_url_var.get().strip()

        if not url:
            messagebox.showerror(
                "Missing URL",
                "Please enter an IP camera stream URL."
            )
            return

        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            cap.release()
            messagebox.showerror(
                "Stream Failed",
                f"Could not open stream:\n{url}"
            )
            return

        ret, _ = cap.read()
        cap.release()

        if ret:
            messagebox.showinfo(
                "Stream Working",
                "IP camera stream opened successfully."
            )
        else:
            messagebox.showerror(
                "Read Failed",
                "Stream opened, but frame could not be read."
            )

    def run_demo(self):
        mode = self.mode_var.get()

        if mode == "Browse Video":
            if not self.selected_video_path:
                messagebox.showerror(
                    "No Video Selected",
                    "Please browse and select a video file first."
                )
                return

            demo.VIDEO_PATH = self.selected_video_path

        elif mode == "Select Camera":
            if not self.available_cameras:
                messagebox.showerror(
                    "No Camera Found",
                    "No connected camera was detected."
                )
                return

            selected_text = self.camera_var.get()

            try:
                camera_index = int(selected_text.split()[-1])
            except Exception:
                messagebox.showerror(
                    "Invalid Camera",
                    "Please select a valid camera."
                )
                return

            demo.VIDEO_PATH = camera_index

        elif mode == "IP Camera Stream":
            url = self.ip_url_var.get().strip()

            if not url:
                messagebox.showerror(
                    "Missing URL",
                    "Please enter an IP camera stream URL."
                )
                return

            demo.VIDEO_PATH = url

        self.root.destroy()
        demo.main()


# ═══════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app = DemoLauncher(root)
    root.mainloop()