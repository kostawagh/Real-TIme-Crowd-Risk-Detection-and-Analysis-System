# Real-Time Crowd Risk Detection and Analysis System

An AI-powered crowd monitoring system that uses **YOLOv8**, **multi-object tracking**, and **machine learning** to analyze crowd movement and classify crowd conditions into **Safe**, **Warning**, and **Danger** in real time.

## Features

- Real-time person detection using YOLOv8
- Multi-object tracking
- Grid-based spatiotemporal feature extraction
- ML-based crowd risk classification
- Live visualization with risk overlays
- Supports video files, webcam, and IP camera streams
- Streamlit dashboard for interactive monitoring
- CSV logging and annotated video output

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

---

## Installation

It is recommended to run this project inside a **Python virtual environment (venv)**.

### 1. Clone the repository

```bash
git clone <repository-link>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install ultralytics opencv-python numpy pandas scikit-learn streamlit matplotlib scipy joblib tqdm filterpy lap supervision
```

*(If your project includes additional libraries, install those as well or create a `requirements.txt`.)*

---

## Project Workflow

```text
Video Input
    ↓
YOLOv8 Detection
    ↓
Object Tracking
    ↓
Feature Extraction
    ↓
Machine Learning Prediction
    ↓
Safe / Warning / Danger
    ↓
Visualization & Logging
```

---

## Applications

- Smart City Surveillance
- Railway Stations
- Religious Gatherings
- Concerts & Festivals
- Sports Events
- Public Crowd Safety

---

## Future Scope

- Better performance in dense crowds
- Multi-camera support
- Edge device deployment
- Automatic alert system
- Temporal deep learning models

---

## Disclaimer

This project is an academic research prototype intended for educational and research purposes. It is designed to assist crowd monitoring and early warning, not replace professional crowd management systems.

---

## Author

**Kaustubh Wagh**
