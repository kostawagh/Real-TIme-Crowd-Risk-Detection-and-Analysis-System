# Real-Time Crowd Risk Detection and Analysis System

An AI-powered computer vision system for **real-time crowd risk assessment** using person detection, multi-object tracking, spatiotemporal feature extraction, and machine learning.

The system analyzes crowd movement from video feeds and classifies crowd conditions into **Safe**, **Warning**, or **Danger**, providing an interpretable early-warning solution for crowd monitoring applications.

---

## Features

- Real-time person detection using **YOLOv8**
- Multi-object tracking for trajectory analysis
- Grid-based spatiotemporal crowd feature extraction
- Machine learning-based crowd risk classification
- Live visualization with risk overlays
- Support for:
  - Video files
  - Laptop webcam
  - IP/Mobile camera streams
- Streamlit dashboard for interactive deployment
- Automatic CSV logging and annotated video generation

---

## System Pipeline

```text
Video Input
      │
      ▼
YOLOv8 Person Detection
      │
      ▼
Multi-Object Tracking
      │
      ▼
Spatiotemporal Feature Extraction
      │
      ▼
Machine Learning Classification
      │
      ▼
Safe / Warning / Danger Prediction
      │
      ▼
Visualization + CSV Logging + Output Video
```

---

## Crowd Features Extracted

The system analyzes crowd behaviour using engineered features rather than relying solely on crowd density.

- Crowd Density
- Average Velocity
- Average Acceleration
- Congestion Index
- Turbulence
- Flow Conflict
- Track Risk Score

These features are computed over a **5×5 spatial grid**, enabling localized risk assessment across the scene.

---

## Machine Learning

Several machine learning models were evaluated for crowd risk classification, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting

After comparison, **Random Forest** was selected for deployment due to its balance of accuracy, recall, robustness, and interpretability.

### Final Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **92.68%** |
| Precision | **75.84%** |
| Recall | **89.46%** |
| F1 Score | **82.09%** |

---

## Dashboard

The Streamlit dashboard supports:

- Video upload
- Live webcam analysis
- IP camera streaming
- Real-time risk visualization
- Live analytics
- CSV download
- Annotated output video download

---

## Output Files

The system generates multiple outputs during execution.

```
tracking_<video>.csv
features_<video>.csv
analytics_<video>.csv
risk_<video>.csv
realtime_ml_<video>.csv
realtime_ml_<video>.mp4
```

These outputs provide both numerical analytics and visual evidence of the detected crowd behaviour.

---

## Applications

- Smart City Surveillance
- Railway Stations
- Religious Gatherings
- Sports Events
- Concerts
- Festivals
- Public Rallies
- Crowd Safety Monitoring

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

## Project Structure

```text
├── models/
├── datasets/
├── realtime/
├── dashboard/
├── outputs/
├── notebooks/
├── utils/
├── requirements.txt
└── README.md
```

*(Folder names may differ depending on your local project structure.)*

---

## Future Improvements

- Improved performance under heavy occlusion
- Temporal deep learning models (LSTM/Transformer)
- Edge device deployment
- Multi-camera crowd fusion
- Automatic alert and notification system
- Geo-spatial crowd mapping

---

## Motivation

Large public gatherings such as festivals, railway stations, sporting events, and religious congregations remain vulnerable to crowd congestion and stampedes. Traditional CCTV systems rely heavily on manual observation, making early detection difficult.

This project aims to transform ordinary surveillance footage into **real-time crowd risk intelligence**, assisting authorities with early warning and improved situational awareness.

---

## Disclaimer

This project is an academic research prototype developed for educational and research purposes. It is intended to assist crowd monitoring and should not be considered a replacement for professional crowd management or emergency response systems.

---

## Author

**Kaustubh Wagh**

If you found this project interesting, consider giving it a ⭐ on GitHub.
