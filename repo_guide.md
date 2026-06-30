## Repository Guide

The repository is organized into separate modules for training and execution.

### Model Training

```
scripts/ml_v2.ipynb
```

This notebook contains the complete machine learning training pipeline, including data preprocessing, feature selection, model training, evaluation, and comparison of different classifiers.

### Real-Time Execution

```
demo.py
```

This is the main execution script for the real-time crowd risk detection system. It integrates the custom modules developed in this project to perform the complete pipeline:

```
Video Input → Detection → Tracking → Feature Extraction → ML Prediction → Visualization
```

### Batch Video Processing

```
launcher.py
```

This script processes videos through the complete pipeline and generates the corresponding output files (CSV logs, analytics, and annotated videos).

> **Note:** To test the project yourself, you will need your own sample videos containing crowds. Place the videos in the appropriate input directory (or update the input path in the script) before running `demo.py` or `launcher.py`.