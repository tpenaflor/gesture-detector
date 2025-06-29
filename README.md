# Gesture Recognition System

## Features

- Real-time hand detection using MediaPipe
- Train custom gestures with keypresses (1-4)
- Machine learning classification (SVM)
- Live feedback with confidence scores

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- scikit-learn
- NumPy

## Installation

1. Create conda environment:
   conda create -n gesture python=3.8
   conda activate gesture

2. Install packages:
   pip install opencv-python mediapipe scikit-learn numpy joblib

## Usage

Run the recognizer:
python gesture_recognizer.py

Controls:

- Press 1-4 to save current hand position as a gesture
- Press 'c' to toggle collection/recognition mode
- Press 'q' to quit

## Training Data

Collected samples are saved to gesture_data.csv
Retrain model manually by running:
python train_gesture_classifier.py

## File Structure

gesture_recognizer.py - Main application
train_gesture_classifier.py - Model training
data_collector.py - Data collection utilities
gesture_classifier.joblib - Saved model
scaler.joblib - Feature normalizer

## Troubleshooting

If you get camera errors:

- Try changing cv2.VideoCapture(0) to (1)
- Check camera permissions

For module errors:

- Verify all packages are installed
- Check Python version (3.8 recommended)
