# Hand-Drawn Object Recognition with Real-Time Hand Tracking

This project allows users to draw shapes or objects in the air using their hands, and the system predicts what was drawn using a trained Convolutional Neural Network (CNN).  
It is inspired by Google's QuickDraw project, but users draw using their hand gestures detected via webcam.

---

## Project Features
- Real-time hand tracking using MediaPipe and OpenCV.
- Virtual drawing canvas controlled by finger movements.
- CNN model trained on 6 object categories: `bird`, `cat`, `circle`, `house`, `square`, and `triangle`.
- Smart UI with brush, eraser, and color selection options.
- Live prediction and score display after user drawing.

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- Scikit-image (for data augmentation)

---

## Project Structure
- `virtual_painter.py` — Main real-time drawing and prediction app.
- `main.py` — Training script for CNN.
- `hand_tracking_module.py` — Utility class for detecting hand landmarks.
- `create_model.py` — CNN model architecture.
- `train_model.py` — Data preparation and training utilities.
- `load_data.py` — Loading and augmenting QuickDraw subset data.
- `Header/` — Images for color and eraser selection buttons.
- `best_model.h5` — Saved best CNN model after training.

---

## Setup Instructions

1. Clone the repository and navigate to the `05-Hand-Drawn-Object-Recognition` folder.

2. Install required Python packages:
   ```bash
   pip install tensorflow opencv-python mediapipe scikit-image matplotlib
