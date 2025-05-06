# Hand-Drawn Object Recognition with Real-Time Hand Tracking

This project allows users to draw shapes or objects in the air using their hands, and the system predicts what was drawn using a trained Convolutional Neural Network (CNN).
It is inspired by Google's QuickDraw project, but users draw using hand gestures detected via webcam.

---

## Project Features

- Real-time hand tracking using MediaPipe and OpenCV.
- Virtual drawing canvas controlled by finger movements.
- CNN model trained on 6 object categories: bird, cat, circle, house, square, and triangle.
- Smart UI with brush, eraser, and color selection options.
- Live prediction and score display after user drawing.
- Data augmentation applied to increase training data diversity (rotation, flip, zoom, brightness changes).

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- Scikit-image
- Scikit-learn
- NLTK (for text stopwords removal, auxiliary)
- Matplotlib

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

1. Clone the repository and navigate to the project folder.

2. Install required Python packages:
   ```bash
   pip install tensorflow opencv-python mediapipe scikit-image scikit-learn matplotlib nltk

3. Ensure the following files/folders exist inside the project:
   - `best_model.h5`
   - `Header/` folder with icon images (already provided).

4. Update the paths in `data_dir` inside `main.py` and `virtual_painter.py` (currently set to a local path).

5. If you don't have the QuickDraw .npy files, you can download subsets from the Google QuickDraw Dataset:
   - https://github.com/googlecreativelab/quickdraw-dataset

6. To run the real-time application:
   ```bash
   python virtual_painter.py

7. Controls:
   - Draw with your index finger.
   - Select color/eraser with two fingers.
   - Press `p` to predict, `c` to clear, `q` to quit.

---

## Notes

- The trained model (`best_model.h5`) is provided and does not require retraining unless desired.
- A working webcam is required to run the virtual painter app.
- This project used a subset of the QuickDraw dataset (6 categories) for faster and more focused training.
