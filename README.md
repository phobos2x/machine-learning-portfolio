# Machine Learning & Deep Learning Portfolio

Welcome to my Machine Learning and Deep Learning project portfolio!
This repository contains a collection of projects demonstrating practical skills across regression, classification, natural language processing (NLP), and computer vision.

Each project showcases expertise in:

- Data cleaning, preprocessing, and augmentation
- Feature engineering (e.g., TF-IDF vectorization, SMOTE oversampling, image transformations)
- Building, evaluating, and selecting machine learning models (regression, classification, CNNs)
- Model saving and deployment preparation (Joblib, Keras `.h5` models)
- Real-time computer vision applications using OpenCV and MediaPipe

---

## Projects

### 1. California Housing Price Prediction

- **Problem:** Predict median house prices using California census features.
- **Techniques:** Exploratory Data Analysis (EDA), feature transformation (log normalization), Random Forest Regression, RMSE evaluation.
- **Note:** The trained model `.pkl` file is not uploaded due to size limits but can be reproduced by running the provided scripts.

### 2. Stock Price Prediction (MSFT)
- **Problem:** Predict Microsoft stock closing prices from historical financial data.
- **Techniques:** Regression models (Linear, Ridge, Lasso, Support Vector Regressor), TF-IDF feature scaling, RMSE and R² evaluation; Ridge Regression selected as best model.

### 3. Heart Failure Prediction
- **Problem:** Predict likelihood of heart failure events based on clinical data.
- **Techniques:** Classification models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting), SMOTE for class balancing, ROC-AUC evaluation; Random Forest selected based on highest AUC.

### 4. Spam Detector System
- **Problem:** Classify SMS messages as spam or ham using natural language processing.
- **Techniques:** Text preprocessing (lowercasing, punctuation removal, stopword filtering), TF-IDF vectorization, classification models (Naive Bayes, Logistic Regression, Random Forest), ROC-AUC evaluation; Logistic Regression selected.

### 5. Hand-Drawn Object Recognition with Real-Time Hand Tracking
- **Problem:** Recognize objects drawn in the air via hand gestures detected by webcam.
- **Techniques:** Real-time hand tracking (MediaPipe + OpenCV), custom CNN model trained on QuickDraw subset (6 categories), live prediction and score display, extensive data augmentation applied.

---

## Skills Demonstrated

- Data Cleaning, Preprocessing, and Augmentation
- Feature Engineering and Model Training
- Supervised Machine Learning (Regression and Classification)
- Natural Language Processing (NLP)
- Real-Time Computer Vision (OpenCV, MediaPipe)
- Convolutional Neural Network (CNN) Design and Training
- Model Evaluation (RMSE, ROC-AUC, Precision, Recall, F1-Score)
- Model Saving and Deployment Preparation
- Real-Time Application Development with Python
