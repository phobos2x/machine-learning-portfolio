# Heart Failure Prediction

This project builds an end-to-end machine learning pipeline to predict the likelihood of heart failure events based on clinical patient data.

## Project Steps

- Data Loading and Exploration: Analyzed patient clinical data, explored feature distributions and correlations.
- Handling Class Imbalance: Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the training dataset.
- Model Building: Trained Logistic Regression, Random Forest, XGBoost, and Gradient Boosting models.
- Evaluation: Assessed models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC scores.
- Model Selection and Saving: Selected Random Forest based on the highest ROC-AUC score and saved the model for future use.

## Key Learnings

- Addressing class imbalance with SMOTE improved model fairness and recall performance.
- Ensemble models like Random Forest and XGBoost showed strong predictive capabilities.
- ROC-AUC evaluation provided deeper insights than basic accuracy metrics alone.
