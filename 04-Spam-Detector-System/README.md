# Spam Detector System

This project builds a machine learning pipeline to classify SMS messages as spam or ham (not spam) using Natural Language Processing (NLP) techniques.

## Project Steps

- Data Loading: Loaded SMS Spam Collection dataset from the UCI Machine Learning Repository.
- Text Preprocessing: Applied lowercasing, punctuation removal, and stopwords removal to clean the text data.
- Feature Engineering: Used TF-IDF vectorization to transform messages into numerical feature vectors.
- Model Building: Trained Naive Bayes, Logistic Regression, and Random Forest classifiers.
- Model Evaluation: Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics; visualized ROC curves.
- Model Selection and Saving: Selected Logistic Regression based on the highest AUC score and saved it for deployment.

## Key Learnings

- Proper text preprocessing significantly boosts NLP classification performance.
- TF-IDF vectorization effectively highlights informative words.
- ROC-AUC analysis provides deeper model evaluation beyond basic accuracy.
