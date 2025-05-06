# Microsoft (MSFT) Stock Price Prediction

This project builds an end-to-end machine learning pipeline to predict Microsoft (MSFT) stock closing prices using historical market data.

## Project Steps

- Data Collection: Downloaded MSFT stock data from Yahoo Finance (2010–present).
- Feature Engineering: Created 7-day and 30-day moving averages to capture short-term and long-term price trends.
- Exploratory Data Analysis (EDA): Visualized stock price movements, moving averages, and analyzed feature correlations.
- Model Building: Trained Linear Regression, Ridge Regression, Lasso Regression, and Support Vector Regressor (SVR) models.
- Feature Scaling: Applied feature scaling to improve Lasso and SVM model performance.
- Model Evaluation: Compared models using RMSE and R² scores on a held-out test set.
- Model Selection and Saving: Selected Ridge Regression based on the lowest RMSE and saved the model for future use.

## Key Learnings

- Moving average features improved the models' ability to capture market trends.
- Proper feature scaling was critical for regularized and kernel-based models (Lasso and SVM).
- Manual comparison of multiple models ensured robust selection based on quantitative metrics.
