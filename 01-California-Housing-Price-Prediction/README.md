# California Housing Price Prediction

This project builds an end-to-end machine learning pipeline to predict median house values across California districts using census data.

## Project Steps

- Exploratory Data Analysis (EDA): Visualized feature distributions, explored feature correlations, and examined key relationships with the target variable.
- Feature Engineering: Identified skewed features and applied log transformations to improve model performance and stability.
- Model Building: Trained both a Linear Regression and a Random Forest Regressor.
- Evaluation: Assessed models using Root Mean Squared Error (RMSE) on a test set and through 5-fold cross-validation.
- Model Selection and Saving: Selected the Random Forest model based on superior performance and saved it using `joblib`.
- *Note:* Model file not uploaded due to GitHub file size restrictions.

## Key Learnings

- Handling feature skewness can significantly impact regression performance.
- Random Forest models outperform simple linear models when capturing complex, non-linear relationships.
- Cross-validation provides more reliable model evaluation compared to a single train-test split.
