# Heart Disease Classification – ML Assignment 2

## a. Problem Statement
The objective of this project is to build and compare multiple machine learning
classification models to predict the presence of heart disease using clinical
patient data. The models are evaluated using standard classification metrics and
deployed using a Streamlit web application.

## b. Dataset Description
The dataset used is the UCI Heart Disease dataset. It contains medical attributes
such as age, sex, chest pain type, cholesterol levels, blood pressure, ECG results,
and other clinical measurements.  
The target variable indicates the presence (1) or absence (0) of heart disease.

- Type: Binary Classification  
- Instances: > 500  
- Features: > 12 after encoding  

## c. Models Used and Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Decision Tree | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| K-Nearest Neighbors | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Naive Bayes | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| Random Forest (Ensemble) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| XGBoost (Ensemble) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |

## Model Observations

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Provides a strong baseline with good interpretability and performs well with scaled features. |
| Decision Tree | Captures non-linear relationships but requires depth control to avoid overfitting. |
| KNN | Performs well with normalized data but is computationally expensive for large datasets. |
| Naive Bayes | Fast and efficient, but independence assumption limits performance. |
| Random Forest | Robust ensemble model with strong overall performance and reduced variance. |
| XGBoost | Achieves the best performance by correcting previous model errors iteratively. |

## Streamlit Application Features
- CSV dataset upload (test data)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report

## Deployment
The application is deployed using Streamlit Community Cloud and can be accessed
via the shared public link.
