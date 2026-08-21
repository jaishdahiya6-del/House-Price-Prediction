# 🏠 Production AI House Price Prediction System

A comprehensive, production-grade Machine Learning system for predicting California housing prices based on 1990 U.S. Census data. Built with a clean modular architecture, Optuna hyperparameter optimization, unit test coverage, and an interactive Streamlit web dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen)
![Optuna](https://img.shields.io/badge/Optuna-3.0%2B-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Executive Summary

This repository delivers an end-to-end regression system capable of estimating median house values in California block groups. It includes missing value imputation, IQR winsorizing for outlier treatment, ratio feature engineering, one-hot categorical encoding, standardization, multi-model candidate evaluation (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM), Optuna cross-validation tuning, automated model selection, and an interactive web app.

### 🏆 Live Model Leaderboard Results

| Model | MAE ($) | RMSE ($) | R² | Adjusted R² | MAPE (%) | 5-Fold CV R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Best)** | **$31,833.84** | **$48,142.61** | **0.8231** | **0.8224** | **18.13%** | **0.8269** |
| LightGBM | $33,255.60 | $49,254.58 | 0.8149 | 0.8141 | 18.97% | 0.8187 |
| GradientBoosting | $35,368.08 | $52,163.50 | 0.7924 | 0.7915 | 20.12% | 0.8079 |
| RandomForest | $36,234.83 | $54,565.24 | 0.7728 | 0.7719 | 20.54% | 0.7853 |
| Ridge | $51,680.56 | $73,038.71 | 0.5929 | 0.5913 | 30.88% | 0.6762 |
| LinearRegression | $51,682.21 | $73,039.25 | 0.5929 | 0.5913 | 30.89% | 0.6762 |
| Lasso | $51,682.21 | $73,039.26 | 0.5929 | 0.5913 | 30.89% | 0.6762 |

*The best model (XGBoost) is automatically serialized to `models/best_model.pkl` along with preprocessor scalers and feature mappings.*

---

## 📁 Repository Structure

```
House-Price-Prediction/
├── app/
│   └── streamlit_app.py        # Interactive Streamlit web application
├── data/
│   ├── raw/                   # Auto-downloaded raw California Housing CSV
│   └── processed/             # Cleaned & preprocessed datasets
├── images/                    # Generated charts & visualization outputs
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── feature_importance.png
│   ├── geographical_distribution.png
│   └── residual_distribution.png
├── models/                    # Serialized best model, scaler, and column order
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis notebook
│   ├── Feature_Engineering.ipynb # Feature engineering & preprocessing exploration
│   └── Model_Training.ipynb    # Model training, Optuna tuning, and evaluation
├── reports/
│   └── leaderboard.csv         # Exported model comparison leaderboard
├── src/
│   ├── data_loader.py          # Data ingestion and local caching
│   ├── preprocessing.py        # Missing value imputation, capping, train/test split
│   ├── feature_engineering.py  # Derived features, encoding, feature scaling
│   ├── evaluate.py             # Evaluation metrics (MAE, MSE, RMSE, R2, MAPE)
│   ├── train.py                # Hyperparameter tuning, CV, training, chart generation
│   ├── predict.py              # Single/batch prediction interface
│   └── utils.py                # Logger, seed setting, joblib object persistence
├── tests/                     # Unit test suite with Pytest
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   └── test_utils.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Description

The dataset consists of **20,640 records** from the 1990 California U.S. Census ([Pace & Barry, 1997](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)). Each record represents a block group:

- `longitude` & `latitude`: Geographical location coordinates.
- `housing_median_age`: Median age of houses in block.
- `total_rooms` & `total_bedrooms`: Aggregate count of rooms/bedrooms.
- `population`: Total population in block group.
- `households`: Total number of households.
- `median_income`: Median income in tens of thousands of USD.
- `ocean_proximity`: Categorical location relative to ocean (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`).
- **Target**: `median_house_value` in USD.

### Engineered Features
- `rooms_per_household` = `total_rooms` / `households`
- `bedrooms_per_room` = `total_bedrooms` / `total_rooms`
- `population_per_household` = `population` / `households`

---

## 🛠️ Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

### 2. Train Models & Tune Hyperparameters

Execute the end-to-end pipeline to download data, clean features, run Optuna hyperparameter tuning, evaluate 7 candidate algorithms, export performance metrics, and generate chart artifacts:
```bash
python src/train.py
```

### 3. Run Predictions in Python

Generate price predictions programmatically:
```python
from src.predict import predict_price

sample_house = {
    "longitude": -118.25,
    "latitude": 34.05,
    "housing_median_age": 25,
    "total_rooms": 3000,
    "total_bedrooms": 600,
    "population": 1400,
    "households": 550,
    "median_income": 5.5,
    "ocean_proximity": "NEAR OCEAN"
}

price = predict_price(sample_house)
print(f"Estimated Median House Value: ${price:,.2f}")
```

### 4. Launch Interactive Web App

Run the Streamlit application locally:
```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Unit Testing

Run the full pytest test suite to verify module behavior and system integrity:
```bash
python -m pytest tests/
```

All 13 unit tests cover data loading, imputation, outlier capping, feature engineering, evaluation metrics, seed setting, model artifact persistence, and prediction inference.

---

## 🚀 Deployment Options

- **Streamlit Community Cloud**: Connect your GitHub repository and set entry point to `app/streamlit_app.py`.
- **Docker Container**:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY . /app
  RUN pip install --no-cache-dir -r requirements.txt
  EXPOSE 8501
  CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
  ```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
