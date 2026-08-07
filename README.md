# 🏠 AI House Price Prediction System

Predicts California house prices from real 1990 census data using a compact,
production-style ML pipeline: modular Python source, model comparison, a
saved best model, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project trains and compares several regression models on the
**California Housing dataset** (20,640 real records, 1990 U.S. census) and
serves the best one through a Streamlit web app. It's intentionally scoped
to be something you could actually run end-to-end in minutes, not a wall of
unused boilerplate.

**Live results from the last training run:**

| Model | MAE | RMSE | R² | Adjusted R² | MAPE (%) | CV R² (5-fold) |
|---|---|---|---|---|---|---|
| **XGBoost (best)** | **$29,801** | **$45,814** | **0.840** | 0.839 | 16.96 | 0.839 |
| GradientBoosting | $33,842 | $50,579 | 0.805 | 0.804 | 19.36 | 0.824 |
| RandomForest | $33,191 | $51,132 | 0.801 | 0.800 | 18.70 | 0.806 |
| Ridge | $51,681 | $73,039 | 0.593 | 0.591 | 30.88 | 0.676 |
| LinearRegression | $51,682 | $73,039 | 0.593 | 0.591 | 30.89 | 0.676 |
| Lasso | $51,682 | $73,039 | 0.593 | 0.591 | 30.89 | 0.676 |

XGBoost was selected automatically (lowest RMSE) and saved to `models/best_model.pkl`.

## Features

- Real dataset (20,640 rows), pulled automatically — no manual download
- Missing-value handling, IQR-based outlier capping, one-hot encoding
- Engineered ratio features (rooms/household, bedrooms/room, population/household)
- 6 models trained and compared with 5-fold cross-validation
- MAE / MSE / RMSE / R² / Adjusted R² / MAPE leaderboard
- Best model auto-selected and serialized with Joblib
- Correlation heatmap, feature importance, actual-vs-predicted, and residual charts
- Interactive Streamlit dashboard with sliders, live prediction, prediction history, and CSV export

## Project Structure

```
House-Price-Prediction/
├── data/
│   ├── raw/                  # cached source CSV (auto-downloaded)
│   └── processed/
├── src/
│   ├── data_loader.py        # fetches & caches the dataset
│   ├── preprocessing.py      # missing values, outliers, train/test split
│   ├── feature_engineering.py# derived features, encoding, scaling
│   ├── train.py               # trains all models, builds leaderboard, saves best model
│   ├── predict.py             # loads saved model and predicts on new input
│   ├── evaluate.py            # MAE/MSE/RMSE/R2/Adjusted R2/MAPE
│   └── utils.py               # logging + save/load helpers
├── app/
│   └── streamlit_app.py       # interactive dashboard
├── models/                    # best_model.pkl, scaler.pkl, feature_columns.pkl
├── images/                    # generated charts
├── reports/
│   └── leaderboard.csv
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

```bash
git clone https://github.com/<your-username>/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
```

## Usage

**1. Train the models** (downloads data, trains 6 models, saves the best one, generates charts):
```bash
python src/train.py
```

**2. Predict from Python:**
```bash
python src/predict.py
```

**3. Launch the dashboard:**
```bash
streamlit run app/streamlit_app.py
```

## Dataset

[California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
(Pace & Barry, 1997) — median house values for California districts from the
1990 census, with features like median income, house age, room/bedroom
counts, population, households, coordinates, and ocean proximity.

## Tech Stack

Python · Pandas · NumPy · Scikit-Learn · XGBoost · Matplotlib · Seaborn ·
Plotly · Streamlit · Joblib

## Future Improvements

- Add LightGBM/CatBoost and hyperparameter search (GridSearch/Optuna)
- SHAP-based explainability per prediction
- Swap in the Kaggle "House Prices: Advanced Regression Techniques" dataset
  for a second, higher-dimensional benchmark
- Dockerize and deploy the Streamlit app

## License

MIT — see [LICENSE](LICENSE).
