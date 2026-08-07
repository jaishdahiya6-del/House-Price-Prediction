"""Train, compare, and select the best regression model.

Run: python train.py
Outputs:
  - models/best_model.pkl, models/scaler.pkl
  - reports/leaderboard.csv
  - images/*.png (correlation heatmap, feature importance, actual vs predicted, residuals)
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
from data_loader import load_data
from preprocessing import handle_missing_values, cap_outliers_iqr, split_data
from feature_engineering import add_derived_features, encode_categorical, scale_features
from evaluate import compute_metrics
from utils import get_logger, save_object

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
for d in (MODELS_DIR, IMAGES_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)

TARGET = "median_house_value"

MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1),
}


def main():
    df = load_data()
    df = handle_missing_values(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df = cap_outliers_iqr(df, numeric_cols)
    df = add_derived_features(df)
    df = encode_categorical(df)

    # Correlation heatmap (numeric columns only)
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f",
                cmap="coolwarm", square=True, annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    X_train, X_test, y_train, y_test = split_data(df, TARGET)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    results = []
    fitted_models = {}
    for name, model in MODELS.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        metrics = compute_metrics(y_test, preds, X_train_s.shape[1])
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2", n_jobs=-1)
        metrics["Model"] = name
        metrics["CV_R2_Mean"] = round(cv_scores.mean(), 4)
        results.append(metrics)
        fitted_models[name] = model
        logger.info(f"{name} -> RMSE={metrics['RMSE']}, R2={metrics['R2']}")

    leaderboard = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    leaderboard = leaderboard[["Model", "MAE", "MSE", "RMSE", "R2", "Adjusted_R2", "MAPE", "CV_R2_Mean"]]
    leaderboard.to_csv(os.path.join(REPORTS_DIR, "leaderboard.csv"), index=False)
    print("\n=== MODEL LEADERBOARD ===")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    logger.info(f"Best model: {best_name}")

    save_object(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    save_object(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    save_object(list(X_train.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))
    save_object(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    # Feature importance (if supported)
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values()
        plt.figure(figsize=(8, 6))
        imp.plot(kind="barh", color="#2c7fb8")
        plt.title(f"Feature Importance - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=120)
        plt.close()

    # Actual vs Predicted
    best_preds = best_model.predict(X_test_s)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, best_preds, alpha=0.3, color="#2c7fb8")
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual Price ($100k)")
    plt.ylabel("Predicted Price ($100k)")
    plt.title(f"Actual vs Predicted - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # Residual plot
    residuals = y_test.values - best_preds
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#e34a33")
    plt.title(f"Residual Distribution - {best_name}")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "residual_distribution.png"), dpi=120)
    plt.close()

    logger.info("Training complete. Artifacts saved to models/, reports/, images/.")
    return leaderboard, best_name


if __name__ == "__main__":
    main()
