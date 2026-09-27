"""Train, compare, and tune regression models for house price prediction.

Run: python src/train.py
Outputs:
  - models/best_model.pkl, models/scaler.pkl, models/feature_columns.pkl, models/best_model_name.pkl
  - reports/leaderboard.csv
  - images/*.png (correlation heatmap, feature importance, actual vs predicted, residuals, geographical distribution, feature distributions)
"""
import os
import sys
import warnings
from typing import Tuple, Dict, Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(os.path.dirname(__file__))
from data_loader import load_data
from preprocessing import handle_missing_values, cap_outliers_iqr, split_data
from feature_engineering import add_derived_features, encode_categorical, scale_features
from evaluate import compute_metrics
from utils import get_logger, save_object, set_seed

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for d in (MODELS_DIR, IMAGES_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)

TARGET = "median_house_value"


def tune_model_with_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 3,
    random_state: int = 42
) -> Any:
    """Tune hyperparameters for tree-based models using Optuna cross-validation.

    Args:
        model_name: Name of model algorithm ('XGBoost', 'LightGBM', 'RandomForest').
        X_train: Scaled training features.
        y_train: Training target series.
        n_trials: Number of Optuna optimization trials.
        random_state: Seed for cross-validation splits.

    Returns:
        Instantiated model with optimal hyperparameter settings.
    """
    logger.info(f"Tuning {model_name} using Optuna ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

        if model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "random_state": random_state,
                "n_jobs": 1,
            }
            model = XGBRegressor(**params)
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                "random_state": random_state,
                "n_jobs": 1,
                "verbose": -1,
            }
            model = LGBMRegressor(**params)
        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                "max_depth": trial.suggest_int("max_depth", 6, 12),
                "random_state": random_state,
                "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")

        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Optuna best parameters for {model_name}: {study.best_params}")

    best_params = study.best_params
    best_params["random_state"] = random_state

    if model_name == "XGBoost":
        best_params["n_jobs"] = 1
        return XGBRegressor(**best_params)
    elif model_name == "LightGBM":
        best_params["n_jobs"] = 1
        best_params["verbose"] = -1
        return LGBMRegressor(**best_params)
    elif model_name == "RandomForest":
        best_params["n_jobs"] = 1
        return RandomForestRegressor(**best_params)


def create_visualizations(
    df: pd.DataFrame,
    best_model: Any,
    X_test_s: pd.DataFrame,
    y_test: pd.Series,
    best_name: str,
    feature_names: Sequence[str]
) -> None:
    """Generate and save comprehensive EDA and model evaluation plots.

    Args:
        df: Raw / engineered dataset for EDA plots.
        best_model: Fitted best performing model object.
        X_test_s: Scaled test feature DataFrame.
        y_test: Test target Series.
        best_name: Name string of best performing model.
        feature_names: Names of feature columns.
    """
    # 1. Correlation heatmap
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f",
                cmap="coolwarm", square=True, annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # 2. Geographical distribution plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["longitude"], df["latitude"],
        c=df[TARGET], cmap="jet", alpha=0.4,
        s=df["population"] / 100, label="Population"
    )
    plt.colorbar(scatter, label="Median House Value ($)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geographical Distribution of Housing Prices in California", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "geographical_distribution.png"), dpi=120)
    plt.close()

    # 3. Feature distributions plot
    num_cols = ["median_income", "housing_median_age", "total_rooms", TARGET]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flatten(), num_cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=ax, color="#2c7fb8")
            ax.set_title(f"Distribution of {col}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_distributions.png"), dpi=120)
    plt.close()

    # 4. Feature importance plot
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values()
        plt.figure(figsize=(9, 6))
        imp.plot(kind="barh", color="#2c7fb8")
        plt.title(f"Feature Importance - {best_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=120)
        plt.close()

    # 5. Actual vs Predicted plot
    best_preds = best_model.predict(X_test_s)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, best_preds, alpha=0.3, color="#2c7fb8")
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    plt.plot(lims, lims, "r--", linewidth=2)
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title(f"Actual vs Predicted - {best_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # 6. Residual distribution plot
    residuals = y_test.values - best_preds
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#e34a33")
    plt.title(f"Residual Distribution - {best_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Prediction Error ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "residual_distribution.png"), dpi=120)
    plt.close()


def main(enable_tuning: bool = True) -> Tuple[pd.DataFrame, str]:
    """Execute end-to-end model training, evaluation, and artifact serialization.

    Args:
        enable_tuning: Whether to run Optuna hyperparameter optimization.

    Returns:
        Tuple of (leaderboard_df, best_model_name).
    """
    set_seed(42)

    df = load_data()
    df = handle_missing_values(df)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != TARGET]
    df = cap_outliers_iqr(df, numeric_cols)
    df = add_derived_features(df)
    df = encode_categorical(df)

    X_train, X_test, y_train, y_test = split_data(df, TARGET, test_size=0.2, random_state=42)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # Base candidate models
    models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    }

    if enable_tuning:
        models["RandomForest"] = tune_model_with_optuna("RandomForest", X_train_s, y_train, n_trials=3)
        models["XGBoost"] = tune_model_with_optuna("XGBoost", X_train_s, y_train, n_trials=3)
        models["LightGBM"] = tune_model_with_optuna("LightGBM", X_train_s, y_train, n_trials=3)
    else:
        models["RandomForest"] = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
        models["XGBoost"] = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=1)
        models["LightGBM"] = LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=1, verbose=-1)

    results = []
    fitted_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        metrics = compute_metrics(y_test, preds, X_train_s.shape[1])
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="r2", n_jobs=1)
        metrics["Model"] = name
        metrics["CV_R2_Mean"] = round(float(cv_scores.mean()), 4)
        results.append(metrics)
        fitted_models[name] = model
        logger.info(f"{name} -> RMSE={metrics['RMSE']}, R2={metrics['R2']}, CV_R2={metrics['CV_R2_Mean']}")

    leaderboard = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    leaderboard = leaderboard[["Model", "MAE", "MSE", "RMSE", "R2", "Adjusted_R2", "MAPE", "CV_R2_Mean"]]
    leaderboard.to_csv(os.path.join(REPORTS_DIR, "leaderboard.csv"), index=False)

    logger.info(f"\n=== MODEL LEADERBOARD ===\n{leaderboard.to_string(index=False)}")

    best_name = leaderboard.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    logger.info(f"Best model selected: {best_name}")

    # Serialize artifacts
    save_object(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    save_object(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    save_object(list(X_train.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))
    save_object(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    # Generate charts
    create_visualizations(df, best_model, X_test_s, y_test, best_name, list(X_train.columns))

    logger.info("Training pipeline complete. Artifacts saved to models/, reports/, images/.")
    return leaderboard, best_name


if __name__ == "__main__":
    main()
