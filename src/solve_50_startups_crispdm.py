"""
Synthetic 50 Startups CRISP-DM Startup Profit Prediction

This beginner-friendly script follows the six CRISP-DM steps:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MATPLOTLIB_CACHE_DIR = OUTPUTS_DIR / "matplotlib_cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = PROJECT_ROOT / "data" / "50_Startups.csv"
MODEL_PATH = OUTPUTS_DIR / "startup_profit_model.pkl"
PLOT_PATH = OUTPUTS_DIR / "feature_selection_performance.png"

TARGET_COLUMN = "Profit"
NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def print_step(step_number, title):
    """Print a clear CRISP-DM step heading."""
    print("\n" + "=" * 70)
    print(f"CRISP-DM Step {step_number}: {title}")
    print("=" * 70)


def check_dataset_exists():
    """Stop early with a helpful message if the CSV file is missing."""
    if not DATA_PATH.exists():
        message = (
            f"Dataset not found: {DATA_PATH}\n\n"
            "Please generate the synthetic teaching dataset first:\n"
            "python src/generate_synthetic_startups.py"
        )
        raise FileNotFoundError(message)


def load_dataset():
    """Load the startup dataset."""
    check_dataset_exists()
    return pd.read_csv(DATA_PATH)


def validate_columns(df):
    """Make sure all required columns are present."""
    required_columns = ALL_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_model(feature_columns):
    """Create a preprocessing and Linear Regression pipeline."""
    numeric_features = [col for col in feature_columns if col in NUMERIC_FEATURES]
    categorical_features = [col for col in feature_columns if col in CATEGORICAL_FEATURES]

    transformers = []

    if numeric_features:
        transformers.append(("numeric", "passthrough", numeric_features))

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    return model


def evaluate_predictions(y_true, y_pred):
    """Calculate regression evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def run_cross_validation(model, X, y):
    """Run 5-fold cross-validation because the dataset is small."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")
    return scores


def create_feature_selection_plot(df):
    """Compare 5-fold CV performance for several simple feature sets."""
    feature_sets = {
        "R&D only": ["R&D Spend"],
        "R&D + Marketing": ["R&D Spend", "Marketing Spend"],
        "All numeric": NUMERIC_FEATURES,
        "All features": ALL_FEATURES,
    }

    labels = []
    mean_scores = []

    y = df[TARGET_COLUMN]

    for label, feature_columns in feature_sets.items():
        X = df[feature_columns]
        model = build_model(feature_columns)
        cv_scores = run_cross_validation(model, X, y)

        labels.append(label)
        mean_scores.append(cv_scores.mean())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, mean_scores, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.title("Feature Selection Performance with 5-Fold CV")
    plt.xlabel("Feature Set")
    plt.ylabel("Mean R2 Score")
    plt.ylim(min(0, min(mean_scores) - 0.1), 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, mean_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print_step(1, "Business Understanding")
    print("Goal: predict startup Profit from spending features and State.")
    print("Dataset type: AI-generated synthetic teaching data.")
    print("Scope: workflow demonstration, not real business research.")
    print("Task type: supervised machine learning regression.")

    print_step(2, "Data Understanding")
    df = load_dataset()
    validate_columns(df)
    print(f"Dataset path: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe(include="all"))

    print_step(3, "Data Preparation")
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    print(f"Features: {ALL_FEATURES}")
    print(f"Target: {TARGET_COLUMN}")
    print("Categorical feature 'State' will be encoded using OneHotEncoder.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")

    print_step(4, "Modeling")
    model = build_model(ALL_FEATURES)
    model.fit(X_train, y_train)
    print("Model trained: LinearRegression inside a preprocessing Pipeline.")

    print_step(5, "Evaluation")
    y_pred = model.predict(X_test)
    r2, mae, rmse = evaluate_predictions(y_test, y_pred)
    cv_scores = run_cross_validation(model, X, y)

    print(f"Test R2 Score: {r2:.4f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"5-fold CV R2 scores: {np.round(cv_scores, 4)}")
    print(f"5-fold CV mean R2: {cv_scores.mean():.4f}")
    print(f"5-fold CV standard deviation: {cv_scores.std():.4f}")

    create_feature_selection_plot(df)
    print(f"Feature selection plot saved to: {PLOT_PATH}")

    print_step(6, "Deployment")
    final_model = build_model(ALL_FEATURES)
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Final model saved to: {MODEL_PATH}")
    print("The saved pipeline can preprocess new data and predict startup profit.")


if __name__ == "__main__":
    main()
