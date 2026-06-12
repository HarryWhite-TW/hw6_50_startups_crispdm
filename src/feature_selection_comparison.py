"""
Advanced feature selection comparison for the synthetic 50 Startups dataset.

This script compares several feature ranking methods on the same one-hot
encoded feature matrix. Each method produces a full ranking, then Linear
Regression is evaluated with the top 1 through top 5 ranked features.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MATPLOTLIB_CACHE_DIR = OUTPUTS_DIR / "matplotlib_cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = PROJECT_ROOT / "data" / "50_Startups.csv"
PLOT_PATH = OUTPUTS_DIR / "feature_selection_performance_allinone.png"

TARGET_COLUMN = "Profit"
RAW_FEATURES = ["R&D Spend", "Administration", "Marketing Spend", "State"]
MAX_FEATURES = 5
RANDOM_STATE = 42


def load_dataset():
    """Load and validate the synthetic startup dataset."""
    if not DATA_PATH.exists():
        message = (
            f"Dataset not found: {DATA_PATH}\n"
            "Generate it first with: python src/generate_synthetic_startups.py"
        )
        raise FileNotFoundError(message)

    df = pd.read_csv(DATA_PATH)
    required_columns = RAW_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def prepare_features(df):
    """One-hot encode State with drop-first while preserving readable names."""
    X = pd.get_dummies(df[RAW_FEATURES], columns=["State"], drop_first=True, dtype=float)
    y = df[TARGET_COLUMN]
    return X, y


def rank_with_forward_selection(X, y):
    """Build a full Sequential Forward Selection ranking using CV RMSE."""
    remaining_features = list(X.columns)
    selected_features = []
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    while remaining_features:
        candidate_scores = []

        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            model = LinearRegression()
            scores = cross_val_score(
                model,
                X[candidate_features],
                y,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            )
            candidate_scores.append((scores.mean(), feature))

        candidate_scores.sort(key=lambda item: (-item[0], item[1]))
        best_feature = candidate_scores[0][1]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features


def rank_with_rfe(X, y):
    """Rank features by recursive feature elimination with Linear Regression."""
    selector = RFE(
        estimator=LinearRegression(),
        n_features_to_select=1,
        step=1,
    )
    selector.fit(X, y)
    ranked = sorted(zip(selector.ranking_, X.columns), key=lambda item: (item[0], item[1]))
    return [feature for _, feature in ranked]


def rank_with_select_k_best(X, y):
    """Rank features by univariate f-regression score."""
    selector = SelectKBest(score_func=f_regression, k="all")
    selector.fit(X, y)
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranked = sorted(zip(scores, X.columns), key=lambda item: (-item[0], item[1]))
    return [feature for _, feature in ranked]


def rank_with_lasso(X, y):
    """Rank features by absolute standardized LassoCV coefficient."""
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000),
            ),
        ]
    )
    model.fit(X, y)
    coefficients = np.abs(model.named_steps["lasso"].coef_)
    ranked = sorted(zip(coefficients, X.columns), key=lambda item: (-item[0], item[1]))
    return [feature for _, feature in ranked]


def rank_with_random_forest(X, y):
    """Rank features by Random Forest impurity-based feature importance."""
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
    )
    model.fit(X, y)
    ranked = sorted(
        zip(model.feature_importances_, X.columns),
        key=lambda item: (-item[0], item[1]),
    )
    return [feature for _, feature in ranked]


def build_rankings(X, y):
    """Create feature rankings for all comparison methods."""
    return {
        "SFS / Forward": rank_with_forward_selection(X, y),
        "RFE": rank_with_rfe(X, y),
        "SelectKBest": rank_with_select_k_best(X, y),
        "Lasso": rank_with_lasso(X, y),
        "Random Forest": rank_with_random_forest(X, y),
    }


def evaluate_rankings(X, y, rankings):
    """Evaluate top-k features from each ranking using Linear Regression."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    records = []

    for method, ranking in rankings.items():
        for k in range(1, min(MAX_FEATURES, len(ranking)) + 1):
            selected_features = ranking[:k]
            model = LinearRegression()
            model.fit(X_train[selected_features], y_train)
            predictions = model.predict(X_test[selected_features])
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            records.append(
                {
                    "method": method,
                    "k": k,
                    "features": selected_features,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

    return pd.DataFrame(records)


def create_comparison_plot(results_df, rankings):
    """Create the all-in-one performance chart and top-5 ranking table."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.35], hspace=0.34, wspace=0.22)
    ax_rmse = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    colors = {
        "SFS / Forward": "#2F6B7C",
        "RFE": "#C44E52",
        "SelectKBest": "#4C72B0",
        "Lasso": "#55A868",
        "Random Forest": "#8172B3",
    }

    for method in rankings:
        method_df = results_df[results_df["method"] == method].sort_values("k")
        ax_rmse.plot(
            method_df["k"],
            method_df["rmse"],
            marker="o",
            linewidth=2.2,
            label=method,
            color=colors[method],
        )
        ax_r2.plot(
            method_df["k"],
            method_df["r2"],
            marker="o",
            linewidth=2.2,
            label=method,
            color=colors[method],
        )

    ax_rmse.set_title("RMSE by Number of Features (All Algorithms)", fontsize=13, weight="bold")
    ax_rmse.set_xlabel("Number of Features")
    ax_rmse.set_ylabel("Test RMSE")
    ax_rmse.set_xticks(range(1, MAX_FEATURES + 1))
    ax_rmse.legend(frameon=True, fontsize=9)

    ax_r2.set_title("R-squared by Number of Features (All Algorithms)", fontsize=13, weight="bold")
    ax_r2.set_xlabel("Number of Features")
    ax_r2.set_ylabel("Test R-squared")
    ax_r2.set_xticks(range(1, MAX_FEATURES + 1))
    ax_r2.legend(frameon=True, fontsize=9)

    table_rows = []
    for method, ranking in rankings.items():
        table_rows.append([method] + ranking[:MAX_FEATURES])

    column_labels = ["Algorithm", "Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]
    ax_table.axis("off")
    ax_table.set_title("Top 5 Feature Ranking by Algorithm", fontsize=13, weight="bold", pad=10)
    table = ax_table.table(
        cellText=table_rows,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row == 0:
            cell.set_facecolor("#F0F3F5")
            cell.set_text_props(weight="bold", color="#222222")
        elif col == 0:
            cell.set_facecolor("#F8FAFB")
            cell.set_text_props(weight="bold", color="#222222")
        else:
            cell.set_facecolor("#FFFFFF")

    fig.suptitle(
        "Feature Selection Comparison on Synthetic 50 Startups-style Data",
        fontsize=16,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "State is one-hot encoded with drop-first. Performance uses Linear Regression on a fixed 80/20 split.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    plt.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(rankings, results_df):
    """Print rankings and the best observed top-k result."""
    print("\nFeature rankings:")
    for method, ranking in rankings.items():
        print(f"- {method}: {', '.join(ranking)}")

    best_row = results_df.sort_values(["rmse", "k", "method"]).iloc[0]
    print("\nBest observed configuration by RMSE:")
    print(f"- Method: {best_row['method']}")
    print(f"- Number of features: {int(best_row['k'])}")
    print(f"- Features: {', '.join(best_row['features'])}")
    print(f"- RMSE: {best_row['rmse']:.2f}")
    print(f"- R-squared: {best_row['r2']:.4f}")


def main():
    df = load_dataset()
    X, y = prepare_features(df)

    print("Advanced Feature Selection Comparison")
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Expanded features: {list(X.columns)}")
    print("Dataset type: AI-generated synthetic teaching data.")

    rankings = build_rankings(X, y)
    results_df = evaluate_rankings(X, y, rankings)
    create_comparison_plot(results_df, rankings)
    print_summary(rankings, results_df)

    print(f"\nAll-in-one comparison plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
