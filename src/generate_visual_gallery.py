"""
Generate a visual gallery for the synthetic 50 Startups CRISP-DM project.

The charts are intentionally lightweight and static so they can be reused in
README.md and the GitHub Pages dashboard without adding a web framework.
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


DATA_PATH = PROJECT_ROOT / "data" / "50_Startups.csv"

OVERVIEW_PATH = OUTPUTS_DIR / "dataset_overview_dashboard.png"
RELATIONSHIPS_PATH = OUTPUTS_DIR / "spending_profit_relationships.png"
HEATMAP_PATH = OUTPUTS_DIR / "correlation_heatmap.png"
METRICS_PATH = OUTPUTS_DIR / "model_metrics_summary.png"

NUMERIC_COLUMNS = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
MODEL_METRICS = {
    "Test R2 Score": 0.9460,
    "Test MAE": 5578.56,
    "Test RMSE": 6971.82,
    "5-fold CV mean R2": 0.9694,
    "5-fold CV std": 0.0165,
}


def load_dataset():
    """Load the existing synthetic dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run python src/generate_synthetic_startups.py first."
        )
    return pd.read_csv(DATA_PATH)


def money(value):
    """Format a numeric value as compact money text."""
    return f"${value:,.0f}"


def save_dataset_overview(df):
    """Create a compact project and dataset summary dashboard."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    state_counts = df["State"].value_counts().sort_index()

    fig = plt.figure(figsize=(14, 8), facecolor="#F7F9FB")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.25)
    ax_title = fig.add_subplot(gs[0, :])
    ax_state = fig.add_subplot(gs[1, 0])
    ax_features = fig.add_subplot(gs[1, 1])
    ax_notice = fig.add_subplot(gs[1, 2])

    ax_title.axis("off")
    cards = [
        ("Rows", f"{len(df)}"),
        ("Input Features", "4"),
        ("Target", "Profit"),
        ("Dataset Type", "Synthetic"),
    ]
    ax_title.text(
        0.02,
        0.82,
        "Dataset Overview Dashboard",
        fontsize=22,
        weight="bold",
        color="#17202A",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.02,
        0.64,
        "AI-generated synthetic 50 Startups-style dataset for CRISP-DM workflow demonstration.",
        fontsize=12,
        color="#53616F",
        transform=ax_title.transAxes,
    )

    for index, (label, value) in enumerate(cards):
        x = 0.02 + index * 0.24
        ax_title.add_patch(
            plt.Rectangle(
                (x, 0.12),
                0.2,
                0.32,
                transform=ax_title.transAxes,
                facecolor="#FFFFFF",
                edgecolor="#D9E2EC",
                linewidth=1.2,
            )
        )
        ax_title.text(x + 0.025, 0.32, value, fontsize=20, weight="bold", transform=ax_title.transAxes)
        ax_title.text(x + 0.025, 0.19, label, fontsize=10, color="#66788A", transform=ax_title.transAxes)

    ax_state.bar(state_counts.index, state_counts.values, color=["#4C78A8", "#72B7B2", "#F58518"])
    ax_state.set_title("Rows by State", weight="bold")
    ax_state.set_ylabel("Rows")
    ax_state.grid(axis="y", alpha=0.25)

    feature_text = "\n".join(
        [
            "Features:",
            "- R&D Spend",
            "- Administration",
            "- Marketing Spend",
            "- State",
            "",
            "Target:",
            "- Profit",
        ]
    )
    ax_features.axis("off")
    ax_features.set_title("Columns", weight="bold", pad=12)
    ax_features.text(
        0.05,
        0.88,
        feature_text,
        fontsize=12,
        va="top",
        linespacing=1.6,
        bbox=dict(facecolor="#FFFFFF", edgecolor="#D9E2EC", boxstyle="round,pad=0.6"),
    )

    notice = (
        "Important Notice\n\n"
        "This dataset is AI-generated synthetic teaching data.\n\n"
        "It is used for educational CRISP-DM workflow demonstration only.\n\n"
        "It is not real business research and does not support causal claims."
    )
    ax_notice.axis("off")
    ax_notice.set_title("Scope", weight="bold", pad=12)
    ax_notice.text(
        0.05,
        0.88,
        notice,
        fontsize=11,
        va="top",
        linespacing=1.55,
        bbox=dict(facecolor="#FFFFFF", edgecolor="#D9E2EC", boxstyle="round,pad=0.6"),
    )

    plt.savefig(OVERVIEW_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_spending_relationships(df):
    """Create scatter plots for spending variables versus profit."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#FFFFFF")
    features = ["R&D Spend", "Marketing Spend", "Administration"]
    colors = {"California": "#4C78A8", "Florida": "#F58518", "New York": "#54A24B"}

    for ax, feature in zip(axes, features):
        for state, state_df in df.groupby("State"):
            ax.scatter(
                state_df[feature],
                state_df["Profit"],
                label=state,
                s=52,
                alpha=0.82,
                color=colors[state],
                edgecolor="white",
                linewidth=0.7,
            )
        z = np.polyfit(df[feature], df["Profit"], 1)
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        ax.plot(x_line, z[0] * x_line + z[1], color="#263238", linewidth=1.8, alpha=0.8)
        ax.set_title(f"{feature} vs Profit", weight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Profit")
        ax.grid(alpha=0.25)

    axes[0].legend(title="State", frameon=True, fontsize=9)
    fig.suptitle("Spending and Profit Relationships", fontsize=16, weight="bold", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Synthetic teaching data only. Trend lines show association in this generated dataset, not real-world causality.",
        ha="center",
        fontsize=10,
        color="#566573",
    )
    plt.savefig(RELATIONSHIPS_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df):
    """Create a numeric correlation heatmap."""
    corr = df[NUMERIC_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#FFFFFF")
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(NUMERIC_COLUMNS)))
    ax.set_yticks(range(len(NUMERIC_COLUMNS)))
    ax.set_xticklabels(NUMERIC_COLUMNS, rotation=30, ha="right")
    ax.set_yticklabels(NUMERIC_COLUMNS)
    ax.set_title("Correlation Heatmap", fontsize=16, weight="bold", pad=16)

    for row in range(len(NUMERIC_COLUMNS)):
        for col in range(len(NUMERIC_COLUMNS)):
            value = corr.iloc[row, col]
            text_color = "white" if abs(value) > 0.55 else "#1F2933"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, weight="bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=15)
    fig.text(
        0.5,
        0.01,
        "Numeric features only; correlations describe this synthetic dataset.",
        ha="center",
        fontsize=10,
        color="#566573",
    )
    plt.savefig(HEATMAP_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_model_metrics_summary():
    """Create a metric card image for the current baseline model results."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#F7F9FB")
    ax.axis("off")
    ax.text(
        0.04,
        0.88,
        "Model Metrics Summary",
        fontsize=22,
        weight="bold",
        color="#17202A",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.78,
        "Linear Regression baseline with one-hot encoded State on synthetic teaching data.",
        fontsize=12,
        color="#53616F",
        transform=ax.transAxes,
    )

    metric_items = list(MODEL_METRICS.items())
    positions = [(0.04, 0.48), (0.36, 0.48), (0.68, 0.48), (0.20, 0.16), (0.52, 0.16)]

    for (label, value), (x, y) in zip(metric_items, positions):
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                0.28,
                0.22,
                transform=ax.transAxes,
                facecolor="#FFFFFF",
                edgecolor="#D9E2EC",
                linewidth=1.2,
            )
        )
        display_value = f"{value:.4f}" if "R2" in label or "std" in label else f"{value:,.2f}"
        ax.text(x + 0.025, y + 0.125, display_value, fontsize=20, weight="bold", transform=ax.transAxes)
        ax.text(x + 0.025, y + 0.055, label, fontsize=10, color="#66788A", transform=ax.transAxes)

    ax.text(
        0.04,
        0.05,
        "Metrics are for workflow demonstration. They should not be read as real startup performance evidence.",
        fontsize=10,
        color="#566573",
        transform=ax.transAxes,
    )
    plt.savefig(METRICS_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_dataset()
    save_dataset_overview(df)
    save_spending_relationships(df)
    save_correlation_heatmap(df)
    save_model_metrics_summary()

    print("Visual gallery generated:")
    for path in [OVERVIEW_PATH, RELATIONSHIPS_PATH, HEATMAP_PATH, METRICS_PATH]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
