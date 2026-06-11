"""
Generate an AI-created synthetic 50 Startups-style dataset.

This data is for educational CRISP-DM workflow demonstration only.
It is not the original Kaggle dataset and should not be used for real
business research or causal conclusions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "50_Startups.csv"


def generate_synthetic_data(row_count=50, random_seed=42):
    """Create a reproducible synthetic startup profit dataset."""
    rng = np.random.default_rng(random_seed)

    states = np.array(["New York", "California", "Florida"])
    state_adjustments = {
        "New York": 3500,
        "California": 1800,
        "Florida": 0,
    }

    rd_spend = rng.uniform(0, 170000, row_count)
    administration = rng.uniform(50000, 160000, row_count)
    marketing_spend = rng.uniform(0, 480000, row_count)
    state = rng.choice(states, row_count, replace=True)
    noise = rng.normal(0, 6000, row_count)

    profit = (
        45000
        + 0.72 * rd_spend
        + 0.04 * administration
        + 0.10 * marketing_spend
        + np.array([state_adjustments[item] for item in state])
        + noise
    )

    df = pd.DataFrame(
        {
            "R&D Spend": np.round(rd_spend, 2),
            "Administration": np.round(administration, 2),
            "Marketing Spend": np.round(marketing_spend, 2),
            "State": state,
            "Profit": np.round(profit, 2),
        }
    )

    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)

    df = generate_synthetic_data()
    df.to_csv(OUTPUT_PATH, index=False)

    print("Synthetic 50 Startups-style dataset generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Output path: {OUTPUT_PATH}")
    print("\nImportant note:")
    print("This is AI-generated synthetic data for educational use only.")
    print("It is not the original Kaggle 50 Startups dataset.")


if __name__ == "__main__":
    main()
