from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_figure(fig, output_dir, filename, show):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_data(df_clean, output_dir, show=False):
    """Save target correlations, historical distributions, and grid vs. finish."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    plt.figure(figsize=(10, 6))
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    correlations = df_clean[numeric_cols].corr()["positionOrder"].sort_values()

    drop_from_corr = ["positionOrder", "raceId", "driverId", "constructorId", "circuitId"]
    correlations = correlations.drop(labels=drop_from_corr, errors="ignore")

    correlations.plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title("Pearson Correlation with Finishing Position (positionOrder)", fontsize=14, pad=12)
    plt.xlabel("Correlation Coefficient (Lower/Negative = Better Finish)")
    plt.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    _save_figure(plt.gcf(), output_dir, "feature_correlations.png", show)

    skew_candidates = [
        "career_points", "career_wins", "career_podiums",
        "constructor_wins", "recent_form", "avg_finish_before_race"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(skew_candidates):
        sns.histplot(df_clean[col], kde=True, ax=axes[i], color="teal", bins=30)
        skew_val = df_clean[col].skew()
        axes[i].set_title(f"{col} (Skew: {skew_val:.2f})", fontsize=12)
        axes[i].set_xlabel("")

    plt.suptitle("Distribution & Skewness of Historical Features", fontsize=15, y=1.02)
    plt.tight_layout()
    _save_figure(plt.gcf(), output_dir, "feature_distributions.png", show)

    plt.figure(figsize=(9, 6))
    sns.boxplot(x="starting_grid", y="positionOrder", data=df_clean[df_clean["starting_grid"] <= 20], color="skyblue")
    plt.title("Starting Grid Position vs. Final Position Order (Grid 1 to 20)", fontsize=14, pad=12)
    plt.xlabel("Starting Grid")
    plt.ylabel("Finishing Position (positionOrder)")
    plt.tight_layout()
    _save_figure(plt.gcf(), output_dir, "grid_vs_finish.png", show)


def plot_ranking_quality(eval_df, output_dir, show=False):
    """Save per-race Spearman correlation and actual vs. predicted positions."""
    per_race_rho = pd.Series({rnd: g["positionOrder"].corr(g["predicted_position"], method="spearman")
                              for rnd, g in eval_df.groupby("race_round")})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(per_race_rho.index, per_race_rho.values, color="steelblue")
    axes[0].axhline(per_race_rho.mean(), color="red", linestyle="--", label=f"mean = {per_race_rho.mean():.2f}")
    axes[0].set_xlabel("2024 round"); axes[0].set_ylabel("Spearman correlation"); axes[0].set_title("Per-race ranking quality")
    axes[0].legend()

    axes[1].scatter(eval_df["positionOrder"], eval_df["predicted_position"], alpha=0.3)
    axes[1].plot([1, 20], [1, 20], "r--")
    axes[1].set_xlabel("Actual position"); axes[1].set_ylabel("Predicted position"); axes[1].set_title("Actual vs predicted position")

    plt.tight_layout()
    _save_figure(fig, output_dir, "ranking_quality_2024.png", show)
