"""
Experiment No. 2
Program to Compute Descriptive Statistics (Mean, Mode, Median, Variance,
Standard Deviation, Quartiles, and Interquartile Range) from a Given Dataset

Datasets used:
  1. Iris Flower Dataset      -> column: petal_length
  2. Titanic Passenger Dataset -> column: fare

Data is loaded from CSV files (not hardcoded), as required by the experiment.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def describe_column(df: pd.DataFrame, column: str, dataset_name: str) -> dict:
    """Compute mean, median, mode, variance, std, quartiles, IQR, and outliers
    for a single numerical column of a dataframe. Missing values are dropped
    before computation and reported."""

    raw = df[column]
    n_missing = raw.isna().sum()
    data = raw.dropna().astype(float)

    mean_val = data.mean()
    median_val = data.median()
    mode_result = stats.mode(data, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]

    # Sample variance/std (ddof=1) -- appropriate since this is a sample,
    # not a full population.
    sample_var = data.var(ddof=1)
    sample_std = data.std(ddof=1)

    # Population variance/std (ddof=0), shown for comparison/viva purposes.
    pop_var = data.var(ddof=0)
    pop_std = data.std(ddof=0)

    q1 = data.quantile(0.25)
    q2 = data.quantile(0.50)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]

    skewness = stats.skew(data)

    results = {
        "dataset": dataset_name,
        "column": column,
        "n": len(data),
        "n_missing": int(n_missing),
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "mode_count": int(mode_count),
        "sample_variance": sample_var,
        "sample_std": sample_std,
        "population_variance": pop_var,
        "population_std": pop_std,
        "Q1": q1,
        "Q2_median": q2,
        "Q3": q3,
        "IQR": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_outliers": len(outliers),
        "outlier_values": outliers.tolist(),
        "skewness": skewness,
        "clean_data": data,  # kept for plotting / outlier removal step
    }
    return results


def print_report(results: dict) -> None:
    print("=" * 70)
    print(f"Dataset: {results['dataset']}   |   Column: {results['column']}")
    print("=" * 70)
    print(f"Count (n)                 : {results['n']}  (missing values dropped: {results['n_missing']})")
    print(f"Mean                      : {results['mean']:.4f}")
    print(f"Median                    : {results['median']:.4f}")
    print(f"Mode                      : {results['mode']:.4f}  (occurs {results['mode_count']} times)")
    print(f"Sample Variance (ddof=1)  : {results['sample_variance']:.4f}")
    print(f"Sample Std Dev (ddof=1)   : {results['sample_std']:.4f}")
    print(f"Population Variance(ddof=0): {results['population_variance']:.4f}")
    print(f"Population Std Dev(ddof=0): {results['population_std']:.4f}")
    print(f"Q1 (25th percentile)      : {results['Q1']:.4f}")
    print(f"Q2 / Median (50th)        : {results['Q2_median']:.4f}")
    print(f"Q3 (75th percentile)      : {results['Q3']:.4f}")
    print(f"IQR (Q3 - Q1)             : {results['IQR']:.4f}")
    print(f"Outlier fences            : [{results['lower_fence']:.4f}, {results['upper_fence']:.4f}]")
    print(f"Number of outliers        : {results['n_outliers']}")
    if results["n_outliers"] > 0:
        shown = results["outlier_values"][:10]
        print(f"Outlier values (first 10) : {[round(v, 2) for v in shown]}")
    print(f"Skewness                  : {results['skewness']:.4f}  "
          f"({'right/positive skew' if results['skewness'] > 0.5 else 'left/negative skew' if results['skewness'] < -0.5 else 'approximately symmetric'})")
    print()


def plot_distribution(results: dict, outpath_prefix: str) -> None:
    """Save a histogram (with mean/median/quartile lines) and a boxplot
    for the given column."""
    data = results["clean_data"]
    dataset = results["dataset"]
    column = results["column"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(data, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].axvline(results["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean = {results['mean']:.2f}")
    axes[0].axvline(results["median"], color="green", linestyle="-", linewidth=2, label=f"Median = {results['median']:.2f}")
    axes[0].axvline(results["Q1"], color="orange", linestyle=":", linewidth=1.5, label=f"Q1 = {results['Q1']:.2f}")
    axes[0].axvline(results["Q3"], color="purple", linestyle=":", linewidth=1.5, label=f"Q3 = {results['Q3']:.2f}")
    axes[0].set_title(f"{dataset}: Histogram of {column}")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=8)

    # Boxplot
    axes[1].boxplot(data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#DD8452", alpha=0.7),
                     medianprops=dict(color="black", linewidth=2))
    axes[1].set_title(f"{dataset}: Boxplot of {column}")
    axes[1].set_ylabel(column)

    plt.tight_layout()
    outfile = f"{outpath_prefix}.png"
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {outfile}")


def plot_before_after_outlier_removal(results: dict, outpath_prefix: str) -> dict:
    """Compare mean/std before and after removing IQR-rule outliers."""
    data = results["clean_data"]
    lower, upper = results["lower_fence"], results["upper_fence"]
    cleaned = data[(data >= lower) & (data <= upper)]

    before_after = {
        "mean_before": data.mean(),
        "mean_after": cleaned.mean(),
        "std_before": data.std(ddof=1),
        "std_after": cleaned.std(ddof=1),
        "n_before": len(data),
        "n_after": len(cleaned),
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([data, cleaned], labels=["Before removal", "After removal"], patch_artist=True,
               boxprops=dict(facecolor="#55A868", alpha=0.7))
    ax.set_title(f"{results['dataset']}: {results['column']} - Outlier Removal Effect")
    ax.set_ylabel(results["column"])
    plt.tight_layout()
    outfile = f"{outpath_prefix}_before_after.png"
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {outfile}")

    return before_after


def main():
    iris = pd.read_csv("iris.csv")
    titanic = pd.read_csv("titanic.csv")

    print("Dataset 1: Iris -> shape", iris.shape, ", dtypes:\n", iris.dtypes, "\n")
    print("Dataset 2: Titanic -> shape", titanic.shape, ", dtypes:\n", titanic.dtypes, "\n")

    iris_results = describe_column(iris, "petal_length", "Iris Dataset")
    titanic_results = describe_column(titanic, "fare", "Titanic Dataset")

    print_report(iris_results)
    print_report(titanic_results)

    plot_distribution(iris_results, "iris_petal_length")
    plot_distribution(titanic_results, "titanic_fare")

    iris_ba = plot_before_after_outlier_removal(iris_results, "iris_petal_length")
    titanic_ba = plot_before_after_outlier_removal(titanic_results, "titanic_fare")

    print("Before/after outlier removal (Iris petal_length):", iris_ba)
    print("Before/after outlier removal (Titanic fare)     :", titanic_ba)

    # Build a summary comparison table and save as CSV
    summary_rows = []
    for r in (iris_results, titanic_results):
        summary_rows.append({
            "Dataset": r["dataset"],
            "Column": r["column"],
            "N": r["n"],
            "Mean": round(r["mean"], 4),
            "Median": round(r["median"], 4),
            "Mode": round(r["mode"], 4),
            "Sample Variance": round(r["sample_variance"], 4),
            "Sample Std Dev": round(r["sample_std"], 4),
            "Q1": round(r["Q1"], 4),
            "Q3": round(r["Q3"], 4),
            "IQR": round(r["IQR"], 4),
            "Skewness": round(r["skewness"], 4),
            "Outliers (count)": r["n_outliers"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("summary_statistics.csv", index=False)
    print("\nSummary table saved -> summary_statistics.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
    