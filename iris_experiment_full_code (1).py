"""
Experiment No. 2
Program to Compute Descriptive Statistics (Mean, Mode, Median, Variance,
Standard Deviation, Quartiles, and Interquartile Range) from a Given Dataset

Dataset used: Iris Flower Dataset -> column: petal_length
Data is loaded from a CSV file (not hardcoded), as required by the experiment.
"""

# ---------------------------------------------------------
# STEP 1: Import required libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# STEP 2: Load the dataset
# ---------------------------------------------------------
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
print("First 5 rows of the dataset:")
print(iris.head())

# ---------------------------------------------------------
# STEP 3: Explore basic information
# ---------------------------------------------------------
print("\nShape of dataset (rows, columns):", iris.shape)
print("\nData types:")
print(iris.dtypes)
print("\nSummary statistics (built-in pandas describe):")
print(iris.describe())

# ---------------------------------------------------------
# STEP 4: Select the numerical column and drop missing values
# ---------------------------------------------------------
data = iris["petal_length"].dropna()
print(f"\nColumn selected: petal_length  |  N = {len(data)}")

# ---------------------------------------------------------
# STEP 5: Central tendency - Mean, Median, Mode
# ---------------------------------------------------------
mean_val = data.mean()
median_val = data.median()
mode_result = stats.mode(data, keepdims=True)
mode_val = mode_result.mode[0]
mode_count = mode_result.count[0]

print("\n--- Central Tendency ---")
print(f"Mean   : {mean_val:.4f}")
print(f"Median : {median_val:.4f}")
print(f"Mode   : {mode_val:.4f}  (occurs {mode_count} times)")

# ---------------------------------------------------------
# STEP 6: Dispersion - Variance and Standard Deviation
# ---------------------------------------------------------
sample_var = data.var(ddof=1)   # sample variance (N-1 denominator)
sample_std = data.std(ddof=1)   # sample standard deviation
pop_var = data.var(ddof=0)      # population variance (N denominator)
pop_std = data.std(ddof=0)      # population standard deviation

print("\n--- Dispersion ---")
print(f"Sample Variance     : {sample_var:.4f}")
print(f"Sample Std Dev      : {sample_std:.4f}")
print(f"Population Variance : {pop_var:.4f}")
print(f"Population Std Dev  : {pop_std:.4f}")

# ---------------------------------------------------------
# STEP 7: Quartiles and Interquartile Range (IQR)
# ---------------------------------------------------------
q1 = data.quantile(0.25)
q2 = data.quantile(0.50)   # same as median
q3 = data.quantile(0.75)
iqr = q3 - q1

print("\n--- Quartiles ---")
print(f"Q1 (25th percentile) : {q1:.4f}")
print(f"Q2 (Median, 50th)    : {q2:.4f}")
print(f"Q3 (75th percentile) : {q3:.4f}")
print(f"IQR (Q3 - Q1)        : {iqr:.4f}")

# ---------------------------------------------------------
# STEP 8: Outlier detection using the IQR rule
# ---------------------------------------------------------
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = data[(data < lower_fence) | (data > upper_fence)]

print("\n--- Outlier Detection (IQR rule) ---")
print(f"Lower fence : {lower_fence:.4f}")
print(f"Upper fence : {upper_fence:.4f}")
print(f"Number of outliers detected : {len(outliers)}")
if len(outliers) > 0:
    print("Outlier values:", outliers.tolist())

# ---------------------------------------------------------
# STEP 9: Skewness (helps interpret the distribution shape)
# ---------------------------------------------------------
skewness = stats.skew(data)
shape = ("approximately symmetric" if abs(skewness) < 0.5
          else "right/positive skew" if skewness > 0
          else "left/negative skew")
print(f"\nSkewness : {skewness:.4f}  ({shape})")

# ---------------------------------------------------------
# STEP 10: Visualization - Histogram and Boxplot
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(data, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
axes[0].axvline(median_val, color="green", linestyle="-", linewidth=2, label=f"Median = {median_val:.2f}")
axes[0].axvline(q1, color="orange", linestyle=":", linewidth=1.5, label=f"Q1 = {q1:.2f}")
axes[0].axvline(q3, color="purple", linestyle=":", linewidth=1.5, label=f"Q3 = {q3:.2f}")
axes[0].set_title("Histogram of Petal Length")
axes[0].set_xlabel("petal_length")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=8)

axes[1].boxplot(data, patch_artist=True,
                boxprops=dict(facecolor="#DD8452", alpha=0.7),
                medianprops=dict(color="black", linewidth=2))
axes[1].set_title("Boxplot of Petal Length")
axes[1].set_ylabel("petal_length")

plt.tight_layout()
plt.savefig("iris_petal_length.png", dpi=150)
plt.show()
print("\nPlot saved as iris_petal_length.png")

# ---------------------------------------------------------
# STEP 11: Compare statistics before and after removing outliers
# ---------------------------------------------------------
cleaned = data[(data >= lower_fence) & (data <= upper_fence)]

print("\n--- Before vs After Outlier Removal ---")
print(f"Before -> N: {len(data)}, Mean: {data.mean():.4f}, Std Dev: {data.std():.4f}")
print(f"After  -> N: {len(cleaned)}, Mean: {cleaned.mean():.4f}, Std Dev: {cleaned.std():.4f}")

# ---------------------------------------------------------
# STEP 12: Save summary results to a CSV file
# ---------------------------------------------------------
summary = pd.DataFrame([{
    "Dataset": "Iris",
    "Column": "petal_length",
    "N": len(data),
    "Mean": round(mean_val, 4),
    "Median": round(median_val, 4),
    "Mode": round(mode_val, 4),
    "Sample Variance": round(sample_var, 4),
    "Sample Std Dev": round(sample_std, 4),
    "Population Variance": round(pop_var, 4),
    "Population Std Dev": round(pop_std, 4),
    "Q1": round(q1, 4),
    "Q2_Median": round(q2, 4),
    "Q3": round(q3, 4),
    "IQR": round(iqr, 4),
    "Skewness": round(skewness, 4),
    "Outliers": len(outliers),
}])
summary.to_csv("iris_summary_statistics.csv", index=False)

print("\n--- Final Summary Table ---")
print(summary.to_string(index=False))
print("\nSummary saved as iris_summary_statistics.csv")
