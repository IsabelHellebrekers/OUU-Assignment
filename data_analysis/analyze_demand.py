import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_demand_history

def analyze_demand_errors(): 
    df = load_demand_history()

    df["error"] = df["Realized_residual_demand"] - df["Forecasted_residual_demand"]

    df["hour"] = df["Time"].dt.hour

    print("\n=== Basic Statistics of Demand Forecast Errors ===")
    print(df["error"].describe())

    hourly_stats = df.groupby("hour")["error"].agg(["mean", "std", "min", "max"])
    print("\n=== Error Statistics Per Hour ===")
    print(hourly_stats)

    pivot = df.pivot_table(index=df["Time"].dt.date,
                           columns="hour",
                           values="error")
    
    cov_matrix = pivot.cov()

    print("\n=== Covariance Matrix of Hourly Errors ===")
    print(cov_matrix)

    cov_matrix.to_csv("results/demand_covariance.csv")
    hourly_stats.to_csv("results/demand_hourly_stats.csv")

    plt.figure(figsize=(8,5))
    sns.histplot(df["error"], kde=True, bins=50)
    plt.title("Distribution of Demand Forecast Errors")
    plt.xlabel("Forecast Error (MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figures/demand_error_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, cmap="viridis")
    plt.title("Covariance Matrix of Hourly Demand Forecast Errors")
    plt.tight_layout()
    plt.savefig("figures/demand_error_heatmap.png")
    plt.close()

if __name__ == "__main__":
    analyze_demand_errors()