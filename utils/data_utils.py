import pandas as pd 
import numpy as np
from pathlib import Path

data_dir = Path(__file__).resolve().parents[1]/"data"

def load_generator_parameters():
    """
    Columns in csv: 'generator_index', 'type', 'max_capacity', 
    'ramp_up', 'ramp_down', 'marginal_cost'
    """

    df = pd.read_csv(data_dir/"Parameters_generators_NL.csv", sep=";")
    df = df.sort_values("generator_index")

    generator_ids = df["generator_index"].astype(int).tolist()
    k = df["max_capacity"].to_numpy(dtype=float)
    gamma_u = df["ramp_up"].to_numpy(dtype=float)
    gamma_d = df["ramp_down"].to_numpy(dtype=float)
    c = df["marginal_cost"].to_numpy(dtype=float)

    return generator_ids, c, k, gamma_u, gamma_d

def compute_emission_means():
    """
    Columns in csv: 'Time', 'Emission_rate_g1', ..., 'Emission_rate_g22'

    Returns mu_emission array
        mean_emission[j - 1] = mean Emission_rate_g{j}
    """

    df = pd.read_csv(data_dir/"Emission_2020_2022.csv", sep=";")

    generator_cols = [f"Emission_rate_g{i}" for i in range(1,23)]
    mean_emission = df[generator_cols].mean(axis=0).to_numpy(dtype=float)

    return mean_emission

def load_demand_forecast():
    df = pd.read_csv(data_dir/"Demand_forecast_20221209.csv", sep=";")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True)
    df = df.sort_values("Time")
     
    forecast = df["Forecasted_residual_demand"].to_numpy(dtype=float)
    return forecast

def compute_demand_residuals(cutoff_date="2022-12-09"):
    """
    error = realized_residual_demand - forecasted_residual_demand
    """
    df = pd.read_csv(data_dir/"Demand_2020_2022.csv", sep=";")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True)

    cutoff_ts = pd.to_datetime(cutoff_date)
    history = df[df["Time"] < cutoff_ts].copy()

    history["error"] = (
        history["Realized_residual_demand"]
        - history["Forecasted_residual_demand"]
    )

    history["hour"] = history["Time"].dt.hour

    residuals_by_hour = []
    for h in range(24):
        values = history.loc[history["hour"] == h, "error"].to_numpy(dtype=float)
        residuals_by_hour.append(values)

    return residuals_by_hour