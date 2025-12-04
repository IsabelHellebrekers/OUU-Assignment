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

def compute_daily_errors(cutoff_date="2022-12-09"):
    df = pd.read_csv(data_dir/"Demand_2020_2022.csv", sep=";")
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True)

    df["error"] = (
        df["Realized_residual_demand"]
        - df["Forecasted_residual_demand"]
    )

    df["date"] = df["Time"].dt.date
    df["hour"] = df["Time"].dt.hour

    cutoff = pd.to_datetime(cutoff_date).date()
    hist = df[df["date"] < cutoff].copy()

    grouped = hist.groupby("date")

    daily_errors = []
    valid_dates = []

    for d, g in grouped:
        if g["hour"].nunique() == 24:
            g_sorted = g.sort_values("hour")
            daily_errors.append(g_sorted["error"].to_numpy(dtype=float))
            valid_dates.append(d)

    daily_errors = np.vstack(daily_errors)

    return daily_errors, valid_dates

def save_solution_csv(results_dir, prefix, obj_val, q_opt, prices, generator_ids):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    df_obj = pd.DataFrame({"objective": [obj_val]})
    df_obj.to_csv(results_dir / f"{prefix}_objective.csv", index=False)

    T, J = q_opt.shape
    df_q = pd.DataFrame(q_opt, 
                        index=np.arange(T),
                        columns=generator_ids)
    df_q.index.name = "hour"
    df_q.to_csv(results_dir / f"{prefix}_q.csv")

    df_p = pd.DataFrame({
        "hour": np.arange(T),
        "price": prices
    })
    df_p.to_csv(results_dir / f"{prefix}_prices.csv", index=False)

    print(f"Saved: {prefix}_objective.csv, {prefix}_q.csv, {prefix}_prices.csv")