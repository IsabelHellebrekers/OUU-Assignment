import numpy as np
import pandas as pd 
from pathlib import Path

from data_utils import (
    load_demand_forecast, 
    compute_daily_errors,
)

this_dir = Path(__file__).resolve().parent
root_dir = this_dir.parent[1]
data_dir = root_dir/"data"
out_dir = root_dir/"results"

def compute_emission_stats():
    df = pd.read_csv(data_dir/"Emission_2020_2022.csv", sep=";")
    gen_cols = [col for col in df.columns if col.startswith("Emission_rate_g")]
    gen_cols = sorted(gen_cols, key=lambda s: int(s.split("g")[1]))

    mu = df[gen_cols].mean(axis=0).to_numpy(dtype=float)
    sigma = df[gen_cols].std(axis=0, ddof=1).to_numpy(dtype=float)

    return mu, sigma

def simulate_test_samples(N_test=2000, seed=99):
    """
    Simulate N_test for out-of-sample validation

    for each scenario s: 
        demand_s[s, t] : hourly residual demand (bootstrap)
    """





