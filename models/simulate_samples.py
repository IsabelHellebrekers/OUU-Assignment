import numpy as np
import pandas as pd 
from pathlib import Path

from data_utils import (
    load_demand_forecast, 
    compute_daily_errors,
)

data_dir = Path(__file__).resolve().parents[1]/"data"
out_dir = Path(__file__).resolve().parents[1]/"results"
sample_dir = Path(__file__).resolve().parents[1]/"test_sample"

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
    """

    rng = np.random.default_rng(seed)

    forecast = load_demand_forecast()
    daily_errors, dates = compute_daily_errors()

    D, T = daily_errors.shape
    assert T == len(forecast)

    demand_test = np.zeros((N_test, T))

    for s in range(N_test):
        d_idx = rng.integers(0,D)
        errors = daily_errors[d_idx, :]
        demand_test[s, :] = np.maximum(forecast + errors, 0.0)
    
    mu, sigma = compute_emission_stats()
    J = len(mu)

    emission_test = rng.normal(loc=mu, scale=sigma, size=(N_test, J))

    sample_dir.mkdir(exist_ok=True)

    np.save(sample_dir/"test_demand.npy", demand_test)
    np.save(sample_dir/"test_emissions.npy", emission_test)

    pd.DataFrame(demand_test).to_csv(sample_dir/ "test_demand.csv", index=False)
    pd.DataFrame(emission_test).to_csv(sample_dir/"test_emissions.csv", index=False)

    print(f"Saved test samples to {sample_dir}")
    return demand_test, emission_test

if __name__ == "__main__":
    simulate_test_samples(N_test=2000, seed=99)



