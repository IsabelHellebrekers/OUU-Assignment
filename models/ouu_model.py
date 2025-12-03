import gurobipy as gp
from gurobipy import GRB
import numpy as np

from utils.data_utils import(
    load_generator_parameters,
    compute_emission_means,
    load_demand_forecast,
    compute_demand_residuals,
)

# Generate scenarios
def sample_demand_scenarios(forecast, residuals_by_hour, N, seed=42):
    rng = np.random.default_rng(seed)
    T = len(forecast)
    scenarios = np.zeros((N, T))

    for s in range(N):
        for t in range(T):
            error = rng.choice(residuals_by_hour[t])
            scenarios[s, t] = forecast[t] + error
            scenarios[s, t] = max(scenarios[s, t], 0)
    
    return scenarios

def build_ouu_model(N=1, seed=42, model_name="ouu"):
    generator_ids, c, k, gamma_u, gamma_d = load_generator_parameters()
    mean_emissions = compute_emission_means()
    forecast = load_demand_forecast()
    residuals_by_hour = compute_demand_residuals()

    T = len(forecast)
    J = len(generator_ids)

    demand_scen = sample_demand_scenarios(forecast, residuals_by_hour, N, seed=42)

    m = gp.Model(model_name)
    hours = range(T)
    generators = range(J)
    scenarios = range(N)

    # Decision variables

    # Capacity constraints

    # Ramp-up and ramp-down rate constraints

    # Demand constraints

    # Objective

    return m, q

def solve_ouu(N=1, seed=42):
    m, q = build_ouu_model(N=N, seed=seed)

    return q_opt, m.ObjVal
