import gurobipy as gp
from gurobipy import GRB
import numpy as np

from data_utils import(
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

def build_ouu_model(N, seed=42, model_name="ouu"):
    generator_ids, c, k, gamma_u, gamma_d = load_generator_parameters()
    mean_emissions = compute_emission_means()
    forecast = load_demand_forecast()
    residuals_by_hour = compute_demand_residuals()

    T = len(forecast)
    J = len(generator_ids)

    demand_scen = sample_demand_scenarios(forecast, residuals_by_hour, N, seed=seed)

    m = gp.Model(model_name)
    hours = range(T)
    generators = range(J)
    scenarios = range(N)

    # Decision variables
    q = m.addVars(hours, generators, lb=0.0, name="q")

    # Capacity constraints
    for j in generators:
        for t in hours:
            q[t, j].ub = float(k[j])

    # Ramp-up and ramp-down rate constraints
    for j in generators:
        for t in range(T - 1):
            m.addConstr(q[t + 1, j] - q[t, j] <= gamma_u[j] * k[j],
                        name=f"ramp_up_{t}_{j}")
            m.addConstr(q[t + 1, j] - q[t, j] >= gamma_d[j] * k[j],
                        name=f"ramp_down_{t}_{j}")

    # Demand constraints
    for s in scenarios:
        for t in hours:
            m.addConstr(
                gp.quicksum(q[t, j] for j in generators) >= demand_scen[s, t],
                name=f"demand_{s}_{t}"
            )

    # Objective
    cost_term = gp.quicksum(c[j] * q[t, j] for t in hours for j in generators)
    emission_term = gp.quicksum(mean_emissions[j] * q[t, j] for t in hours for j in generators)
    m.setObjective(0.65 * cost_term + 0.35 * emission_term, GRB.MINIMIZE)

    info = {
        "generator_ids": generator_ids,
        "c": c,
        "k": k,
        "gamma_u": gamma_u,
        "gamma_d": gamma_d,
        "forecast": forecast,
        "mean_emissions": mean_emissions,
        "residuals_by_hour": residuals_by_hour,
        "demand_scenarios": demand_scen
    }

    return m, q, info

def solve_ouu(N, seed=42):
    m, q, info = build_ouu_model(N=N, seed=42)

    m.Params.OutputFlag = 1
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimization failed (status {m.Status})")
        
    T = len(info["forecast"])
    J = len(info["generator_ids"])

    q_opt = np.zeros((T, J))
    for t in range(T): 
        for j in range(J):
            q_opt[t, j] = q[t, j].X

    return q_opt, m.ObjVal, info

if __name__ == "__main__":
    N = 3000
    seed = 42
    
    print(f"Running OUU model with N={N}...")

    q_opt, obj_val, info = solve_ouu(N=N, seed=seed)

    print("\n==== OUU objective value ====")
    print(obj_val)

    print("\n ==== Optimal production per hour (MW) ====")
    generator_ids = info["generator_ids"]

    T, J = q_opt.shape
    for t in range(T):
        print(f"\nHour {t}:")
        for j in range(J):
            print(f" Generator {generator_ids[j]}: {q_opt[t, j]:.2f}")
        print(f" Total: {q_opt[t, :].sum():.2f}")
    
